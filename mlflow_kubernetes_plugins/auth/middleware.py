"""FastAPI middleware wiring for the Kubernetes auth plugin."""

from __future__ import annotations

import atexit
import copy
import json
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from fastapi import Request
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server import app as mlflow_app
from mlflow.server import handlers as mlflow_handlers
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.workspace_helpers import WORKSPACE_HEADER_NAME, resolve_workspace_from_header
from mlflow.utils import workspace_context
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import QueryParams
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import Scope

import mlflow_kubernetes_plugins.auth.core as core_mod
from mlflow_kubernetes_plugins.auth.authorizer import KubernetesAuthConfig, KubernetesAuthorizer
from mlflow_kubernetes_plugins.auth.collection_filters import (
    apply_response_collection_filters,
    can_skip_response_collection_filters,
)
from mlflow_kubernetes_plugins.auth.compiler import (
    _compile_authorization_rules,
    _extract_path_params,
    _validate_fastapi_route_authorization,
)
from mlflow_kubernetes_plugins.auth.core import (
    _AUTHORIZATION_HANDLED,
    _authorize_request_async,
    _canonicalize_path,
    _is_unprotected_path,
)
from mlflow_kubernetes_plugins.auth.graphql import (
    get_graphql_authorization_middleware as _get_graphql_authorization_middleware,
)
from mlflow_kubernetes_plugins.auth.graphql import (
    validate_graphql_field_authorization as _validate_graphql_field_authorization,
)
from mlflow_kubernetes_plugins.auth.request_context import build_fastapi_authorization_request
from mlflow_kubernetes_plugins.auth.resource_names import apply_response_cache_updates

if TYPE_CHECKING:
    from flask import Flask

_REQUEST_RAW_BODY_STATE_KEY = "mlflow_k8s_raw_body"
_REQUEST_JSON_BODY_STATE_KEY = "mlflow_k8s_json_body"
_REQUEST_BODY_LOADED_STATE_KEY = "mlflow_k8s_body_loaded"
_GRAPHQL_AUTHORIZER: ContextVar[KubernetesAuthorizer | None] = ContextVar(
    "mlflow_k8s_graphql_authorizer",
    default=None,
)

def _replace_scope_headers(scope: Scope, updates: dict[str, str]) -> None:
    """Replace (or add) ASGI scope headers, matching case-insensitively."""
    encoded = {k.lower().encode("latin-1"): v.encode("latin-1") for k, v in updates.items()}
    headers = [
        (header_name, header_value)
        for header_name, header_value in scope.get("headers", [])
        if header_name.lower() not in encoded
    ]
    headers.extend(encoded.items())
    scope["headers"] = headers


class KubernetesAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Kubernetes-based authorization."""

    def __init__(self, app, authorizer, config_values):
        super().__init__(app)
        self.authorizer = authorizer
        self.config_values = config_values

    @staticmethod
    def _set_request_json_body(request: Request, payload: dict[str, object]) -> None:
        """Overwrite the ASGI request body with *payload* and update scope state/headers."""
        raw_body = json.dumps(payload).encode("utf-8")
        state = request.scope.setdefault("state", {})
        state[_REQUEST_RAW_BODY_STATE_KEY] = raw_body
        state[_REQUEST_JSON_BODY_STATE_KEY] = payload
        state[_REQUEST_BODY_LOADED_STATE_KEY] = True
        request._body = raw_body
        _replace_scope_headers(request.scope, {
            "content-length": str(len(raw_body)),
            "content-type": "application/json",
        })

    @staticmethod
    def _set_request_query_params(request: Request, query_params: dict[str, object]) -> None:
        """Re-encode *query_params* into the ASGI scope and Starlette's cached QueryParams."""
        items: list[tuple[str, str]] = []
        for key, value in query_params.items():
            if isinstance(value, list):
                items.extend((key, str(item)) for item in value)
            elif value is not None:
                items.append((key, str(value)))
        query_string = urlencode(items, doseq=True).encode("utf-8")
        request.scope["query_string"] = query_string
        request._query_params = QueryParams(query_string)

    @staticmethod
    def _request_can_have_json_body(request: Request) -> bool:
        return request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"} and (
            "json" in request.headers.get("content-type", "").lower()
        )

    @classmethod
    async def _ensure_request_json_body(cls, request: Request) -> object | None:
        """Lazily load and cache the JSON body from the ASGI request.

        Returns the parsed payload (or ``None`` when the content-type is not JSON).
        Subsequent calls return the cached value without re-reading the body stream.
        """
        if not cls._request_can_have_json_body(request):
            return None
        state = request.scope.setdefault("state", {})
        if state.get(_REQUEST_BODY_LOADED_STATE_KEY):
            return state.get(_REQUEST_JSON_BODY_STATE_KEY)

        # BaseHTTPMiddleware caches request.body() on the outer Request and replays request._body
        # to the downstream app, so this gives us lazy loading without a second ASGI middleware.
        raw_body = bytes(await request.body())
        state[_REQUEST_RAW_BODY_STATE_KEY] = raw_body
        try:
            payload = json.loads(raw_body) if raw_body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        state[_REQUEST_JSON_BODY_STATE_KEY] = payload
        state[_REQUEST_BODY_LOADED_STATE_KEY] = True
        return payload

    @classmethod
    def _apply_request_context_rewrites(
        cls,
        request: Request,
        *,
        original_request_context,
        updated_request_context,
        username: str | None,
        override_run_user: bool,
    ) -> None:
        """Propagate authorization-side mutations back into the live ASGI request.

        This covers two cases: rewriting the JSON body (e.g. injecting ``user_id``
        for run-ownership override, or applying collection filter changes) and
        updating query parameters that were narrowed during authorization.
        The ASGI request is only mutated when a change is actually detected.
        """
        body_changed = (
            updated_request_context.json_body is not original_request_context.json_body
            and updated_request_context.json_body != original_request_context.json_body
        )
        needs_user_override = override_run_user and username
        if isinstance(updated_request_context.json_body, dict) and (body_changed or needs_user_override):
            payload = (
                copy.deepcopy(updated_request_context.json_body)
                if needs_user_override
                else updated_request_context.json_body
            )
            if needs_user_override:
                payload["user_id"] = username
            cls._set_request_json_body(request, payload)

        if updated_request_context.query_params != original_request_context.query_params:
            cls._set_request_query_params(request, updated_request_context.query_params)

    async def _filter_workspace_list_response(self, response, *, auth_result) -> None:
        """Strip workspaces the caller cannot access from a ``{"workspaces": [...]}`` response."""
        payload, _ = await self._read_json_response_payload(response)
        if not isinstance(payload, dict):
            return
        workspaces = payload.get("workspaces")
        if not isinstance(workspaces, list):
            return
        workspace_names = [ws.get("name") for ws in workspaces if isinstance(ws, dict)]
        accessible = self.authorizer.accessible_workspaces(
            auth_result.identity,
            [name for name in workspace_names if isinstance(name, str)],
        )
        filtered_workspaces = [
            ws for ws in workspaces if isinstance(ws, dict) and ws.get("name") in accessible
        ]
        if filtered_workspaces != workspaces:
            updated_payload = dict(payload)
            updated_payload["workspaces"] = filtered_workspaces
            self._replace_json_response_payload(response, updated_payload)

    async def _apply_collection_response_filters(
        self,
        response,
        *,
        auth_result,
        response_workspace_name: str | None,
    ) -> None:
        if not auth_result.response_filter_required:
            return
        if not response_workspace_name or response.status_code >= 400:
            return
        if can_skip_response_collection_filters(
            auth_result.rules,
            authorizer=self.authorizer,
            identity=auth_result.identity,
            workspace_name=response_workspace_name,
        ):
            return
        payload, _ = await self._read_json_response_payload(response)
        if not isinstance(payload, dict):
            self._replace_json_response_payload(response, {})
            return
        filtered_payload, enforceable = apply_response_collection_filters(
            payload,
            auth_result.rules,
            authorizer=self.authorizer,
            identity=auth_result.identity,
            workspace_name=response_workspace_name,
        )
        if not enforceable:
            self._replace_json_response_payload(response, {})
        elif filtered_payload != payload:
            self._replace_json_response_payload(response, filtered_payload)

    async def _apply_response_authorization_filters(self, response, *, auth_result) -> None:
        response_workspace_name = None
        if isinstance(auth_result.request_context.workspace, str):
            response_workspace_name = auth_result.request_context.workspace.strip() or None
        if auth_result.rules[0].apply_workspace_filter and response.status_code < 400:
            await self._filter_workspace_list_response(response, auth_result=auth_result)
        await self._apply_collection_response_filters(
            response,
            auth_result=auth_result,
            response_workspace_name=response_workspace_name,
        )

    @staticmethod
    async def _read_json_response_payload(response) -> tuple[dict[str, object] | None, bytes | None]:
        """Consume a streaming Starlette response body and JSON-decode it.

        The consumed bytes are re-attached to ``response.body_iterator`` so
        downstream middleware can still read them.  Returns ``(None, None)`` when
        the response is not JSON, or ``(None, raw_bytes)`` on decode failure.
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return None, None

        body = getattr(response, "body", None)
        if not isinstance(body, (bytes, bytearray)):
            collected = bytearray()
            async for chunk in response.body_iterator:
                collected.extend(chunk)
            body = bytes(collected)
            response.body_iterator = iterate_in_threadpool(iter([body]))
        else:
            body = bytes(body)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, body
        if not isinstance(payload, dict):
            return None, body
        return payload, body

    @staticmethod
    def _replace_json_response_payload(response, payload: dict[str, object]) -> None:
        """Overwrite the response body with the re-serialized *payload*."""
        updated_body = json.dumps(payload).encode("utf-8")
        response.body = updated_body
        response.body_iterator = iterate_in_threadpool(iter([updated_body]))
        response.headers["content-length"] = str(len(updated_body))

    async def dispatch(self, request: Request, call_next):
        """Process each request through the authorization pipeline."""
        authorization_token = _AUTHORIZATION_HANDLED.set(None)
        try:
            state = request.scope.setdefault("state", {})
            state.setdefault(_REQUEST_RAW_BODY_STATE_KEY, b"")
            state.setdefault(_REQUEST_JSON_BODY_STATE_KEY, None)
            state.setdefault(_REQUEST_BODY_LOADED_STATE_KEY, False)
            canonical_path = _canonicalize_path(
                raw_path=str(request.url.path or ""),
                scope_path=request.scope.get("path"),
                root_path=request.scope.get("root_path"),
            )
            fastapi_app = request.scope.get("app")
            if fastapi_app is None:
                exc = MlflowException(
                    "FastAPI app missing from request scope.",
                    error_code=databricks_pb2.INTERNAL_ERROR,
                )
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )

            # Skip authentication for unprotected paths
            if _is_unprotected_path(canonical_path):
                return await call_next(request)

            workspace_name = workspace_context.get_request_workspace()
            workspace_set = False

            if workspace_name is None:
                # FastAPI executes middlewares in reverse order, so this auth middleware can run
                # before the MLflow workspace middleware. Resolve here using the same helper, which
                # also falls back to the configured default workspace when the header is missing
                # or empty.
                try:
                    workspace = resolve_workspace_from_header(request.headers.get(WORKSPACE_HEADER_NAME))
                except MlflowException as exc:
                    return JSONResponse(
                        status_code=exc.get_http_status_code(),
                        content=json.loads(exc.serialize_as_json()),
                    )

                if workspace is not None:
                    workspace_name = workspace.name
                    workspace_context.set_server_request_workspace(workspace_name)
                    workspace_set = True

            path_params = _extract_path_params(canonical_path, request.method) or dict(
                request.path_params
            )

            async def _ensure_auth_request_json_body():
                await self._ensure_request_json_body(request)
                return state.get(_REQUEST_JSON_BODY_STATE_KEY)

            auth_request = build_fastapi_authorization_request(
                request,
                self.config_values,
                path=canonical_path,
                workspace=workspace_name,
                json_body=state.get(_REQUEST_JSON_BODY_STATE_KEY),
                path_params=path_params,
                ensure_json_body=_ensure_auth_request_json_body,
            )

            try:
                auth_result = await _authorize_request_async(
                    auth_request,
                    authorizer=self.authorizer,
                    config_values=self.config_values,
                )
                _AUTHORIZATION_HANDLED.set(auth_result)
                graphql_authorizer_token = _GRAPHQL_AUTHORIZER.set(self.authorizer)
            except MlflowException as exc:
                if workspace_set:
                    workspace_context.clear_server_request_workspace()
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )
            self._apply_request_context_rewrites(
                request,
                original_request_context=auth_request,
                updated_request_context=auth_result.request_context,
                username=auth_result.username,
                override_run_user=auth_result.rules[0].override_run_user,
            )

            # Continue with the request, clearing any temporary workspace context.
            try:
                response = await call_next(request)
            finally:
                if workspace_set:
                    workspace_context.clear_server_request_workspace()
                _GRAPHQL_AUTHORIZER.reset(graphql_authorizer_token)
            await self._apply_response_authorization_filters(response, auth_result=auth_result)
            apply_response_cache_updates(
                auth_result.request_context,
                auth_result.rules,
                status_code=response.status_code,
            )
            return response
        finally:
            _AUTHORIZATION_HANDLED.reset(authorization_token)

def _registered_graphql_auth_middleware():
    authorizer = _GRAPHQL_AUTHORIZER.get()
    if authorizer is None:
        return []
    return _get_graphql_authorization_middleware(authorizer)


def create_app(app: Flask | None = None):
    """Enable Kubernetes-based authorization for the MLflow tracking server."""
    if app is None:
        app = mlflow_app

    parent_logger = getattr(app, "logger", logging.getLogger("mlflow"))
    core_mod._logger = parent_logger
    core_mod._logger.info("Kubernetes authorization plugin initialized")

    config_values = KubernetesAuthConfig.from_env()
    authorizer = KubernetesAuthorizer(config_values=config_values)
    atexit.register(authorizer.close)
    mlflow_handlers._get_graphql_auth_middleware = _registered_graphql_auth_middleware

    _compile_authorization_rules()
    fastapi_app = create_fastapi_app(app)
    fastapi_app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=authorizer,
        config_values=config_values,
    )
    _validate_fastapi_route_authorization(fastapi_app)
    _validate_graphql_field_authorization()
    return fastapi_app


__all__ = ["KubernetesAuthMiddleware", "create_app"]
