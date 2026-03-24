"""Flask and FastAPI middleware wiring for the Kubernetes auth plugin."""

from __future__ import annotations

import io
import json
import logging

from fastapi import Request
from flask import Flask, Response, g, request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def _auth_module():
    import mlflow_kubernetes_plugins.auth as auth_mod

    return auth_mod


class KubernetesAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Kubernetes-based authorization."""

    def __init__(self, app, authorizer, config_values):
        super().__init__(app)
        self.authorizer = authorizer
        self.config_values = config_values

    async def dispatch(self, request: Request, call_next):
        """Process each request through the authorization pipeline."""
        auth_mod = _auth_module()
        authorization_token = auth_mod._AUTHORIZATION_HANDLED.set(None)
        try:
            canonical_path = auth_mod._canonicalize_path(
                raw_path=str(request.url.path or ""),
                scope_path=request.scope.get("path"),
                root_path=request.scope.get("root_path"),
            )
            fastapi_app = request.scope.get("app")
            if fastapi_app is None:
                exc = auth_mod.MlflowException(
                    "FastAPI app missing from request scope.",
                    error_code=auth_mod.databricks_pb2.INTERNAL_ERROR,
                )
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )

            # Skip authentication for unprotected paths
            if auth_mod._is_unprotected_path(canonical_path):
                return await call_next(request)

            workspace_name = auth_mod.workspace_context.get_request_workspace()
            workspace_set = False

            if workspace_name is None:
                # FastAPI executes middlewares in reverse order, so this auth middleware can run
                # before the MLflow workspace middleware. Resolve here using the same helper, which
                # also falls back to the configured default workspace when the header is missing
                # or empty.
                try:
                    workspace = auth_mod.resolve_workspace_from_header(
                        request.headers.get(auth_mod.WORKSPACE_HEADER_NAME)
                    )
                except auth_mod.MlflowException as exc:
                    return JSONResponse(
                        status_code=exc.get_http_status_code(),
                        content=json.loads(exc.serialize_as_json()),
                    )

                if workspace is not None:
                    workspace_name = workspace.name
                    auth_mod.workspace_context.set_server_request_workspace(workspace_name)
                    workspace_set = True

            if canonical_path.endswith("/graphql"):
                # Let Flask authorize GraphQL to avoid consuming/rebuffering the ASGI body.
                try:
                    return await call_next(request)
                finally:
                    if workspace_set:
                        auth_mod.workspace_context.clear_server_request_workspace()

            try:
                auth_result = auth_mod._authorize_request_context(
                    auth_mod.build_fastapi_authorization_request(
                        request,
                        self.config_values,
                        path=canonical_path,
                        workspace=workspace_name,
                    ),
                    authorizer=self.authorizer,
                    config_values=self.config_values,
                )
                auth_mod._AUTHORIZATION_HANDLED.set(auth_result)
            except auth_mod.MlflowException as exc:
                if workspace_set:
                    auth_mod.workspace_context.clear_server_request_workspace()
                if (
                    workspace_name is None
                    and exc.error_code
                    == auth_mod.databricks_pb2.ErrorCode.Name(
                        auth_mod.databricks_pb2.INVALID_PARAMETER_VALUE
                    )
                    and exc.message == auth_mod._WORKSPACE_REQUIRED_ERROR_MESSAGE
                ):
                    exc = auth_mod.MlflowException(
                        auth_mod._WORKSPACE_REQUIRED_ERROR_MESSAGE,
                        error_code=auth_mod.databricks_pb2.INTERNAL_ERROR,
                    )
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )

            # Continue with the request, clearing any temporary workspace context.
            try:
                response = await call_next(request)
            finally:
                if workspace_set:
                    auth_mod.workspace_context.clear_server_request_workspace()
            return response
        finally:
            auth_mod._AUTHORIZATION_HANDLED.reset(authorization_token)


def _override_run_user(username: str) -> None:
    """Rewrite the request payload so MLflow sees the authenticated user as run owner."""
    if not request.mimetype or "json" not in request.mimetype.lower():
        return

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return

    payload["user_id"] = username
    data = json.dumps(payload).encode("utf-8")

    # Reset cached JSON with proper structure expected by Werkzeug
    # The keys are boolean values for the 'silent' parameter
    request._cached_json = {True: payload, False: payload}  # type: ignore[attr-defined]
    request._cached_data = data  # type: ignore[attr-defined]
    request.environ["wsgi.input"] = io.BytesIO(data)
    request.environ["CONTENT_LENGTH"] = str(len(data))
    request.environ["CONTENT_TYPE"] = "application/json"


def _record_authorization_metadata(auth_result) -> None:
    """Record auth metadata on the Flask request context."""
    primary_rule = auth_result.rules[0]
    if auth_result.username and primary_rule.override_run_user:
        _override_run_user(auth_result.username)

    g.mlflow_k8s_identity = auth_result.identity
    g.mlflow_k8s_apply_workspace_filter = primary_rule.apply_workspace_filter


def create_app(app: Flask | None = None):
    """Enable Kubernetes-based authorization for the MLflow tracking server."""
    auth_mod = _auth_module()
    if app is None:
        app = auth_mod.mlflow_app

    parent_logger = getattr(app, "logger", logging.getLogger("mlflow"))
    auth_mod._logger = parent_logger
    auth_mod._logger.info("Kubernetes authorization plugin initialized")

    config_values = auth_mod.KubernetesAuthConfig.from_env()
    authorizer = auth_mod.KubernetesAuthorizer(config_values=config_values)

    auth_mod._compile_authorization_rules()

    @app.before_request
    def _k8s_auth_before_request():
        auth_result = auth_mod._AUTHORIZATION_HANDLED.get()
        if auth_result is not None:
            _record_authorization_metadata(auth_result)
            return None

        canonical_path = auth_mod._canonicalize_path(
            raw_path=request.path or "",
            path_info=request.environ.get("PATH_INFO"),
            script_name=request.environ.get("SCRIPT_NAME"),
        )
        if auth_mod._is_unprotected_path(canonical_path):
            return None

        try:
            auth_result = auth_mod._authorize_request_context(
                auth_mod.build_flask_authorization_request(
                    config_values,
                    path=canonical_path,
                    workspace=auth_mod.workspace_context.get_request_workspace(),
                ),
                authorizer=authorizer,
                config_values=config_values,
            )
        except auth_mod.MlflowException as exc:
            response = Response(mimetype="application/json")
            response.set_data(exc.serialize_as_json())
            response.status_code = exc.get_http_status_code()
            return response

        # Use the first rule for metadata - these properties are consistent across all rules
        _record_authorization_metadata(auth_result)

        return None

    @app.after_request
    def _k8s_auth_after_request(response: Response):
        try:
            should_filter = getattr(g, "mlflow_k8s_apply_workspace_filter", False)
            identity = getattr(g, "mlflow_k8s_identity", None)
            can_filter_response = response.mimetype == "application/json" and response.status_code < 400
            if not (should_filter and identity and can_filter_response):
                return response

            try:
                payload = json.loads(response.get_data(as_text=True))
            except Exception:
                payload = None

            if not isinstance(payload, dict):
                return response

            workspaces = payload.get("workspaces")
            if not isinstance(workspaces, list):
                return response

            workspace_names = [ws.get("name") for ws in workspaces if isinstance(ws, dict)]
            accessible = authorizer.accessible_workspaces(
                identity, [name for name in workspace_names if isinstance(name, str)]
            )
            payload["workspaces"] = [
                ws for ws in workspaces if isinstance(ws, dict) and ws.get("name") in accessible
            ]
            response.set_data(json.dumps(payload))
            response.headers["Content-Length"] = str(len(response.get_data()))
        finally:
            for attr in (
                "mlflow_k8s_identity",
                "mlflow_k8s_apply_workspace_filter",
            ):
                if hasattr(g, attr):
                    delattr(g, attr)
            auth_mod._AUTHORIZATION_HANDLED.set(None)

        return response

    if auth_mod._MLFLOW_SGI_NAME.get() == "uvicorn":
        fastapi_app = auth_mod.create_fastapi_app(app)

        # Add Kubernetes auth middleware to FastAPI
        # Important: This must be added AFTER security middleware but BEFORE routes
        # to ensure proper middleware ordering
        #
        # Note: The KubernetesAuthMiddleware runs for all requests when the Flask app
        # is mounted under FastAPI. It sets a context variable so the Flask auth hooks
        # can skip duplicate authorization work when FastAPI already handled it.
        fastapi_app.add_middleware(
            KubernetesAuthMiddleware,
            authorizer=authorizer,
            config_values=config_values,
        )
        auth_mod._validate_fastapi_route_authorization(fastapi_app)
        auth_mod._validate_graphql_field_authorization()
        return fastapi_app
    return app


__all__ = ["KubernetesAuthMiddleware", "_override_run_user", "create_app"]
