"""Framework-specific request adapters for authorization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import Request as FastAPIRequest

if TYPE_CHECKING:
    from mlflow_kubernetes_plugins.auth.authorizer import KubernetesAuthConfig


@dataclass(frozen=True)
class AuthorizationRequest:
    authorization_header: str | None
    forwarded_access_token: str | None
    remote_user_header_value: str | None
    remote_groups_header_value: str | None
    path: str
    method: str
    workspace: str | None
    headers: dict[str, str] = field(default_factory=dict)
    path_params: dict[str, object] = field(default_factory=dict)
    query_params: dict[str, object] = field(default_factory=dict)
    json_body: object | None = None
    graphql_payload: dict[str, object] | None = None
    ensure_json_body: Callable[[], Awaitable[object]] | None = None
    """Async callback that loads the request body and returns the parsed JSON
    payload (or ``None``).  Used for lazy body loading in the ASGI middleware
    path where the body stream is not synchronously available at
    request-construction time."""


def _collect_query_params(items: list[tuple[str, str]]) -> dict[str, object]:
    """Collapse multi-valued query parameters into a ``{key: value | [values]}`` dict."""
    params: dict[str, object] = {}
    for key, value in items:
        current = params.get(key)
        if current is None:
            params[key] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            params[key] = [current, value]
    return params


def _collect_headers(headers: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _first_value(mapping: dict[str, object], key: str) -> str | None:
    """Return the first non-empty string for *key*, collapsing lists to their first element."""
    value = mapping.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) and value else None


def _build_graphql_payload(
    path: str,
    *,
    json_body: object | None,
    query_params: dict[str, object],
) -> dict[str, object] | None:
    """Build the GraphQL operation payload from the request body or query string.

    POST requests typically send the payload as a JSON body; GET requests encode
    ``query``, ``operationName``, and ``variables`` as query-string parameters
    (per the GraphQL-over-HTTP spec).  Returns ``None`` for non-GraphQL paths.
    """
    if not path.endswith("/graphql"):
        return None
    if isinstance(json_body, dict):
        return json_body

    query = _first_value(query_params, "query")
    operation_name = _first_value(query_params, "operationName")
    variables_raw = _first_value(query_params, "variables")
    variables: dict[str, object] | None = None
    if variables_raw:
        try:
            parsed_variables = json.loads(variables_raw)
        except json.JSONDecodeError:
            parsed_variables = None
        if isinstance(parsed_variables, dict):
            variables = parsed_variables

    if not query and not operation_name and variables is None:
        return None

    payload: dict[str, object] = {}
    if query:
        payload["query"] = query
    if operation_name:
        payload["operationName"] = operation_name
    if variables is not None:
        payload["variables"] = variables
    return payload or None


def build_fastapi_authorization_request(
    request: FastAPIRequest,
    config_values: "KubernetesAuthConfig",
    *,
    path: str,
    workspace: str | None,
    json_body: object | None = None,
    path_params: dict[str, object] | None = None,
    ensure_json_body: Callable[[], Awaitable[object]] | None = None,
) -> AuthorizationRequest:
    # FastAPI/ASGI requests do not populate Flask globals, so the caller passes any pre-parsed
    # body and canonicalized path params in explicitly.
    query_params = _collect_query_params(list(request.query_params.multi_items()))
    return AuthorizationRequest(
        authorization_header=request.headers.get("Authorization"),
        forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
        remote_user_header_value=request.headers.get(config_values.user_header),
        remote_groups_header_value=request.headers.get(config_values.groups_header),
        path=path,
        method=request.method,
        workspace=workspace,
        headers=_collect_headers(request.headers),
        path_params=path_params or dict(request.path_params),
        query_params=query_params,
        json_body=json_body,
        graphql_payload=_build_graphql_payload(
            path,
            json_body=json_body,
            query_params=query_params,
        ),
        ensure_json_body=ensure_json_body,
    )
