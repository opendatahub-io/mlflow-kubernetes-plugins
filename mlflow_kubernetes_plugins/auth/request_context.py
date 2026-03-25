"""Framework-specific request adapters for authorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request as FastAPIRequest
from flask import request as flask_request

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
    graphql_payload: dict[str, object] | None = None


def build_flask_authorization_request(
    config_values: "KubernetesAuthConfig",
    *,
    path: str,
    workspace: str | None,
) -> AuthorizationRequest:
    graphql_payload: dict[str, object] | None = None
    if path.endswith("/graphql"):
        try:
            payload = flask_request.get_json(silent=True) or {}
            if isinstance(payload, dict):
                graphql_payload = payload
        except Exception:
            graphql_payload = None

    return AuthorizationRequest(
        authorization_header=flask_request.headers.get("Authorization"),
        forwarded_access_token=flask_request.headers.get("X-Forwarded-Access-Token"),
        remote_user_header_value=flask_request.headers.get(config_values.user_header),
        remote_groups_header_value=flask_request.headers.get(config_values.groups_header),
        path=path,
        method=flask_request.method,
        workspace=workspace,
        graphql_payload=graphql_payload,
    )


def build_fastapi_authorization_request(
    request: FastAPIRequest,
    config_values: "KubernetesAuthConfig",
    *,
    path: str,
    workspace: str | None,
) -> AuthorizationRequest:
    return AuthorizationRequest(
        authorization_header=request.headers.get("Authorization"),
        forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
        remote_user_header_value=request.headers.get(config_values.user_header),
        remote_groups_header_value=request.headers.get(config_values.groups_header),
        path=path,
        method=request.method,
        workspace=workspace,
    )
