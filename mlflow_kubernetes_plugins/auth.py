"""Kubernetes-backed authorization plugin for the MLflow tracking server."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import werkzeug
from flask import has_request_context, request
from mlflow.environment_variables import _MLFLOW_SGI_NAME  # noqa: F401
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server import app as mlflow_app  # noqa: F401
from mlflow.server.fastapi_app import create_fastapi_app  # noqa: F401
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, get_endpoints  # noqa: F401
from mlflow.server.workspace_helpers import (
    WORKSPACE_HEADER_NAME,  # noqa: F401
    resolve_workspace_from_header,  # noqa: F401
)
from mlflow.utils import workspace_context  # noqa: F401

# Re-export selected names here so tests, entrypoints, and existing import paths continue to
# resolve through `mlflow_kubernetes_plugins.auth` while the implementation is split across
# package modules.
from mlflow_kubernetes_plugins.auth.authorizer import (
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
    _AuthorizationCache,  # noqa: F401
    _CacheEntry,  # noqa: F401
    _create_api_client_for_subject_access_reviews,  # noqa: F401
    _load_kubernetes_configuration,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.compiler import (
    _compile_authorization_rules,  # noqa: F401
    _find_authorization_rules,
    _reset_compiled_rules,  # noqa: F401
    _validate_fastapi_route_authorization,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.constants import (
    AUTHORIZATION_MODE_ENV,  # noqa: F401
    DEFAULT_REMOTE_GROUPS_HEADER,  # noqa: F401
    DEFAULT_REMOTE_GROUPS_SEPARATOR,
    DEFAULT_REMOTE_USER_HEADER,  # noqa: F401
    REMOTE_GROUPS_HEADER_ENV,  # noqa: F401
    REMOTE_USER_HEADER_ENV,  # noqa: F401
    RESOURCE_ASSISTANTS,  # noqa: F401
    RESOURCE_DATASETS,  # noqa: F401
    RESOURCE_EXPERIMENTS,  # noqa: F401
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
    RESOURCE_GATEWAY_SECRETS,
    RESOURCE_REGISTERED_MODELS,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.constants import (
    UNPROTECTED_PATH_PREFIXES as _UNPROTECTED_PATH_PREFIXES,
)
from mlflow_kubernetes_plugins.auth.constants import (
    UNPROTECTED_PATHS as _UNPROTECTED_PATHS,
)
from mlflow_kubernetes_plugins.auth.constants import (
    WORKSPACE_MUTATION_DENIED_MESSAGE as _WORKSPACE_MUTATION_DENIED_MESSAGE,
)
from mlflow_kubernetes_plugins.auth.constants import (
    WORKSPACE_REQUIRED_ERROR_MESSAGE as _WORKSPACE_REQUIRED_ERROR_MESSAGE,
)
from mlflow_kubernetes_plugins.auth.request_context import (
    AuthorizationRequest,
    build_fastapi_authorization_request,  # noqa: F401
    build_flask_authorization_request,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.rules import (
    GRAPHQL_OPERATION_RULES,  # noqa: F401
    PATH_AUTHORIZATION_RULES,  # noqa: F401
    REQUEST_AUTHORIZATION_RULES,  # noqa: F401
    AuthorizationRule,
    _normalize_rules,  # noqa: F401
)

if not hasattr(werkzeug, "__version__"):  # pragma: no cover - compatibility shim
    werkzeug.__version__ = "werkzeug"

# Re-export GraphQL constants from auth_graphql module
from mlflow_kubernetes_plugins.auth.graphql import (
    GRAPHQL_FIELD_RESOURCE_MAP,
    GRAPHQL_FIELD_VERB_MAP,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    _build_graphql_operation_rules,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.graphql import (
    validate_graphql_field_authorization as _validate_graphql_field_authorization,  # noqa: F401
)
from mlflow_kubernetes_plugins.auth.middleware import (
    KubernetesAuthMiddleware,  # noqa: F401
    _override_run_user,  # noqa: F401
    create_app,
)

_logger = logging.getLogger(__name__)
_AUTHORIZATION_HANDLED: ContextVar[_AuthorizationResult | None] = ContextVar(
    "_AUTHORIZATION_HANDLED", default=None
)


@dataclass(frozen=True)
class _RequestIdentity:
    token: str | None = None
    user: str | None = None
    groups: tuple[str, ...] = ()

    def subject_hash(
        self,
        mode: AuthorizationMode,
        *,
        missing_user_label: str = "Remote user header",
    ) -> str:
        if mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
            if not self.token:
                raise MlflowException(
                    "Bearer token is required for SelfSubjectAccessReview mode.",
                    error_code=databricks_pb2.UNAUTHENTICATED,
                )
            return sha256(self.token.encode("utf-8")).hexdigest()

        user = (self.user or "").strip()
        if not user:
            raise MlflowException(
                f"{missing_user_label} is required for SubjectAccessReview mode.",
                error_code=databricks_pb2.UNAUTHENTICATED,
            )
        # Use the null byte as a delimiter so user/group names cannot collide accidentally.
        normalized_groups = "\x00".join(sorted(self.groups))
        serialized = "\x00".join([user, normalized_groups])
        return sha256(serialized.encode("utf-8")).hexdigest()


def _unwrap_handler(handler):
    while hasattr(handler, "__wrapped__"):
        handler = handler.__wrapped__
    return handler


def _get_static_prefix() -> str | None:
    if prefix := os.environ.get(STATIC_PREFIX_ENV_VAR, None):
        return prefix

    return None


_STATIC_PREFIX_APPLICABLE_PREFIXES: tuple[str, ...] = (
    "/",
    "/health",
    "/metrics",
    "/version",
    "/server-info",
    "/ajax-api",
    "/get-artifact",
    "/model-versions/get-artifact",
    "/static-files",
    "/graphql",
)


def _strip_prefix(path: str, prefix: str | None) -> str:
    """Remove a leading deployment prefix when present."""
    if not path or not prefix:
        return path

    normalized = prefix.strip().rstrip("/")
    if not normalized:
        return path

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized == "/":
        return path

    if path == normalized:
        return "/"

    if path.startswith(f"{normalized}/"):
        stripped = path[len(normalized) :]
        return stripped if stripped.startswith("/") else f"/{stripped}"

    return path


def _strip_static_prefix(path: str) -> str:
    prefix = _get_static_prefix()
    if not prefix:
        return path

    stripped = _strip_prefix(path, prefix)
    if stripped == path:
        # Prefix not present or empty/invalid; nothing to do.
        return path

    # Only strip the static prefix for known static-prefixed routes
    if any(
        stripped == candidate or stripped.startswith(f"{candidate}/")
        for candidate in _STATIC_PREFIX_APPLICABLE_PREFIXES
    ):
        return stripped

    return path


def _canonicalize_path(
    raw_path: str,
    path_info: str | None = None,
    scope_path: str | None = None,
    root_path: str | None = None,
    script_name: str | None = None,
) -> str:
    """
    Normalize a request path into the canonical form used for authorization lookups.

    Behavior:
    - Choose the most canonical source first: `path_info` (WSGI), then `scope_path`
      (ASGI), otherwise `raw_path` from the URL.
    - Ensure the path is absolute (leading slash).
    - Strip deployment prefixes (ASGI `root_path`, WSGI `SCRIPT_NAME`) idempotently; if
      they were already removed upstream, this is a no-op.
    - Strip the MLflow static prefix (when configured) only for known static-prefixed
      routes (e.g., `/ajax-api`, `/static-files`, `/graphql`, root). This produces
      the canonical path we use for allowlist matching (e.g., `/ajax-api/...` even
      when the incoming URL was `/mlflow/ajax-api/...`).

    Returns:
        Canonicalized, absolute path suitable for authorization rule matching.
    """
    path = path_info or scope_path or raw_path or ""
    if not path:
        return path

    if not path.startswith("/"):
        path = f"/{path}"

    # Idempotent stripping of deployment prefixes; safe even if already removed upstream.
    path = _strip_prefix(path, root_path)
    path = _strip_prefix(path, script_name)
    return _strip_static_prefix(path)


@lru_cache(maxsize=None)
def _re_compile_path(path: str) -> re.Pattern[str]:
    def _replace(match: re.Match[str]) -> str:
        if match.group(1).startswith("path:"):
            return "(.+)"
        return "([^/]+)"

    return re.compile(re.sub(r"<([^>]+)>", _replace, path))


def _is_unprotected_path(path: str) -> bool:
    return (
        any(path.startswith(prefix) for prefix in _UNPROTECTED_PATH_PREFIXES)
        or path in _UNPROTECTED_PATHS
    )


_TEMPLATE_TOKEN_PATTERN = re.compile(r"<[^>]+>|{[^}]+}")


def _fastapi_path_to_template(path: str) -> str:
    """Convert FastAPI-style `{param}` segments into Flask-style `<param>` tokens."""
    return re.sub(r"{([^}]+)}", r"<\1>", path)


def _templated_path_to_probe(path: str, placeholder: str = "probe") -> str:
    """Replace templated segments with a concrete placeholder for matching."""
    return _TEMPLATE_TOKEN_PATTERN.sub(placeholder, path)


def _parse_jwt_subject(token: str, claim: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_segment = parts[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        payload = json.loads(decoded)
        value = payload.get(claim)
        return value if isinstance(value, str) and value else None
    except Exception as exc:
        _logger.error(
            "Failed to extract claim '%s' from JWT payload: %s",
            claim,
            exc,
            exc_info=True,
        )
        return None


@dataclass
class _AuthorizationResult:
    identity: _RequestIdentity
    rules: list[AuthorizationRule]
    username: str | None

    @property
    def token(self) -> str | None:
        return self.identity.token


def _resolve_bearer_token(
    authorization_header: str | None, forwarded_access_token: str | None
) -> str:
    if authorization_header:
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
        # fall through to forwarded token if available

    if forwarded_access_token and (token := forwarded_access_token.strip()):
        if token.lower().startswith("bearer "):
            if token := token[7:].strip():
                return token
        elif token:
            return token

    if authorization_header:
        raise MlflowException(
            "Authorization header must be in the format 'Bearer <token>'.",
            error_code=databricks_pb2.UNAUTHENTICATED,
        )

    raise MlflowException(
        "Missing Authorization header or X-Forwarded-Access-Token header.",
        error_code=databricks_pb2.UNAUTHENTICATED,
    )


def _parse_remote_groups(
    header_value: str | None, separator: str = DEFAULT_REMOTE_GROUPS_SEPARATOR
) -> tuple[str, ...]:
    if not header_value:
        return ()

    tokens = [header_value] if not separator else header_value.split(separator)
    return tuple(token.strip() for token in tokens if token and token.strip())


def _extract_workspace_scope_from_request(rule: AuthorizationRule) -> str | None:
    """
    Attempt to recover the workspace name from the active Flask request.

    Workspace CRUD endpoints encode the target workspace name in either the route parameters
    (e.g., ``/workspaces/<workspace_name>``) or, for creation, within the JSON payload. These
    endpoints do not require the workspace context header, so we fall back to parsing the request
    when the context is absent.
    """

    if not has_request_context():
        return None

    view_args = getattr(request, "view_args", None)
    if isinstance(view_args, dict):
        if isinstance(candidate := view_args.get("workspace_name"), str) and (
            candidate := candidate.strip()
        ):
            return candidate

    if not rule.workspace_access_check:
        return None

    if (request.method or "").upper() == "POST":
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            if isinstance(candidate := payload.get("name"), str) and (
                candidate := candidate.strip()
            ):
                return candidate

    return None


def _enforce_gateway_dependency_permissions(
    authorizer: "KubernetesAuthorizer",
    identity: _RequestIdentity,
    workspace_name: str | None,
    rule: AuthorizationRule,
) -> None:
    """
    Enforce cross-resource dependency permissions for gateway operations.

    This mirrors the basic auth plugin's USE permission level via the 'use' subresource:
    - CreateGatewayEndpoint/UpdateGatewayEndpoint: requires USE on model definitions
    - CreateGatewayModelDefinition/UpdateGatewayModelDefinition: requires USE on secrets

    The 'use' subresource allows fine-grained RBAC control: users can have 'get' permission
    to read a resource without having 'create' on '<resource>/use' to reference it.
    """
    if not workspace_name:
        return

    # Gateway endpoints require USE permission on model definitions
    # Checked via 'create' verb on 'gatewaymodeldefinitions/use' subresource
    if rule.resource == RESOURCE_GATEWAY_ENDPOINTS and rule.verb in ("create", "update"):
        if not authorizer.is_allowed(
            identity, RESOURCE_GATEWAY_MODEL_DEFINITIONS, "create", workspace_name, "use"
        ):
            raise MlflowException(
                "Permission denied: creating or updating a gateway endpoint requires "
                "'use' permission on gateway model definitions.",
                error_code=databricks_pb2.PERMISSION_DENIED,
            )

    # Gateway model definitions require USE permission on secrets
    # Checked via 'create' verb on 'gatewaysecrets/use' subresource
    if rule.resource == RESOURCE_GATEWAY_MODEL_DEFINITIONS and rule.verb in ("create", "update"):
        if not authorizer.is_allowed(
            identity, RESOURCE_GATEWAY_SECRETS, "create", workspace_name, "use"
        ):
            raise MlflowException(
                "Permission denied: creating or updating a gateway model definition requires "
                "'use' permission on gateway secrets.",
                error_code=databricks_pb2.PERMISSION_DENIED,
            )


def _authorize_request_context(
    request_context: AuthorizationRequest,
    *,
    authorizer: KubernetesAuthorizer,
    config_values: KubernetesAuthConfig,
) -> _AuthorizationResult:
    """
    Resolve the caller identity and ensure the MLflow request is permitted.

    Depending on the configured authorization mode, the caller is represented either by a bearer
    token (validated via `SelfSubjectAccessReview`) or by proxy-provided username/group headers that
    are evaluated through `SubjectAccessReview`. The resolved AuthorizationRule determines which
    Kubernetes resource/verb combination must be authorized within the workspace context. Any
    failures surface as `MlflowException` instances so HTTP handlers can relay a structured error.

    Returns:
        _AuthorizationResult: Includes the normalized identity, matched authorization rules (may be
            multiple for GraphQL), and the username derived from the token or proxy headers (used
            to override run ownership).

    Raises:
        MlflowException: If authentication information is missing/invalid, the workspace context is
            required but absent, or Kubernetes denies the requested access.
    """
    if config_values.authorization_mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
        token = _resolve_bearer_token(
            request_context.authorization_header,
            request_context.forwarded_access_token,
        )
        identity = _RequestIdentity(token=token)
        username = _parse_jwt_subject(token, config_values.username_claim)
    else:
        remote_user = (request_context.remote_user_header_value or "").strip()
        if not remote_user:
            raise MlflowException(
                f"Missing required '{config_values.user_header}' header for "
                "SubjectAccessReview mode.",
                error_code=databricks_pb2.UNAUTHENTICATED,
            )
        groups = _parse_remote_groups(
            request_context.remote_groups_header_value,
            config_values.groups_separator,
        )
        identity = _RequestIdentity(user=remote_user, groups=groups)
        username = remote_user

    workspace_name = None
    if isinstance(request_context.workspace, str):
        workspace_name = request_context.workspace.strip() or None

    rules = _find_authorization_rules(
        request_context.path,
        request_context.method,
        graphql_payload=request_context.graphql_payload,
    )
    if rules is None or len(rules) == 0:
        _logger.warning(
            "No Kubernetes authorization rule matched request %s %s; returning 404.",
            request_context.method,
            request_context.path,
        )
        raise MlflowException(
            "Endpoint not found.",
            error_code=databricks_pb2.ENDPOINT_NOT_FOUND,
        )

    # Extract workspace from request if not provided via header
    if not workspace_name:
        workspace_name = _extract_workspace_scope_from_request(rules[0])

    # Check authorization for each rule - user must have permission for each resource type
    for rule in rules:
        if rule.deny:
            raise MlflowException(
                _WORKSPACE_MUTATION_DENIED_MESSAGE,
                error_code=databricks_pb2.PERMISSION_DENIED,
            )

        if rule.requires_workspace and not workspace_name:
            raise MlflowException(
                _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        if rule.verb is not None:
            # Standard RBAC check
            if not rule.resource:
                raise MlflowException(
                    f"Authorization rule for '{request_context.method} {request_context.path}' "
                    "is missing an RBAC resource mapping.",
                    error_code=databricks_pb2.INTERNAL_ERROR,
                )
            allowed = authorizer.is_allowed(
                identity,
                rule.resource,
                rule.verb,
                workspace_name,
                rule.subresource,
            )
            if not allowed:
                raise MlflowException(
                    "Permission denied for requested operation.",
                    error_code=databricks_pb2.PERMISSION_DENIED,
                )
            _enforce_gateway_dependency_permissions(authorizer, identity, workspace_name, rule)
        elif rule.workspace_access_check and not rule.apply_workspace_filter:
            # Workspace access check without RBAC verb
            if not workspace_name:
                raise MlflowException(
                    _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                )
            if not authorizer.can_access_workspace(identity, workspace_name, verb="get"):
                raise MlflowException(
                    "Permission denied for requested operation.",
                    error_code=databricks_pb2.PERMISSION_DENIED,
                )
        elif rule.apply_workspace_filter:
            pass  # Authorization is handled via response filtering
        else:
            raise MlflowException(
                f"Authorization rule for '{request_context.method} {request_context.path}' is "
                "missing a verb or other required configuration.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )

    return _AuthorizationResult(identity=identity, rules=rules, username=username)


def _authorize_request(
    *,
    authorization_header: str | None,
    forwarded_access_token: str | None,
    remote_user_header_value: str | None,
    remote_groups_header_value: str | None,
    path: str,
    method: str,
    authorizer: KubernetesAuthorizer,
    config_values: KubernetesAuthConfig,
    workspace: str | None,
    graphql_payload: dict[str, object] | None = None,
) -> _AuthorizationResult:
    return _authorize_request_context(
        AuthorizationRequest(
            authorization_header=authorization_header,
            forwarded_access_token=forwarded_access_token,
            remote_user_header_value=remote_user_header_value,
            remote_groups_header_value=remote_groups_header_value,
            path=path,
            method=method,
            workspace=workspace,
            graphql_payload=graphql_payload,
        ),
        authorizer=authorizer,
        config_values=config_values,
    )


__all__ = [
    "create_app",
    "GRAPHQL_FIELD_RESOURCE_MAP",
    "GRAPHQL_FIELD_VERB_MAP",
    "K8S_GRAPHQL_OPERATION_RESOURCE_MAP",
    "K8S_GRAPHQL_OPERATION_VERB_MAP",
]
