"""Kubernetes-backed authorization plugin for the MLflow tracking server."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256

import werkzeug
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

from mlflow_kubernetes_plugins.auth.authorizer import (
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
)
from mlflow_kubernetes_plugins.auth.collection_filters import (
    apply_request_collection_filter,
    is_graphql_collection_policy,
    is_response_filter_policy,
)
from mlflow_kubernetes_plugins.auth.constants import (
    DEFAULT_REMOTE_GROUPS_SEPARATOR,
    RESOURCE_GATEWAY_BUDGETS,
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
    RESOURCE_GATEWAY_SECRETS,
)
from mlflow_kubernetes_plugins.auth.constants import (
    UNPROTECTED_PATH_PREFIXES as _UNPROTECTED_PATH_PREFIXES,
)
from mlflow_kubernetes_plugins.auth.constants import (
    UNPROTECTED_PATHS as _UNPROTECTED_PATHS,
)
from mlflow_kubernetes_plugins.auth.constants import (
    WORKSPACE_REQUIRED_ERROR_MESSAGE as _WORKSPACE_REQUIRED_ERROR_MESSAGE,
)
from mlflow_kubernetes_plugins.auth.request_context import (
    AuthorizationRequest,
    _build_graphql_payload,
)
from mlflow_kubernetes_plugins.auth.resource_names import (
    ResourceNameResolutionError,
    ResourceReferenceNotPresentError,
    resolve_gateway_model_definition_names_for_use,
    resolve_gateway_secret_names_for_use,
    resolve_resource_names,
)
from mlflow_kubernetes_plugins.auth.rules import (
    AuthorizationRule,
)

if not hasattr(werkzeug, "__version__"):  # pragma: no cover - compatibility shim
    werkzeug.__version__ = "werkzeug"

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
    request_context: AuthorizationRequest
    username: str | None
    response_filter_required: bool = False

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


def _extract_workspace_scope_from_request(
    request_context: AuthorizationRequest, rule: AuthorizationRule
) -> str | None:
    """
    Attempt to recover the workspace name from the request context.

    Workspace CRUD endpoints encode the target workspace name in either the route parameters
    (e.g., ``/workspaces/<workspace_name>``) or, for creation, within the JSON payload. These
    endpoints do not require the workspace context header, so we fall back to parsing the request
    when the context is absent.
    """
    path_params = request_context.path_params
    if isinstance(path_params.get("workspace_name"), str) and (
        candidate := path_params["workspace_name"].strip()
    ):
        return candidate

    if not rule.workspace_access_check:
        return None

    if (request_context.method or "").upper() == "POST":
        payload = request_context.json_body
        if isinstance(payload, dict):
            if isinstance(candidate := payload.get("name"), str) and (
                candidate := candidate.strip()
            ):
                return candidate

    return None


def _enforce_gateway_dependency_permissions(
    authorizer: "KubernetesAuthorizer",
    identity: _RequestIdentity,
    request_context: AuthorizationRequest,
    workspace_name: str | None,
    rule: AuthorizationRule,
) -> None:
    """
    Enforce cross-resource dependency permissions for gateway operations.

    This mirrors the basic auth plugin's USE permission level via the 'use' subresource. Broad
    namespace-level USE access short-circuits the checks. Otherwise, when the request names a
    specific dependency, the plugin retries with the dependency's MLflow resource name so
    fine-grained resourceNames can authorize the request.

    The 'use' subresource allows fine-grained RBAC control: users can have 'get' permission
    to read a resource without having 'create' on '<resource>/use' to reference it.
    """
    if not workspace_name:
        return

    # Gateway endpoints require USE permission on model definitions.
    if (
        rule.resource == RESOURCE_GATEWAY_ENDPOINTS
        and rule.subresource != "use"
        and rule.verb in ("create", "update")
    ):
        if authorizer.is_allowed(
            identity, RESOURCE_GATEWAY_MODEL_DEFINITIONS, "create", workspace_name, "use"
        ):
            return
        try:
            dependency_names = resolve_gateway_model_definition_names_for_use(request_context)
        except ResourceNameResolutionError:
            dependency_names = ()
        if dependency_names and all(
            authorizer.is_allowed(
                identity,
                RESOURCE_GATEWAY_MODEL_DEFINITIONS,
                "create",
                workspace_name,
                "use",
                resource_name=dependency_name,
            )
            for dependency_name in dependency_names
        ):
            return
        raise MlflowException(
            "Permission denied: creating or updating a gateway endpoint requires "
            "'use' permission on gateway model definitions.",
            error_code=databricks_pb2.PERMISSION_DENIED,
        )

    # Gateway model definitions require USE permission on secrets.
    if (
        rule.resource == RESOURCE_GATEWAY_MODEL_DEFINITIONS
        and rule.subresource != "use"
        and rule.verb in ("create", "update")
    ):
        if authorizer.is_allowed(
            identity, RESOURCE_GATEWAY_SECRETS, "create", workspace_name, "use"
        ):
            return
        try:
            dependency_names = resolve_gateway_secret_names_for_use(request_context)
        except ResourceNameResolutionError:
            dependency_names = ()
        if dependency_names and all(
            authorizer.is_allowed(
                identity,
                RESOURCE_GATEWAY_SECRETS,
                "create",
                workspace_name,
                "use",
                resource_name=dependency_name,
            )
            for dependency_name in dependency_names
        ):
            return
        raise MlflowException(
            "Permission denied: creating or updating a gateway model definition requires "
            "'use' permission on gateway secrets.",
            error_code=databricks_pb2.PERMISSION_DENIED,
        )


async def _enforce_gateway_budget_scope(
    request_context: AuthorizationRequest, rule: AuthorizationRule
) -> AuthorizationRequest:
    if rule.resource != RESOURCE_GATEWAY_BUDGETS or rule.verb not in {"create", "update"}:
        return request_context

    updated_request_context = await _ensure_request_context_json_body(request_context)
    payload = updated_request_context.json_body
    if not isinstance(payload, dict):
        raise MlflowException(
            "Gateway budget policy requests must use a JSON object body.",
            error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
        )

    for raw_target_scope in (payload.get("target_scope"), payload.get("targetScope")):
        normalized_target_scope = raw_target_scope
        if isinstance(raw_target_scope, str):
            normalized_target_scope = raw_target_scope.strip()
            if normalized_target_scope.isdigit():
                normalized_target_scope = int(normalized_target_scope)
            else:
                normalized_target_scope = normalized_target_scope.upper()
        if normalized_target_scope in {1, "GLOBAL"}:
            raise MlflowException(
                "Gateway budget policies must remain workspace-scoped. "
                "GLOBAL target_scope is not supported by the Kubernetes auth plugin.",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
    return updated_request_context


async def _ensure_request_context_json_body(
    request_context: AuthorizationRequest,
) -> AuthorizationRequest:
    """Return an ``AuthorizationRequest`` that has its JSON body loaded.

    If the body is already present (or no loader callback is available), the
    original *request_context* is returned unchanged.  Otherwise the callback
    is invoked to fetch the raw body and a new ``AuthorizationRequest`` is
    constructed with the loaded ``json_body`` and derived ``graphql_payload``.
    """
    if request_context.json_body is not None or request_context.ensure_json_body is None:
        return request_context
    loaded_body = await request_context.ensure_json_body()
    return replace(
        request_context,
        json_body=loaded_body,
        graphql_payload=_build_graphql_payload(
            request_context.path,
            json_body=loaded_body,
            query_params=request_context.query_params,
        ),
        ensure_json_body=None,
    )


async def _authorize_request_async(
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

    from mlflow_kubernetes_plugins.auth.compiler import _find_authorization_rules

    updated_request_context = request_context
    if request_context.path.endswith("/graphql") and request_context.graphql_payload is None:
        updated_request_context = await _ensure_request_context_json_body(updated_request_context)

    rules = _find_authorization_rules(
        updated_request_context.path,
        updated_request_context.method,
        graphql_payload=updated_request_context.graphql_payload,
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

    if not workspace_name:
        workspace_name = _extract_workspace_scope_from_request(updated_request_context, rules[0])
        if workspace_name:
            updated_request_context = replace(updated_request_context, workspace=workspace_name)

    response_filter_required = False
    for rule in rules:
        if rule.deny:
            deny_message = rule.deny_message
            if deny_message is None:
                raise MlflowException(
                    f"Authorization rule for '{request_context.method} {request_context.path}' "
                    "is missing a deny_message.",
                    error_code=databricks_pb2.INTERNAL_ERROR,
                )
            raise MlflowException(
                deny_message,
                error_code=databricks_pb2.PERMISSION_DENIED,
            )

        if rule.requires_workspace and not workspace_name:
            raise MlflowException(
                _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        if rule.verb is not None:
            if rule.resource == RESOURCE_GATEWAY_BUDGETS and rule.verb in {"create", "update"}:
                updated_request_context = await _enforce_gateway_budget_scope(
                    updated_request_context, rule
                )
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
            # Fallback to resourceName when the user doesn't have the broad permissions.
            # This approach allows us to reuse cache SelfSubjectAccessReview of the common case
            # where a user has access to all resources to the workspace.
            if not allowed and rule.resource_name_parsers:
                updated_request_context = await _ensure_request_context_json_body(
                    updated_request_context
                )
                try:
                    resource_names = resolve_resource_names(
                        updated_request_context, rule.resource_name_parsers
                    )
                except ResourceReferenceNotPresentError:
                    if rule.allow_if_resource_reference_missing:
                        continue
                    resource_names = ()
                except ResourceNameResolutionError:
                    resource_names = ()
                allowed = bool(resource_names) and all(
                    authorizer.is_allowed(
                        identity,
                        rule.resource,
                        rule.verb,
                        workspace_name,
                        rule.subresource,
                        resource_name=resource_name,
                    )
                    for resource_name in resource_names
                )
            if not allowed and rule.collection_policy:
                updated_request_context, request_filter_applied = apply_request_collection_filter(
                    updated_request_context,
                    rule.collection_policy,
                    authorizer=authorizer,
                    identity=identity,
                    workspace_name=workspace_name,
                )
                if not request_filter_applied:
                    updated_request_context = await _ensure_request_context_json_body(
                        updated_request_context
                    )
                    updated_request_context, request_filter_applied = (
                        apply_request_collection_filter(
                            updated_request_context,
                            rule.collection_policy,
                            authorizer=authorizer,
                            identity=identity,
                            workspace_name=workspace_name,
                        )
                    )
                if request_filter_applied:
                    allowed = True
                elif is_response_filter_policy(
                    rule.collection_policy
                ) or is_graphql_collection_policy(rule.collection_policy):
                    response_filter_required = True
                    allowed = True
            if not allowed:
                raise MlflowException(
                    "Permission denied for requested operation.",
                    error_code=databricks_pb2.PERMISSION_DENIED,
                )
            _enforce_gateway_dependency_permissions(
                authorizer, identity, updated_request_context, workspace_name, rule
            )
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

    if any(rule.override_run_user for rule in rules):
        updated_request_context = await _ensure_request_context_json_body(updated_request_context)

    return _AuthorizationResult(
        identity=identity,
        rules=rules,
        request_context=updated_request_context,
        username=username,
        response_filter_required=response_filter_required,
    )


__all__ = [
    "AuthorizationRequest",
    "AuthorizationRule",
    "KubernetesAuthConfig",
    "KubernetesAuthorizer",
    "_AUTHORIZATION_HANDLED",
    "_AuthorizationResult",
    "_RequestIdentity",
    "_authorize_request_async",
    "_canonicalize_path",
    "_enforce_gateway_budget_scope",
    "_ensure_request_context_json_body",
    "_enforce_gateway_dependency_permissions",
    "_extract_workspace_scope_from_request",
    "_fastapi_path_to_template",
    "_get_static_prefix",
    "_is_unprotected_path",
    "_logger",
    "_parse_jwt_subject",
    "_parse_remote_groups",
    "_re_compile_path",
    "_resolve_bearer_token",
    "_strip_prefix",
    "_strip_static_prefix",
    "_templated_path_to_probe",
    "_unwrap_handler",
]
