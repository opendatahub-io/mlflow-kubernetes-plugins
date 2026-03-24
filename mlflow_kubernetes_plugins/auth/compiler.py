"""Compiled authorization rule lookup tables and startup validation."""

from __future__ import annotations

import re

from fastapi import FastAPI
from fastapi.routing import APIRoute
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server import app as mlflow_app
from mlflow.server import handlers as mlflow_handlers

from mlflow_kubernetes_plugins.auth.graphql import (
    determine_graphql_rules as _determine_graphql_rules,
)
from mlflow_kubernetes_plugins.auth.graphql import (
    extract_graphql_query_info as _extract_graphql_query_info,
)
from mlflow_kubernetes_plugins.auth.rules import (
    PATH_AUTHORIZATION_RULES,
    REQUEST_AUTHORIZATION_RULES,
    AuthorizationRule,
    _normalize_rules,
)

_AUTH_RULES: dict[tuple[str, str], list[AuthorizationRule]] = {}
_AUTH_REGEX_RULES: list[tuple[re.Pattern[str], str, list[AuthorizationRule]]] = []
_HANDLER_RULES: dict[object, list[AuthorizationRule]] = {}
_RULES_COMPILED = False


def _auth_module():
    import mlflow_kubernetes_plugins.auth as auth_mod

    return auth_mod


def _reset_compiled_rules() -> None:
    global _RULES_COMPILED

    _RULES_COMPILED = False
    _AUTH_RULES.clear()
    _AUTH_REGEX_RULES.clear()
    _HANDLER_RULES.clear()


def _compile_authorization_rules() -> None:
    global _RULES_COMPILED
    if _RULES_COMPILED:
        return

    auth_mod = _auth_module()

    # Rebuild every cache/artifact so reconfiguration (e.g., during tests) is deterministic.
    _HANDLER_RULES.clear()

    exact_rules: dict[tuple[str, str], list[AuthorizationRule]] = {}
    regex_rules: list[tuple[re.Pattern[str], str, list[AuthorizationRule]]] = []
    uncovered: list[tuple[str, str]] = []

    def _get_request_authorization_handler(request_class):
        # Record the AuthorizationRule associated with the concrete Flask handler so we can
        # reference it later when iterating through Flask endpoints.
        handler = mlflow_handlers.get_handler(request_class)
        value = REQUEST_AUTHORIZATION_RULES.get(request_class)
        if handler is not None and value is not None:
            _HANDLER_RULES[auth_mod._unwrap_handler(handler)] = _normalize_rules(value)
        return handler

    # Inspect the protobuf-driven Flask routes and copy over authorization metadata.
    for path, handler, methods in auth_mod.get_endpoints(_get_request_authorization_handler):
        if not path:
            continue

        canonical_path = auth_mod._canonicalize_path(raw_path=path)
        if auth_mod._is_unprotected_path(canonical_path):
            continue

        base_handler = auth_mod._unwrap_handler(handler)
        rules = _HANDLER_RULES.get(base_handler)
        if rules is None:
            # If a protobuf route lacks a handler-derived rule, fall back to the explicit
            # PATH_AUTHORIZATION_RULES definition; otherwise flag it as uncovered.
            if all(
                PATH_AUTHORIZATION_RULES.get((canonical_path, method)) is not None
                for method in methods
            ):
                continue
            uncovered.extend((canonical_path, method) for method in methods)
            continue

        for method in methods:
            # Regex patterns are required for templated paths; literal paths can be matched exactly.
            if "<" in canonical_path:
                regex_rules.append((auth_mod._re_compile_path(canonical_path), method, rules))
            else:
                exact_rules[(canonical_path, method)] = rules

    # Include custom Flask routes (e.g., get-artifact) that aren't part of the protobuf services.
    for rule in mlflow_app.url_map.iter_rules():
        view_func = mlflow_app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue

        canonical_path = auth_mod._canonicalize_path(raw_path=rule.rule)
        if auth_mod._is_unprotected_path(canonical_path):
            continue

        base_handler = auth_mod._unwrap_handler(view_func)
        if base_handler in _HANDLER_RULES:
            continue

        methods = {m for m in (rule.methods or set()) if m not in {"HEAD", "OPTIONS"}}
        # These custom routes rely exclusively on PATH_AUTHORIZATION_RULES; track any gaps.
        missing_methods = [
            (canonical_path, method)
            for method in methods
            if PATH_AUTHORIZATION_RULES.get((canonical_path, method)) is None
        ]
        if missing_methods:
            uncovered.extend(missing_methods)

    # Explicit allowlist entries (with and without templated segments) always win.
    for (path, method), path_value in PATH_AUTHORIZATION_RULES.items():
        normalized = _normalize_rules(path_value)
        if "<" in path:
            regex_rules.append((auth_mod._re_compile_path(path), method, normalized))
        else:
            exact_rules[(path, method)] = normalized

    if uncovered:
        formatted = ", ".join(f"{method} {path}" for path, method in uncovered)
        raise MlflowException(
            "Kubernetes auth plugin cannot determine authorization mapping for endpoints: "
            f"{formatted}. Update the plugin allow list or verb mapping.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )

    _AUTH_RULES.update(exact_rules)
    _AUTH_REGEX_RULES.extend(regex_rules)
    _RULES_COMPILED = True


def _validate_fastapi_route_authorization(fastapi_app: FastAPI) -> None:
    """Ensure all protected FastAPI routes are covered by authorization rules."""
    auth_mod = _auth_module()
    missing: list[tuple[str, str]] = []

    for route in getattr(fastapi_app, "routes", []):
        if not isinstance(route, APIRoute):
            continue
        methods = getattr(route, "methods", set()) or set()
        canonical_path = auth_mod._canonicalize_path(raw_path=route.path or "")
        if not canonical_path or auth_mod._is_unprotected_path(canonical_path):
            continue
        template_path = auth_mod._fastapi_path_to_template(canonical_path)
        # Use a concrete probe path so _find_authorization_rules follows the same regex path
        # matching logic that real requests do.
        probe_path = auth_mod._templated_path_to_probe(template_path)

        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            if _find_authorization_rules(probe_path, method) is None:
                missing.append((method, canonical_path))

    if missing:
        formatted = ", ".join(f"{method} {path}" for method, path in missing)
        raise MlflowException(
            "Kubernetes auth plugin is missing authorization rules for FastAPI endpoints: "
            f"{formatted}. Update PATH_AUTHORIZATION_RULES before enabling the plugin.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )


def _find_authorization_rules(
    request_path: str, method: str, graphql_payload: dict[str, object] | None = None
) -> list[AuthorizationRule] | None:
    """Find authorization rules for a request."""
    auth_mod = _auth_module()
    canonical_path = auth_mod._canonicalize_path(raw_path=request_path or "")

    rules = _AUTH_RULES.get((canonical_path, method))
    if rules is not None:
        # Special handling for GraphQL operations
        # SECURITY: Always parse the query to determine authorization rules.
        # We cannot trust operationName alone because a malicious client could
        # send operationName="GetRun" but include model registry fields in the
        # query, bypassing authorization checks for those resources.
        if canonical_path.endswith("/graphql"):
            payload = graphql_payload or {}

            query_string = payload.get("query", "")
            if not query_string:
                auth_mod._logger.error("Could not determine GraphQL authorization: no query provided.")
                return None

            query_info = _extract_graphql_query_info(query_string)
            if not query_info.root_fields and not query_info.has_nested_model_registry_access:
                auth_mod._logger.error(
                    "Could not determine GraphQL authorization: query could not be "
                    "parsed or contained no recognized fields."
                )
                return None

            return _determine_graphql_rules(query_info, AuthorizationRule)
        return rules

    for pattern, pattern_method, candidate in _AUTH_REGEX_RULES:
        if pattern_method == method and pattern.fullmatch(canonical_path):
            return candidate

    return None


__all__ = [
    "_AUTH_REGEX_RULES",
    "_AUTH_RULES",
    "_HANDLER_RULES",
    "_RULES_COMPILED",
    "_compile_authorization_rules",
    "_find_authorization_rules",
    "_reset_compiled_rules",
    "_validate_fastapi_route_authorization",
]
