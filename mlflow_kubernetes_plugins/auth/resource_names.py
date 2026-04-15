"""Resource-name parsing and lookup helpers for authorization fallback."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException

from mlflow_kubernetes_plugins.auth.request_context import AuthorizationRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mlflow_kubernetes_plugins.auth.rules import AuthorizationRule

RESOURCE_NAME_PARSER_EXPERIMENT_NAME = "experiment_name"
RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME = "experiment_id_to_name"
RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES = "experiment_ids_to_names"
RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME = "new_experiment_name"
RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME = "run_id_to_experiment_name"
RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME = "model_id_to_experiment_name"
RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME = "dataset_id_to_name"
RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME = "registered_model_name"
RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME = "new_registered_model_name"
RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME = "webhook_id_to_registered_model_name"
RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME = "job_id_to_experiment_name"
RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME = "issue_id_to_experiment_name"
RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME = "trace_request_id_to_experiment_name"
RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME = "trace_id_to_experiment_name"
RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME = "trace_v3_experiment_id_to_name"
RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME = "artifact_experiment_id_to_name"
RESOURCE_NAME_PARSER_OTEL_EXPERIMENT_ID_HEADER_TO_NAME = "otel_experiment_id_header_to_name"
RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME = "gateway_secret_id_to_name"
RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME = "gateway_endpoint_id_to_name"
RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME = "gateway_proxy_endpoint_name"
RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME = "gateway_model_definition_id_to_name"
RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME = "graphql_experiment_id_to_name"
RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME = "graphql_run_id_to_experiment_name"
RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES = (
    "graphql_run_ids_to_experiment_names"
)

_EXPERIMENT_ID_PATTERN = re.compile(r"^(\d+)/")
_GATEWAY_PROXY_ENDPOINT_PATTERN = re.compile(r"^gateway/([^/]+)/invocations$")
_EXPERIMENT_NAME_CACHE_TTL_SECONDS = 60.0
_LOOKUP_CACHE_MAX_ENTRIES = 4096


class ResourceNameResolutionError(RuntimeError):
    """Raised when a resource name cannot be resolved safely."""


class _NameLookupCache:
    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()

    def _reap_expired(self, now: float) -> None:
        expired_keys = [
            cache_key for cache_key, (_, expires_at) in self._entries.items() if expires_at <= now
        ]
        for cache_key in expired_keys:
            self._entries.pop(cache_key, None)

    def get(self, cache_key: str) -> str | None:
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            name, expires_at = entry
            now = time.time()
            if expires_at > now:
                self._entries.move_to_end(cache_key)
                return name
            self._entries.pop(cache_key, None)
        return None

    def set(self, cache_key: str, name: str) -> None:
        with self._lock:
            now = time.time()
            self._reap_expired(now)
            self._entries.pop(cache_key, None)
            while len(self._entries) >= self._max_entries:
                self._entries.popitem(last=False)
            self._entries[cache_key] = (name, now + self._ttl_seconds)

    def invalidate(self, cache_key: str) -> None:
        with self._lock:
            self._entries.pop(cache_key, None)


_experiment_name_cache = _NameLookupCache(
    _EXPERIMENT_NAME_CACHE_TTL_SECONDS,
    max_entries=_LOOKUP_CACHE_MAX_ENTRIES,
)
_run_experiment_name_cache = _NameLookupCache(
    _EXPERIMENT_NAME_CACHE_TTL_SECONDS,
    max_entries=_LOOKUP_CACHE_MAX_ENTRIES,
)


def _normalize_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _get_nested_mapping(
    mapping: dict[str, object],
    *candidate_keys: str,
    context: str,
) -> dict[str, object]:
    """Return a nested mapping from any supported field spelling.

    MLflow request and response payloads are not fully consistent across REST and GraphQL surfaces,
    so the auth helpers intentionally accept either snake_case or camelCase field names here.
    """
    for key in candidate_keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            return value
    raise ResourceNameResolutionError(f"{context} is required for authorization.")


def _get_single_value(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if isinstance(value, list):
        normalized_values = _normalize_values(value)
        if not normalized_values:
            return None
        first_value = normalized_values[0]
        return first_value if all(item == first_value for item in normalized_values[1:]) else None
    return _normalize_string(value)


def _get_request_param(request_context: AuthorizationRequest, param: str) -> str:
    value = _get_optional_request_param(request_context, param)
    if value is None:
        raise ResourceNameResolutionError(f"Missing required parameter '{param}' for authorization.")
    return value


def _normalize_values(value: object) -> list[str]:
    raw_values = value if isinstance(value, list) else ([] if value is None else [value])
    normalized_values: list[str] = []
    for item in raw_values:
        normalized_item = _normalize_string(item)
        if normalized_item is not None:
            normalized_values.append(normalized_item)
    return normalized_values


def _get_request_param_values(request_context: AuthorizationRequest, param: str) -> list[str]:
    method = (request_context.method or "").upper()
    has_json_body = isinstance(request_context.json_body, dict)
    body = request_context.json_body if has_json_body else {}

    if method == "GET":
        values = _normalize_values(request_context.query_params.get(param))
    elif method in {"POST", "PATCH", "PUT"}:
        values = _normalize_values(body.get(param))
    elif method == "DELETE":
        if has_json_body:
            values = _normalize_values(body.get(param))
        else:
            values = _normalize_values(request_context.query_params.get(param))
    else:
        raise ResourceNameResolutionError(
            f"Unsupported HTTP method '{request_context.method}' for authorization parsing."
        )

    values.extend(_normalize_values(request_context.path_params.get(param)))

    if not values and param == "run_id":
        return _get_request_param_values(request_context, "run_uuid")
    return values


def _get_optional_request_param(request_context: AuthorizationRequest, param: str) -> str | None:
    method = (request_context.method or "").upper()
    has_json_body = isinstance(request_context.json_body, dict)
    body = request_context.json_body if has_json_body else {}

    if method == "GET":
        value = _get_single_value(request_context.query_params, param)
    elif method in {"POST", "PATCH", "PUT"}:
        value = _get_single_value(body, param)
    elif method == "DELETE":
        if has_json_body:
            value = _get_single_value(body, param)
        else:
            value = _get_single_value(request_context.query_params, param)
    else:
        raise ResourceNameResolutionError(
            f"Unsupported HTTP method '{request_context.method}' for authorization parsing."
        )

    if value is None:
        value = _get_single_value(request_context.path_params, param)

    if value is None and param == "run_id":
        return _get_optional_request_param(request_context, "run_uuid")
    return value


def _get_header(request_context: AuthorizationRequest, header_name: str) -> str | None:
    return _normalize_string(request_context.headers.get(header_name.lower()))


def _get_tracking_store():
    from mlflow.server.handlers import _get_tracking_store

    return _get_tracking_store()


def _get_model_registry_store():
    from mlflow.server.handlers import _get_model_registry_store

    return _get_model_registry_store()


def _resolve_experiment_name_from_experiment_id(experiment_id: str) -> str:
    if cached_name := _experiment_name_cache.get(experiment_id):
        return cached_name

    try:
        experiment = _get_tracking_store().get_experiment(experiment_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment for experiment_id '{experiment_id}'."
        ) from exc
    experiment_name = _normalize_string(getattr(experiment, "name", None))
    if experiment_name is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment name for experiment_id '{experiment_id}'."
        )
    _experiment_name_cache.set(experiment_id, experiment_name)
    return experiment_name


def _resolve_experiment_name_from_run_id(run_id: str) -> str:
    if cached_experiment_id := _run_experiment_name_cache.get(run_id):
        return _resolve_experiment_name_from_experiment_id(cached_experiment_id)
    try:
        run = _get_tracking_store().get_run(run_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve run_id '{run_id}'.") from exc
    experiment_id = _normalize_string(getattr(getattr(run, "info", None), "experiment_id", None))
    if experiment_id is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment_id for run_id '{run_id}'."
        )
    _run_experiment_name_cache.set(run_id, experiment_id)
    return _resolve_experiment_name_from_experiment_id(experiment_id)


def _resolve_dataset_name_from_dataset_id(dataset_id: str) -> str:
    try:
        dataset = _get_tracking_store().get_dataset(dataset_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve dataset_id '{dataset_id}'.") from exc
    dataset_name = _normalize_string(getattr(dataset, "name", None))
    if dataset_name is None:
        raise ResourceNameResolutionError(
            f"Could not resolve dataset name for dataset_id '{dataset_id}'."
        )
    return dataset_name


def _resolve_experiment_name_from_model_id(model_id: str) -> str:
    try:
        model = _get_tracking_store().get_logged_model(model_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve model_id '{model_id}'.") from exc
    experiment_id = _normalize_string(getattr(model, "experiment_id", None))
    if experiment_id is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment_id for model_id '{model_id}'."
        )
    return _resolve_experiment_name_from_experiment_id(experiment_id)


def _resolve_experiment_name_from_job_id(job_id: str) -> str:
    from mlflow.server.jobs import get_job

    try:
        job = get_job(job_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve job_id '{job_id}'.") from exc
    try:
        params = json.loads(getattr(job, "params", "") or "{}")
    except json.JSONDecodeError as exc:
        raise ResourceNameResolutionError(
            f"Could not decode job params for job_id '{job_id}'."
        ) from exc

    experiment_id = _normalize_string(params.get("experiment_id"))
    if experiment_id is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment_id for job_id '{job_id}'."
        )
    return _resolve_experiment_name_from_experiment_id(experiment_id)


def _resolve_experiment_name_from_issue_id(issue_id: str) -> str:
    store = _get_tracking_store()
    getter = getattr(store, "get_issue", None)
    if getter is None:
        raise ResourceNameResolutionError("Issue lookup is unavailable in this MLflow version.")
    try:
        issue = getter(issue_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve issue_id '{issue_id}'.") from exc
    experiment_id = _normalize_string(getattr(issue, "experiment_id", None))
    if experiment_id is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment_id for issue_id '{issue_id}'."
        )
    return _resolve_experiment_name_from_experiment_id(experiment_id)


def _resolve_experiment_name_from_trace_id(trace_id: str) -> str:
    try:
        trace = _get_tracking_store().get_trace_info(trace_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve trace_id '{trace_id}'.") from exc
    experiment_id = _normalize_string(getattr(trace, "experiment_id", None))
    if experiment_id is None:
        raise ResourceNameResolutionError(
            f"Could not resolve experiment_id for trace_id '{trace_id}'."
        )
    return _resolve_experiment_name_from_experiment_id(experiment_id)


def _resolve_registered_model_name_from_webhook_id(webhook_id: str) -> str:
    store = _get_model_registry_store()
    getter = getattr(store, "get_webhook", None)
    if getter is None:
        raise ResourceNameResolutionError("Webhook lookup is unavailable in this MLflow version.")
    try:
        webhook = getter(webhook_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve webhook_id '{webhook_id}'.") from exc
    registered_model_name = _normalize_string(getattr(webhook, "name", None))
    if registered_model_name is None:
        raise ResourceNameResolutionError(
            f"Could not resolve registered model name for webhook_id '{webhook_id}'."
        )
    return registered_model_name


def _resolve_gateway_secret_name_from_secret_id(secret_id: str) -> str:
    store = _get_tracking_store()
    getter = getattr(store, "get_secret_info", None) or getattr(store, "get_gateway_secret_info", None)
    if getter is None:
        raise ResourceNameResolutionError("Gateway secret lookup is unavailable in this MLflow version.")
    try:
        secret = getter(secret_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve secret_id '{secret_id}'.") from exc
    secret_name = _normalize_string(getattr(secret, "secret_name", None))
    if secret_name is None:
        raise ResourceNameResolutionError(
            f"Could not resolve gateway secret name for secret_id '{secret_id}'."
        )
    return secret_name


def _resolve_gateway_endpoint_name_from_endpoint_id(endpoint_id: str) -> str:
    try:
        endpoint = _get_tracking_store().get_gateway_endpoint(endpoint_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(f"Could not resolve endpoint_id '{endpoint_id}'.") from exc
    endpoint_name = _normalize_string(getattr(endpoint, "name", None))
    if endpoint_name is None:
        raise ResourceNameResolutionError(
            f"Could not resolve gateway endpoint name for endpoint_id '{endpoint_id}'."
        )
    return endpoint_name


def _resolve_gateway_model_definition_name_from_id(model_definition_id: str) -> str:
    try:
        model_definition = _get_tracking_store().get_gateway_model_definition(model_definition_id)
    except MlflowException as exc:
        raise ResourceNameResolutionError(
            f"Could not resolve model_definition_id '{model_definition_id}'."
        ) from exc
    model_definition_name = _normalize_string(getattr(model_definition, "name", None))
    if model_definition_name is None:
        raise ResourceNameResolutionError(
            "Could not resolve gateway model definition name for "
            f"model_definition_id '{model_definition_id}'."
        )
    return model_definition_name


def _get_experiment_id_from_artifact_request(request_context: AuthorizationRequest) -> str | None:
    artifact_path = _get_single_value(request_context.path_params, "artifact_path") or _get_single_value(
        request_context.query_params, "path"
    )
    if artifact_path is None:
        return None
    match = _EXPERIMENT_ID_PATTERN.match(artifact_path)
    return match.group(1) if match else None


def _parse_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_get_request_param(request_context, "experiment_name"),)


def _parse_experiment_id_to_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    experiment_id = _get_request_param(request_context, "experiment_id")
    return (_resolve_experiment_name_from_experiment_id(experiment_id),)


def _parse_experiment_ids_to_names(request_context: AuthorizationRequest) -> tuple[str, ...]:
    experiment_ids = _get_request_param_values(request_context, "experiment_ids")
    if not experiment_ids:
        raise ResourceNameResolutionError(
            "Missing required parameter 'experiment_ids' for authorization."
        )
    return tuple(
        _resolve_experiment_name_from_experiment_id(experiment_id)
        for experiment_id in dict.fromkeys(experiment_ids)
    )


def _parse_new_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_get_request_param(request_context, "new_name"),)


def _parse_run_id_to_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    run_id = _get_optional_request_param(request_context, "run_id")
    if run_id is None and request_context.path == "/ajax-api/2.0/mlflow/upload-artifact":
        run_id = _get_single_value(request_context.query_params, "run_id") or _get_single_value(
            request_context.query_params, "run_uuid"
        )
    if run_id is None:
        raise ResourceNameResolutionError("Missing required parameter 'run_id' for authorization.")
    return (_resolve_experiment_name_from_run_id(run_id),)


def _parse_dataset_id_to_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_resolve_dataset_name_from_dataset_id(_get_request_param(request_context, "dataset_id")),)


def _parse_model_id_to_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_resolve_experiment_name_from_model_id(_get_request_param(request_context, "model_id")),)


def _parse_registered_model_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_get_request_param(request_context, "name"),)


def _parse_new_registered_model_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_get_request_param(request_context, "new_name"),)


def _parse_webhook_id_to_registered_model_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    return (
        _resolve_registered_model_name_from_webhook_id(_get_request_param(request_context, "webhook_id")),
    )


def _parse_job_id_to_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_resolve_experiment_name_from_job_id(_get_request_param(request_context, "job_id")),)


def _parse_issue_id_to_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_resolve_experiment_name_from_issue_id(_get_request_param(request_context, "issue_id")),)


def _parse_trace_request_id_to_experiment_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    return (_resolve_experiment_name_from_trace_id(_get_request_param(request_context, "request_id")),)


def _parse_trace_id_to_experiment_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (_resolve_experiment_name_from_trace_id(_get_request_param(request_context, "trace_id")),)


def _parse_trace_v3_experiment_id_to_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    if not isinstance(request_context.json_body, dict):
        raise ResourceNameResolutionError("StartTraceV3 request body is required for authorization.")

    trace = _get_nested_mapping(
        request_context.json_body,
        "trace",
        context="StartTraceV3 trace payload",
    )
    trace_info = _get_nested_mapping(
        trace,
        "trace_info",
        "traceInfo",
        context="StartTraceV3 trace_info payload",
    )
    trace_location = _get_nested_mapping(
        trace_info,
        "trace_location",
        "traceLocation",
        context="StartTraceV3 trace location payload",
    )
    mlflow_experiment = _get_nested_mapping(
        trace_location,
        "mlflow_experiment",
        "mlflowExperiment",
        context="StartTraceV3 MLflow experiment location",
    )
    experiment_id = _normalize_string(
        mlflow_experiment.get("experiment_id") or mlflow_experiment.get("experimentId")
    )
    if experiment_id is None:
        raise ResourceNameResolutionError(
            "StartTraceV3 request did not include an MLflow experiment ID."
        )
    return (_resolve_experiment_name_from_experiment_id(experiment_id),)


def _parse_artifact_experiment_id_to_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    experiment_id = _get_experiment_id_from_artifact_request(request_context)
    if experiment_id is None:
        raise ResourceNameResolutionError(
            "Artifact request did not contain an experiment-id-prefixed artifact path."
        )
    return (_resolve_experiment_name_from_experiment_id(experiment_id),)


def _parse_otel_experiment_id_header_to_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    experiment_id = _get_header(request_context, "x-mlflow-experiment-id")
    if experiment_id is None:
        raise ResourceNameResolutionError("Missing X-MLflow-Experiment-Id header.")
    return (_resolve_experiment_name_from_experiment_id(experiment_id),)


def _parse_gateway_secret_id_to_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (
        _resolve_gateway_secret_name_from_secret_id(_get_request_param(request_context, "secret_id")),
    )


def _parse_gateway_endpoint_id_to_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    return (
        _resolve_gateway_endpoint_name_from_endpoint_id(
            _get_request_param(request_context, "endpoint_id")
        ),
    )


def _parse_gateway_proxy_endpoint_name(request_context: AuthorizationRequest) -> tuple[str, ...]:
    if path_endpoint_name := _get_optional_request_param(request_context, "endpoint_name"):
        return (path_endpoint_name,)
    gateway_path = _get_optional_request_param(request_context, "gateway_path")
    model_name = _get_optional_request_param(request_context, "model")
    if gateway_path is not None:
        normalized_gateway_path = gateway_path.strip("/")
        match = _GATEWAY_PROXY_ENDPOINT_PATTERN.fullmatch(normalized_gateway_path)
        if match is not None:
            endpoint_name = match.group(1)
            if model_name is not None and model_name != endpoint_name:
                raise ResourceNameResolutionError(
                    "Conflicting endpoint identifiers in gateway_path and model."
                )
            return (endpoint_name,)
        raise ResourceNameResolutionError(
            f"Could not resolve endpoint name from gateway_path '{normalized_gateway_path}'."
        )
    if model_name is not None:
        return (model_name,)
    raise ResourceNameResolutionError(
        "Missing required parameter 'gateway_path' for authorization."
    )


def _parse_gateway_model_definition_id_to_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    return (
        _resolve_gateway_model_definition_name_from_id(
            _get_request_param(request_context, "model_definition_id")
        ),
    )


def resolve_gateway_model_definition_names_for_use(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    """Resolve gateway model-definition names referenced by endpoint mutation requests."""
    model_definition_ids: list[str] = _get_request_param_values(
        request_context, "model_definition_id"
    )
    if isinstance(request_context.json_body, dict):
        model_config = request_context.json_body.get("model_config")
        if isinstance(model_config, dict):
            model_definition_ids.extend(
                _normalize_values(model_config.get("model_definition_id"))
            )
        model_configs = request_context.json_body.get("model_configs")
        if isinstance(model_configs, list):
            for model_config in model_configs:
                if isinstance(model_config, dict):
                    model_definition_ids.extend(
                        _normalize_values(model_config.get("model_definition_id"))
                    )
    return tuple(
        _resolve_gateway_model_definition_name_from_id(mid)
        for mid in dict.fromkeys(model_definition_ids)
    )


def resolve_gateway_secret_names_for_use(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    """Resolve gateway secret names referenced by model-definition mutation requests."""
    secret_ids = _get_request_param_values(request_context, "secret_id")
    return tuple(
        _resolve_gateway_secret_name_from_secret_id(sid) for sid in dict.fromkeys(secret_ids)
    )


def _get_graphql_inputs(
    request_context: AuthorizationRequest, allowed_fields: set[str]
) -> tuple[dict[str, object], ...]:
    payload = request_context.graphql_payload
    if not isinstance(payload, dict):
        raise ResourceNameResolutionError("GraphQL payload is required for authorization.")

    query_string = payload.get("query")
    if not isinstance(query_string, str) or not query_string.strip():
        raise ResourceNameResolutionError("GraphQL query is required for authorization.")

    variables = payload.get("variables")
    from mlflow_kubernetes_plugins.auth.graphql import extract_graphql_query_info

    # This reparses the GraphQL payload on the fallback-only path so the name-scoped
    # parser can inspect the same root-field arguments used to derive authorization rules.
    query_info = extract_graphql_query_info(
        query_string,
        variables if isinstance(variables, dict) else None,
    )
    matching_fields = [
        root_field
        for root_field in query_info.root_field_inputs
        if root_field.field_name in allowed_fields
    ]
    if not matching_fields:
        raise ResourceNameResolutionError(
            "GraphQL authorization requires at least one supported root field."
        )

    input_payloads: list[dict[str, object]] = []
    for root_field in matching_fields:
        input_payload = root_field.args.get("input")
        if not isinstance(input_payload, dict):
            raise ResourceNameResolutionError("GraphQL authorization requires an input object.")
        input_payloads.append(input_payload)
    return tuple(input_payloads)


def _parse_graphql_experiment_id_to_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    names: list[str] = []
    for input_payload in _get_graphql_inputs(request_context, {"mlflowGetExperiment"}):
        experiment_id = _normalize_string(
            input_payload.get("experimentId") or input_payload.get("experiment_id")
        )
        if experiment_id is None:
            raise ResourceNameResolutionError(
                "GraphQL experiment query did not include an experiment ID."
            )
        names.append(_resolve_experiment_name_from_experiment_id(experiment_id))
    return tuple(names)


def _parse_graphql_run_id_to_experiment_name(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    names: list[str] = []
    for input_payload in _get_graphql_inputs(
        request_context, {"mlflowGetRun", "mlflowListArtifacts"}
    ):
        run_id = _normalize_string(input_payload.get("runId") or input_payload.get("run_id"))
        if run_id is None:
            run_id = _normalize_string(input_payload.get("runUuid") or input_payload.get("run_uuid"))
        if run_id is None:
            raise ResourceNameResolutionError("GraphQL run query did not include a run ID.")
        names.append(_resolve_experiment_name_from_run_id(run_id))
    return tuple(names)


def _parse_graphql_run_ids_to_experiment_names(
    request_context: AuthorizationRequest,
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for input_payload in _get_graphql_inputs(
        request_context, {"mlflowGetMetricHistoryBulkInterval"}
    ):
        run_ids = input_payload.get("runIds") or input_payload.get("run_ids")
        if not isinstance(run_ids, list):
            raise ResourceNameResolutionError(
                "GraphQL metric history query did not include run IDs."
            )
        for run_id in run_ids:
            normalized_run_id = _normalize_string(run_id)
            if normalized_run_id is None:
                continue
            experiment_name = _resolve_experiment_name_from_run_id(normalized_run_id)
            if experiment_name not in seen:
                names.append(experiment_name)
                seen.add(experiment_name)

    if not names:
        raise ResourceNameResolutionError(
            "GraphQL metric history query did not include any usable run IDs."
        )
    return tuple(names)


RESOURCE_NAME_PARSERS: dict[str, "Callable[[AuthorizationRequest], tuple[str, ...]]"] = {
    RESOURCE_NAME_PARSER_EXPERIMENT_NAME: _parse_experiment_name,
    RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME: _parse_experiment_id_to_name,
    RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES: _parse_experiment_ids_to_names,
    RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME: _parse_new_experiment_name,
    RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME: _parse_run_id_to_experiment_name,
    RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME: _parse_model_id_to_experiment_name,
    RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME: _parse_dataset_id_to_name,
    RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME: _parse_registered_model_name,
    RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME: _parse_new_registered_model_name,
    RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME: (
        _parse_webhook_id_to_registered_model_name
    ),
    RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME: _parse_job_id_to_experiment_name,
    RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME: _parse_issue_id_to_experiment_name,
    RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME: (
        _parse_trace_request_id_to_experiment_name
    ),
    RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME: _parse_trace_id_to_experiment_name,
    RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME: _parse_trace_v3_experiment_id_to_name,
    RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME: _parse_artifact_experiment_id_to_name,
    RESOURCE_NAME_PARSER_OTEL_EXPERIMENT_ID_HEADER_TO_NAME: (
        _parse_otel_experiment_id_header_to_name
    ),
    RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME: _parse_gateway_secret_id_to_name,
    RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME: _parse_gateway_endpoint_id_to_name,
    RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME: _parse_gateway_proxy_endpoint_name,
    RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME: (
        _parse_gateway_model_definition_id_to_name
    ),
    RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME: _parse_graphql_experiment_id_to_name,
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME: (
        _parse_graphql_run_id_to_experiment_name
    ),
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES: (
        _parse_graphql_run_ids_to_experiment_names
    ),
}


def resolve_resource_names(
    request_context: AuthorizationRequest,
    parser_ids: Iterable[str],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for parser_id in parser_ids:
        parser = RESOURCE_NAME_PARSERS.get(parser_id)
        if parser is None:
            raise ResourceNameResolutionError(
                f"Unknown resource-name parser '{parser_id}'."
            )
        for name in parser(request_context):
            normalized_name = _normalize_string(name)
            if normalized_name is None or normalized_name in seen:
                continue
            names.append(normalized_name)
            seen.add(normalized_name)
    if not names:
        raise ResourceNameResolutionError("No resource names could be resolved.")
    return tuple(names)


def update_experiment_name_cache(experiment_id: str, experiment_name: str) -> None:
    normalized_id = _normalize_string(experiment_id)
    normalized_name = _normalize_string(experiment_name)
    if normalized_id is None or normalized_name is None:
        return
    _experiment_name_cache.set(normalized_id, normalized_name)


def apply_response_cache_updates(
    request_context: AuthorizationRequest,
    rules: Iterable["AuthorizationRule"],
    *,
    status_code: int,
) -> None:
    """Refresh short-lived name caches from successful write responses.

    Today this is used to keep experiment-id-to-name fallback lookups current after renames so
    follow-up resourceName authorization checks can avoid a tracking-store round trip.
    """
    if status_code >= 400:
        return

    parser_ids = {parser_id for rule in rules for parser_id in rule.resource_name_parsers}
    if {
        RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
        RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME,
    }.issubset(parser_ids):
        try:
            update_experiment_name_cache(
                _get_request_param(request_context, "experiment_id"),
                _get_request_param(request_context, "new_name"),
            )
        except ResourceNameResolutionError:
            return


__all__ = [
    "RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES",
    "RESOURCE_NAME_PARSER_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME",
    "RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES",
    "RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME",
    "RESOURCE_NAME_PARSER_OTEL_EXPERIMENT_ID_HEADER_TO_NAME",
    "RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME",
    "RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME",
    "RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME",
    "RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME",
    "RESOURCE_NAME_PARSERS",
    "ResourceNameResolutionError",
    "apply_response_cache_updates",
    "resolve_gateway_model_definition_names_for_use",
    "resolve_gateway_secret_names_for_use",
    "resolve_resource_names",
    "update_experiment_name_cache",
]
