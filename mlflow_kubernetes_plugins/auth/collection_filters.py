"""Collection request/response filtering for fine-grained authorization."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mlflow_kubernetes_plugins.auth.constants import (
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
)
from mlflow_kubernetes_plugins.auth.request_context import AuthorizationRequest
from mlflow_kubernetes_plugins.auth.resource_names import (
    ResourceNameResolutionError,
    _normalize_string,
    _resolve_experiment_name_from_experiment_id,
    _resolve_experiment_name_from_run_id,
)

if TYPE_CHECKING:
    from mlflow_kubernetes_plugins.auth.authorizer import KubernetesAuthorizer
    from mlflow_kubernetes_plugins.auth.core import _RequestIdentity
    from mlflow_kubernetes_plugins.auth.rules import AuthorizationRule


COLLECTION_POLICY_BROAD_ONLY = "broad_only"
COLLECTION_POLICY_GRAPHQL_FILTER = "graphql_filter"
COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS_BODY = "request_filter_experiment_ids_body"
COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY = "request_filter_experiment_id_body"
COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_QUERY_GET_BODY_POST = (
    "request_filter_experiment_id_query_get_body_post"
)
COLLECTION_POLICY_REQUEST_RUN_ID_QUERY = "request_filter_run_id_query"
COLLECTION_POLICY_REQUEST_RUN_IDS_QUERY_GET_BODY_ON_EMPTY_QUERY = (
    "request_filter_run_ids_query_get_body_on_empty_query"
)
COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS_BODY = "request_filter_trace_locations_body"
COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES = "response_filter_dataset_summaries"
COLLECTION_POLICY_RESPONSE_EXPERIMENTS = "response_filter_experiments"
COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS = "response_filter_registered_models"
COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS = "response_filter_model_versions"
COLLECTION_POLICY_RESPONSE_TRACES = "response_filter_traces"

_EXPERIMENT_READ_RULE = (RESOURCE_EXPERIMENTS, "get")
_REGISTERED_MODEL_READ_RULE = (RESOURCE_REGISTERED_MODELS, "get")

_MISSING = object()
_INVALID = object()

_REQUEST_FILTER_POLICIES = {
    COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS_BODY,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_QUERY_GET_BODY_POST,
    COLLECTION_POLICY_REQUEST_RUN_ID_QUERY,
    COLLECTION_POLICY_REQUEST_RUN_IDS_QUERY_GET_BODY_ON_EMPTY_QUERY,
    COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS_BODY,
}
_RESPONSE_FILTER_POLICIES = {
    COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES,
    COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
    COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS,
    COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS,
    COLLECTION_POLICY_RESPONSE_TRACES,
}


def is_request_filter_policy(policy: str | None) -> bool:
    return policy in _REQUEST_FILTER_POLICIES


def is_response_filter_policy(policy: str | None) -> bool:
    return policy in _RESPONSE_FILTER_POLICIES


def is_graphql_collection_policy(policy: str | None) -> bool:
    return policy == COLLECTION_POLICY_GRAPHQL_FILTER


def response_filter_policies(rules: list["AuthorizationRule"]) -> set[str]:
    return {
        rule.collection_policy for rule in rules if is_response_filter_policy(rule.collection_policy)
    }


def can_skip_response_collection_filters(
    rules: list["AuthorizationRule"],
    *,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    """Return True when the caller already has the broad permissions for every response filter."""
    applicable_rules = [rule for rule in rules if is_response_filter_policy(rule.collection_policy)]
    return bool(applicable_rules) and all(
        bool(rule.resource)
        and bool(rule.verb)
        and authorizer.is_allowed(
            identity,
            rule.resource,
            rule.verb,
            workspace_name,
            rule.subresource,
        )
        for rule in applicable_rules
    )


def _is_allowed_named_resource(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    permission: tuple[str, str],
    resource_name: str,
) -> bool:
    resource, verb = permission
    return authorizer.is_allowed(
        identity,
        resource,
        verb,
        workspace_name,
        resource_name=resource_name,
    )


def _first_present_value(mapping: dict[str, object], *candidate_keys: str) -> object | None:
    """Return the first populated field across snake_case and camelCase payload variants."""
    for key in candidate_keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _can_read_experiment_id(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    experiment_id: str,
) -> bool:
    try:
        experiment_name = _resolve_experiment_name_from_experiment_id(experiment_id)
    except ResourceNameResolutionError:
        return False
    return _is_allowed_named_resource(
        authorizer,
        identity,
        workspace_name,
        _EXPERIMENT_READ_RULE,
        experiment_name,
    )


def _can_read_run_id(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    run_id: str,
) -> bool:
    try:
        experiment_name = _resolve_experiment_name_from_run_id(run_id)
    except ResourceNameResolutionError:
        return False
    return _is_allowed_named_resource(
        authorizer,
        identity,
        workspace_name,
        _EXPERIMENT_READ_RULE,
        experiment_name,
    )


def filter_readable_experiment_ids(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    experiment_ids: list[str],
) -> list[str]:
    readable_ids: list[str] = []
    for experiment_id in experiment_ids:
        normalized_id = _normalize_string(experiment_id)
        if normalized_id is None:
            continue
        if _can_read_experiment_id(authorizer, identity, workspace_name, normalized_id):
            readable_ids.append(normalized_id)
    return readable_ids


def filter_readable_run_ids(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    run_ids: list[str],
) -> list[str]:
    readable_ids: list[str] = []
    for run_id in run_ids:
        normalized_id = _normalize_string(run_id)
        if normalized_id is None:
            continue
        if _can_read_run_id(authorizer, identity, workspace_name, normalized_id):
            readable_ids.append(normalized_id)
    return readable_ids


def _contains_any_key(mapping: dict[str, object] | None, *candidate_keys: str) -> bool:
    return mapping is not None and any(key in mapping for key in candidate_keys)


def _extract_mapping_field(
    mapping: dict[str, object] | None, *candidate_keys: str
) -> tuple[object, str | None]:
    if mapping is None:
        return _MISSING, None
    present_keys = [key for key in candidate_keys if key in mapping]
    if not present_keys:
        return _MISSING, None
    if len(present_keys) > 1:
        return _INVALID, None
    key = present_keys[0]
    return mapping.get(key), key


def _normalize_strict_request_values(value: object, *, allow_scalar: bool) -> list[str] | object:
    if isinstance(value, list):
        raw_values = value
    elif value is None or not allow_scalar:
        return _INVALID
    else:
        raw_values = [value]

    normalized_values: list[str] = []
    for raw_value in raw_values:
        normalized_value = _normalize_string(raw_value)
        if normalized_value is None:
            return _INVALID
        normalized_values.append(normalized_value)
    return normalized_values


def _normalize_strict_request_value(value: object) -> str | object:
    if value is None or isinstance(value, list):
        return _INVALID
    normalized_value = _normalize_string(value)
    return normalized_value if normalized_value is not None else _INVALID


def _filter_request_experiment_ids_body(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body = request_context.json_body if isinstance(request_context.json_body, dict) else None

    raw_body_value, body_key = _extract_mapping_field(body, "experiment_ids", "experimentIds")
    if raw_body_value is _MISSING:
        return request_context, False
    if raw_body_value is _INVALID or body_key is None:
        return request_context, False

    body_experiment_ids = _normalize_strict_request_values(raw_body_value, allow_scalar=False)
    if body_experiment_ids is _INVALID:
        return request_context, False
    assert isinstance(body_experiment_ids, list)

    readable_body_ids = filter_readable_experiment_ids(
        authorizer,
        identity,
        workspace_name,
        body_experiment_ids,
    )
    if not readable_body_ids:
        return request_context, False

    filtered_body = dict(body)
    filtered_body[body_key] = readable_body_ids
    return replace(request_context, json_body=filtered_body), True


def _filter_request_single_experiment_id_body(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body = request_context.json_body if isinstance(request_context.json_body, dict) else None

    # Protobuf JSON parsing accepts both the proto field name and its json_name alias in the body.
    raw_body_value, _ = _extract_mapping_field(body, "experiment_id", "experimentId")
    if raw_body_value is _MISSING:
        return request_context, False
    if raw_body_value is _INVALID:
        return request_context, False

    body_experiment_id = _normalize_strict_request_value(raw_body_value)
    if body_experiment_id is _INVALID or not _can_read_experiment_id(
        authorizer, identity, workspace_name, body_experiment_id
    ):
        return request_context, False

    return request_context, True


def _filter_request_single_experiment_id_query(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    raw_query_value, _ = _extract_mapping_field(request_context.query_params, "experiment_id")
    if raw_query_value is _MISSING:
        return request_context, False
    if raw_query_value is _INVALID:
        return request_context, False

    query_experiment_id = _normalize_strict_request_value(raw_query_value)
    if query_experiment_id is _INVALID or not _can_read_experiment_id(
        authorizer, identity, workspace_name, query_experiment_id
    ):
        return request_context, False

    return request_context, True


def _filter_request_single_experiment_id_query_get_body_post(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    method = request_context.method.upper()
    if method == "GET":
        if request_context.query_params:
            return _filter_request_single_experiment_id_query(
                request_context, authorizer, identity, workspace_name
            )
        # Upstream _get_request_message() falls back to the JSON body on GET when args are empty.
        return _filter_request_single_experiment_id_body(
            request_context, authorizer, identity, workspace_name
        )
    if method == "POST":
        return _filter_request_single_experiment_id_body(
            request_context, authorizer, identity, workspace_name
        )
    return request_context, False


def _filter_request_run_ids_body(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body = request_context.json_body if isinstance(request_context.json_body, dict) else None

    raw_body_value, body_key = _extract_mapping_field(body, "run_ids", "runIds")
    if raw_body_value is _MISSING:
        return request_context, False
    if raw_body_value is _INVALID or body_key is None:
        return request_context, False

    body_run_ids = _normalize_strict_request_values(raw_body_value, allow_scalar=False)
    if body_run_ids is _INVALID:
        return request_context, False
    assert isinstance(body_run_ids, list)

    readable_body_ids = filter_readable_run_ids(authorizer, identity, workspace_name, body_run_ids)
    if not readable_body_ids:
        return request_context, False

    filtered_body = dict(body)
    filtered_body[body_key] = readable_body_ids
    return replace(request_context, json_body=filtered_body), True


def _filter_request_run_ids_query_get_body_on_empty_query(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    if request_context.method.upper() == "GET" and not request_context.query_params:
        return _filter_request_run_ids_body(request_context, authorizer, identity, workspace_name)
    return _filter_request_run_ids_query_run_ids(
        request_context, authorizer, identity, workspace_name
    )


def _filter_request_run_ids_query(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    *,
    query_key: str,
) -> tuple[AuthorizationRequest, bool]:
    raw_query_value, _ = _extract_mapping_field(request_context.query_params, query_key)
    if raw_query_value is _MISSING:
        return request_context, False
    if raw_query_value is _INVALID:
        return request_context, False

    query_run_ids = _normalize_strict_request_values(raw_query_value, allow_scalar=True)
    if query_run_ids is _INVALID:
        return request_context, False
    assert isinstance(query_run_ids, list)

    readable_query_ids = filter_readable_run_ids(authorizer, identity, workspace_name, query_run_ids)
    if not readable_query_ids:
        return request_context, False

    filtered_query_params = dict(request_context.query_params)
    filtered_query_params[query_key] = readable_query_ids
    return replace(request_context, query_params=filtered_query_params), True


def _filter_request_run_ids_query_run_ids(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    return _filter_request_run_ids_query(
        request_context,
        authorizer,
        identity,
        workspace_name,
        query_key="run_ids",
    )


def _filter_request_run_ids_query_run_id(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    return _filter_request_run_ids_query(
        request_context,
        authorizer,
        identity,
        workspace_name,
        query_key="run_id",
    )


def _filter_request_trace_locations_body(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body = request_context.json_body if isinstance(request_context.json_body, dict) else None
    if body is None:
        return request_context, False

    raw_locations, body_key = _extract_mapping_field(body, "locations")
    if raw_locations is _MISSING:
        return request_context, False
    if raw_locations is _INVALID or body_key is None:
        return request_context, False
    if not isinstance(raw_locations, list):
        return request_context, False

    filtered_locations: list[dict[str, object]] = []
    for location in raw_locations:
        if not isinstance(location, dict):
            return request_context, False
        has_mlflow_location = _contains_any_key(location, "mlflow_experiment", "mlflowExperiment")
        has_inference_table = _contains_any_key(location, "inference_table", "inferenceTable")
        # A single TraceLocation should not try to specify multiple upstream location variants.
        if has_mlflow_location and has_inference_table:
            return request_context, False
        if has_inference_table:
            continue
        mlflow_location, _ = _extract_mapping_field(location, "mlflow_experiment", "mlflowExperiment")
        if (mlflow_location is _MISSING or mlflow_location is _INVALID) or not isinstance(
            mlflow_location, dict
        ):
            return request_context, False
        experiment_id_value, _ = _extract_mapping_field(
            mlflow_location, "experiment_id", "experimentId"
        )
        if experiment_id_value is _MISSING or experiment_id_value is _INVALID:
            return request_context, False
        experiment_id = _normalize_strict_request_value(experiment_id_value)
        if experiment_id is _INVALID:
            return request_context, False
        if _can_read_experiment_id(authorizer, identity, workspace_name, experiment_id):
            filtered_locations.append(location)

    if not filtered_locations:
        return request_context, False

    filtered_body = dict(body)
    filtered_body[body_key] = filtered_locations
    return replace(request_context, json_body=filtered_body), True


def apply_request_collection_filter(
    request_context: AuthorizationRequest,
    policy: str | None,
    *,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    """Narrow a collection request to resources the caller can access.

    Returns ``(updated_request_context, applied)`` where *applied* is ``True``
    when the filter found identifiers in the request and narrowed them to the
    subset the caller is authorized for.
    """
    if policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS_BODY:
        return _filter_request_experiment_ids_body(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY:
        return _filter_request_single_experiment_id_body(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_QUERY_GET_BODY_POST:
        return _filter_request_single_experiment_id_query_get_body_post(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_RUN_ID_QUERY:
        return _filter_request_run_ids_query_run_id(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_RUN_IDS_QUERY_GET_BODY_ON_EMPTY_QUERY:
        return _filter_request_run_ids_query_get_body_on_empty_query(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS_BODY:
        return _filter_request_trace_locations_body(
            request_context, authorizer, identity, workspace_name
        )
    return request_context, False


def _filter_payload_experiments(
    payload: dict[str, object],
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    experiments = payload.get("experiments")
    if experiments is None:
        return True
    if not isinstance(experiments, list):
        return False
    payload["experiments"] = [
        experiment
        for experiment in experiments
        if isinstance(experiment, dict)
        and (
            experiment_name := _normalize_string(experiment.get("name"))
        )
        and _is_allowed_named_resource(
            authorizer,
            identity,
            workspace_name,
            _EXPERIMENT_READ_RULE,
            experiment_name,
        )
    ]
    return True


def _filter_payload_dataset_summaries(
    payload: dict[str, object],
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    enforced = True
    for payload_key in ("datasets", "dataset_summaries"):
        value = payload.get(payload_key)
        if value is None:
            continue
        if not isinstance(value, list):
            enforced = False
            continue
        filtered_summaries: list[dict[str, object]] = []
        for summary in value:
            if not isinstance(summary, dict):
                continue
            experiment_id = _normalize_string(
                _first_present_value(summary, "experiment_id", "experimentId")
            )
            if experiment_id is None:
                continue
            if _can_read_experiment_id(authorizer, identity, workspace_name, experiment_id):
                filtered_summaries.append(summary)
        payload[payload_key] = filtered_summaries
    return enforced


def _filter_payload_registered_models(
    payload: dict[str, object],
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    registered_models = payload.get("registered_models")
    if registered_models is None:
        return True
    if not isinstance(registered_models, list):
        return False
    payload["registered_models"] = [
        model
        for model in registered_models
        if isinstance(model, dict)
        and (
            model_name := _normalize_string(model.get("name"))
        )
        and _is_allowed_named_resource(
            authorizer,
            identity,
            workspace_name,
            _REGISTERED_MODEL_READ_RULE,
            model_name,
        )
    ]
    return True


def _filter_payload_model_versions(
    payload: dict[str, object],
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    model_versions = payload.get("model_versions")
    if model_versions is None:
        return True
    if not isinstance(model_versions, list):
        return False
    payload["model_versions"] = [
        model_version
        for model_version in model_versions
        if isinstance(model_version, dict)
        and (
            model_name := _normalize_string(model_version.get("name"))
        )
        and _is_allowed_named_resource(
            authorizer,
            identity,
            workspace_name,
            _REGISTERED_MODEL_READ_RULE,
            model_name,
        )
    ]
    return True


def _trace_experiment_id(trace: dict[str, object]) -> str | None:
    """Extract an experiment ID from trace payloads across MLflow's mixed field spellings."""
    trace_info = (
        trace.get("trace_info")
        or trace.get("traceInfo")
        or trace.get("info")
    )
    if isinstance(trace_info, dict):
        experiment_id = _normalize_string(
            _first_present_value(trace_info, "experiment_id", "experimentId")
        )
        if experiment_id is not None:
            return experiment_id
        trace_location = _first_present_value(trace_info, "trace_location", "traceLocation")
        if isinstance(trace_location, dict):
            mlflow_location = _first_present_value(
                trace_location, "mlflow_experiment", "mlflowExperiment"
            )
            if isinstance(mlflow_location, dict):
                return _normalize_string(
                    _first_present_value(mlflow_location, "experiment_id", "experimentId")
                )
    return None


def _filter_payload_traces(
    payload: dict[str, object],
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> bool:
    enforced = True
    for payload_key in ("traces", "trace_infos"):
        value = payload.get(payload_key)
        if value is None:
            continue
        if not isinstance(value, list):
            enforced = False
            continue
        filtered_traces: list[dict[str, object]] = []
        for trace in value:
            if not isinstance(trace, dict):
                continue
            experiment_id = _trace_experiment_id(trace)
            if experiment_id is None:
                continue
            if _can_read_experiment_id(authorizer, identity, workspace_name, experiment_id):
                filtered_traces.append(trace)
        payload[payload_key] = filtered_traces
    return enforced


def apply_response_collection_filters(
    payload: dict[str, object],
    rules: list["AuthorizationRule"],
    *,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[dict[str, object], bool]:
    """Filter response collections and report whether all filters could enforce.

    Returns ``(filtered_payload, enforceable)`` where *enforceable* is ``False``
    when any expected collection key was present but had an unrecognizable type,
    meaning the filter could not guarantee that unauthorized data was removed.
    """
    filtered_payload = dict(payload)
    enforceable = True
    applied_policies = response_filter_policies(rules)
    for policy in applied_policies:
        if policy == COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES:
            enforceable &= _filter_payload_dataset_summaries(
                filtered_payload, authorizer, identity, workspace_name
            )
        elif policy == COLLECTION_POLICY_RESPONSE_EXPERIMENTS:
            enforceable &= _filter_payload_experiments(
                filtered_payload, authorizer, identity, workspace_name
            )
        elif policy == COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS:
            enforceable &= _filter_payload_registered_models(
                filtered_payload, authorizer, identity, workspace_name
            )
        elif policy == COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS:
            enforceable &= _filter_payload_model_versions(
                filtered_payload, authorizer, identity, workspace_name
            )
        elif policy == COLLECTION_POLICY_RESPONSE_TRACES:
            enforceable &= _filter_payload_traces(
                filtered_payload, authorizer, identity, workspace_name
            )
    return filtered_payload, enforceable


def filter_graphql_experiment_ids(
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
    experiment_ids: list[str],
) -> list[str]:
    return filter_readable_experiment_ids(authorizer, identity, workspace_name, experiment_ids)


def filter_graphql_model_versions_result(
    result: object,
    *,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
):
    if not hasattr(result, "model_versions") or result.model_versions is None:
        return None
    filtered = [
        model_version
        for model_version in result.model_versions
        if (
            model_name := _normalize_string(getattr(model_version, "name", None))
        )
        and _is_allowed_named_resource(
            authorizer,
            identity,
            workspace_name,
            _REGISTERED_MODEL_READ_RULE,
            model_name,
        )
    ]
    del result.model_versions[:]
    result.model_versions.extend(filtered)
    return result


__all__ = [
    "COLLECTION_POLICY_BROAD_ONLY",
    "COLLECTION_POLICY_GRAPHQL_FILTER",
    "COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY",
    "COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_QUERY_GET_BODY_POST",
    "COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS_BODY",
    "COLLECTION_POLICY_REQUEST_RUN_ID_QUERY",
    "COLLECTION_POLICY_REQUEST_RUN_IDS_QUERY_GET_BODY_ON_EMPTY_QUERY",
    "COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS_BODY",
    "COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES",
    "COLLECTION_POLICY_RESPONSE_EXPERIMENTS",
    "COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS",
    "COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS",
    "COLLECTION_POLICY_RESPONSE_TRACES",
    "apply_request_collection_filter",
    "apply_response_collection_filters",
    "can_skip_response_collection_filters",
    "filter_graphql_experiment_ids",
    "filter_graphql_model_versions_result",
    "filter_readable_run_ids",
    "is_graphql_collection_policy",
    "is_request_filter_policy",
    "is_response_filter_policy",
    "response_filter_policies",
]
