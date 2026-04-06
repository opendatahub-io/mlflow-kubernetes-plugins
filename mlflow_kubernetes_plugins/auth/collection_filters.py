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
COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS = "request_filter_experiment_ids"
COLLECTION_POLICY_REQUEST_EXPERIMENT_ID = "request_filter_experiment_id"
COLLECTION_POLICY_REQUEST_RUN_IDS = "request_filter_run_ids"
COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS = "request_filter_trace_locations"
COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES = "response_filter_dataset_summaries"
COLLECTION_POLICY_RESPONSE_EXPERIMENTS = "response_filter_experiments"
COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS = "response_filter_registered_models"
COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS = "response_filter_model_versions"
COLLECTION_POLICY_RESPONSE_TRACES = "response_filter_traces"

_EXPERIMENT_READ_RULE = (RESOURCE_EXPERIMENTS, "get")
_REGISTERED_MODEL_READ_RULE = (RESOURCE_REGISTERED_MODELS, "get")

_REQUEST_FILTER_POLICIES = {
    COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    COLLECTION_POLICY_REQUEST_RUN_IDS,
    COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS,
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


def _request_values(request_context: AuthorizationRequest, key: str) -> list[str]:
    """Collect a request field from JSON body and query params.

    These request filters intentionally consider both body and query params because some MLflow
    routes still send identifiers in the query string even on mutating requests.
    """
    values: list[str] = []
    if isinstance(request_context.json_body, dict):
        body_value = request_context.json_body.get(key)
        if isinstance(body_value, list):
            values.extend(str(value) for value in body_value)
        elif body_value is not None:
            values.append(str(body_value))

    query_value = request_context.query_params.get(key)
    if isinstance(query_value, list):
        values.extend(str(value) for value in query_value)
    elif query_value is not None:
        values.append(str(query_value))

    return values


def _normalize_request_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [normalized for item in value if (normalized := _normalize_string(item))]
    if value is None:
        return []
    normalized = _normalize_string(value)
    return [normalized] if normalized is not None else []


def _normalize_request_value(value: object) -> str | None:
    values = _normalize_request_values(value)
    if not values:
        return None
    first_value = values[0]
    return first_value if all(value == first_value for value in values[1:]) else None


def _filter_request_experiment_ids(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body_has_experiment_ids = (
        isinstance(request_context.json_body, dict) and "experiment_ids" in request_context.json_body
    )
    query_has_experiment_ids = "experiment_ids" in request_context.query_params
    if not body_has_experiment_ids and not query_has_experiment_ids:
        return request_context, False

    body_experiment_ids = (
        _normalize_request_values(request_context.json_body.get("experiment_ids"))
        if body_has_experiment_ids and isinstance(request_context.json_body, dict)
        else []
    )
    query_experiment_ids = (
        _normalize_request_values(request_context.query_params.get("experiment_ids"))
        if query_has_experiment_ids
        else []
    )
    readable_body_ids = filter_readable_experiment_ids(
        authorizer,
        identity,
        workspace_name,
        body_experiment_ids,
    )
    readable_query_ids = filter_readable_experiment_ids(
        authorizer,
        identity,
        workspace_name,
        query_experiment_ids,
    )
    if not readable_body_ids and not readable_query_ids:
        return request_context, False

    updated_request_context = request_context
    if body_has_experiment_ids and isinstance(request_context.json_body, dict):
        filtered_body = dict(request_context.json_body)
        filtered_body["experiment_ids"] = readable_body_ids
        updated_request_context = replace(updated_request_context, json_body=filtered_body)

    if query_has_experiment_ids:
        filtered_query_params = dict(request_context.query_params)
        if readable_query_ids:
            filtered_query_params["experiment_ids"] = readable_query_ids
        else:
            filtered_query_params.pop("experiment_ids", None)
        updated_request_context = replace(updated_request_context, query_params=filtered_query_params)

    return updated_request_context, True


def _filter_request_single_experiment_id(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body_has_experiment_id = (
        isinstance(request_context.json_body, dict) and "experiment_id" in request_context.json_body
    )
    query_has_experiment_id = "experiment_id" in request_context.query_params
    if not body_has_experiment_id and not query_has_experiment_id:
        return request_context, False

    body_experiment_id = (
        _normalize_request_value(request_context.json_body.get("experiment_id"))
        if body_has_experiment_id and isinstance(request_context.json_body, dict)
        else None
    )
    query_experiment_id = (
        _normalize_request_value(request_context.query_params.get("experiment_id"))
        if query_has_experiment_id
        else None
    )

    if body_has_experiment_id and (
        body_experiment_id is None
        or not _can_read_experiment_id(authorizer, identity, workspace_name, body_experiment_id)
    ):
        return request_context, False
    if query_has_experiment_id and (
        query_experiment_id is None
        or not _can_read_experiment_id(authorizer, identity, workspace_name, query_experiment_id)
    ):
        return request_context, False

    return request_context, True


def _filter_request_run_ids(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    body_has_run_ids = isinstance(request_context.json_body, dict) and "run_ids" in request_context.json_body
    query_has_run_ids = "run_ids" in request_context.query_params
    if not body_has_run_ids and not query_has_run_ids:
        return request_context, False

    body_run_ids = (
        _normalize_request_values(request_context.json_body.get("run_ids"))
        if body_has_run_ids and isinstance(request_context.json_body, dict)
        else []
    )
    query_run_ids = (
        _normalize_request_values(request_context.query_params.get("run_ids")) if query_has_run_ids else []
    )
    readable_body_ids = filter_readable_run_ids(authorizer, identity, workspace_name, body_run_ids)
    readable_query_ids = filter_readable_run_ids(authorizer, identity, workspace_name, query_run_ids)
    if not readable_body_ids and not readable_query_ids:
        return request_context, False

    updated_request_context = request_context
    if body_has_run_ids and isinstance(request_context.json_body, dict):
        filtered_body = dict(request_context.json_body)
        filtered_body["run_ids"] = readable_body_ids
        updated_request_context = replace(updated_request_context, json_body=filtered_body)

    if query_has_run_ids:
        filtered_query_params = dict(request_context.query_params)
        if readable_query_ids:
            filtered_query_params["run_ids"] = readable_query_ids
        else:
            filtered_query_params.pop("run_ids", None)
        updated_request_context = replace(updated_request_context, query_params=filtered_query_params)

    return updated_request_context, True


def _filter_request_trace_locations(
    request_context: AuthorizationRequest,
    authorizer: "KubernetesAuthorizer",
    identity: "_RequestIdentity",
    workspace_name: str,
) -> tuple[AuthorizationRequest, bool]:
    if not isinstance(request_context.json_body, dict):
        return request_context, False
    raw_locations = request_context.json_body.get("locations")
    if not isinstance(raw_locations, list):
        return request_context, False

    filtered_locations: list[dict[str, object]] = []
    for location in raw_locations:
        if not isinstance(location, dict):
            continue
        mlflow_location = _first_present_value(location, "mlflow_experiment", "mlflowExperiment")
        if not isinstance(mlflow_location, dict):
            continue
        experiment_id = _normalize_string(
            _first_present_value(mlflow_location, "experiment_id", "experimentId")
        )
        if experiment_id is None:
            continue
        if _can_read_experiment_id(authorizer, identity, workspace_name, experiment_id):
            filtered_locations.append(location)

    if not filtered_locations:
        return request_context, False

    filtered_body = dict(request_context.json_body)
    filtered_body["locations"] = filtered_locations
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
    if policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS:
        return _filter_request_experiment_ids(request_context, authorizer, identity, workspace_name)
    if policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_ID:
        return _filter_request_single_experiment_id(
            request_context, authorizer, identity, workspace_name
        )
    if policy == COLLECTION_POLICY_REQUEST_RUN_IDS:
        return _filter_request_run_ids(request_context, authorizer, identity, workspace_name)
    if policy == COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS:
        return _filter_request_trace_locations(request_context, authorizer, identity, workspace_name)
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
    "COLLECTION_POLICY_REQUEST_EXPERIMENT_ID",
    "COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS",
    "COLLECTION_POLICY_REQUEST_RUN_IDS",
    "COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS",
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
