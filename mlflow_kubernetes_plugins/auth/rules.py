"""Explicit authorization rule tables for the Kubernetes auth plugin."""

from __future__ import annotations

from typing import NamedTuple

from mlflow.protos.mlflow_artifacts_pb2 import (
    AbortMultipartUpload,
    CompleteMultipartUpload,
    CreateMultipartUpload,
    DeleteArtifact,
    DownloadArtifact,
    UploadArtifact,
)
from mlflow.protos.mlflow_artifacts_pb2 import ListArtifacts as ListArtifactsMlflowArtifacts
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    RenameRegisteredModel,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.service_pb2 import (
    AddDatasetToExperiments,
    AttachModelToGatewayEndpoint,
    BatchGetTraces,
    CalculateTraceFilterCorrelation,
    CancelPromptOptimizationJob,
    CreateAssessment,
    CreateDataset,
    CreateExperiment,
    CreateGatewayEndpoint,
    CreateGatewayEndpointBinding,
    CreateGatewayModelDefinition,
    CreateGatewaySecret,
    CreateLoggedModel,
    CreatePromptOptimizationJob,
    CreateRun,
    CreateWorkspace,
    DeleteDataset,
    DeleteDatasetRecords,
    DeleteDatasetTag,
    DeleteExperiment,
    DeleteExperimentTag,
    DeleteGatewayEndpoint,
    DeleteGatewayEndpointBinding,
    DeleteGatewayEndpointTag,
    DeleteGatewayModelDefinition,
    DeleteGatewaySecret,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeletePromptOptimizationJob,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    DeleteTraces,
    DeleteTracesV3,
    DeleteTraceTag,
    DeleteTraceTagV3,
    DeleteWorkspace,
    DetachModelFromGatewayEndpoint,
    EndTrace,
    FinalizeLoggedModel,
    GetAssessmentRequest,
    GetDataset,
    GetDatasetExperimentIds,
    GetDatasetRecords,
    GetExperiment,
    GetExperimentByName,
    GetGatewayEndpoint,
    GetGatewayModelDefinition,
    GetGatewaySecretInfo,
    GetLoggedModel,
    GetMetricHistory,
    GetMetricHistoryBulkInterval,
    GetPromptOptimizationJob,
    GetRun,
    GetScorer,
    GetTraceInfo,
    GetTraceInfoV3,
    GetWorkspace,
    LinkPromptsToTrace,
    LinkTracesToRun,
    ListArtifacts,
    ListGatewayEndpointBindings,
    ListGatewayEndpoints,
    ListGatewayModelDefinitions,
    ListGatewaySecretInfos,
    ListLoggedModelArtifacts,
    ListScorers,
    ListScorerVersions,
    ListWorkspaces,
    LogBatch,
    LogInputs,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogOutputs,
    LogParam,
    QueryTraceMetrics,
    RegisterScorer,
    RemoveDatasetFromExperiments,
    RestoreExperiment,
    RestoreRun,
    SearchDatasets,
    SearchEvaluationDatasets,
    SearchExperiments,
    SearchLoggedModels,
    SearchPromptOptimizationJobs,
    SearchRuns,
    SearchTraces,
    SearchTracesV3,
    SetDatasetTags,
    SetExperimentTag,
    SetGatewayEndpointTag,
    SetLoggedModelTags,
    SetTag,
    SetTraceTag,
    SetTraceTagV3,
    StartTrace,
    StartTraceV3,
    UpdateAssessment,
    UpdateExperiment,
    UpdateGatewayEndpoint,
    UpdateGatewayModelDefinition,
    UpdateGatewaySecret,
    UpdateRun,
    UpdateWorkspace,
    UpsertDatasetRecords,
)
from mlflow.protos.webhooks_pb2 import (
    CreateWebhook,
    DeleteWebhook,
    GetWebhook,
    ListWebhooks,
    TestWebhook,
    UpdateWebhook,
)
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH

from mlflow_kubernetes_plugins.auth.constants import (
    ALLOWED_RESOURCES,
    RESOURCE_ASSISTANTS,
    RESOURCE_DATASETS,
    RESOURCE_EXPERIMENTS,
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
    RESOURCE_GATEWAY_SECRETS,
    RESOURCE_REGISTERED_MODELS,
)
from mlflow_kubernetes_plugins.auth.graphql import _build_graphql_operation_rules


class AuthorizationRule(NamedTuple):
    verb: str | None
    resource: str | None = None
    subresource: str | None = None
    override_run_user: bool = False
    apply_workspace_filter: bool = False
    requires_workspace: bool = True
    deny: bool = False
    workspace_access_check: bool = False


def _assistants_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_ASSISTANTS, **kwargs)


def _datasets_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_DATASETS, **kwargs)


def _experiments_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_EXPERIMENTS, **kwargs)


def _registered_models_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_REGISTERED_MODELS, **kwargs)


def _gateway_secrets_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_GATEWAY_SECRETS, **kwargs)


def _gateway_endpoints_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_GATEWAY_ENDPOINTS, **kwargs)


def _gateway_endpoints_use_rule(**kwargs) -> AuthorizationRule:
    return _gateway_endpoints_rule("create", subresource="use", **kwargs)


def _gateway_model_definitions_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_GATEWAY_MODEL_DEFINITIONS, **kwargs)


def _workspaces_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    if verb is not None and not (
        kwargs.get("deny", False)
        or kwargs.get("apply_workspace_filter", False)
        or kwargs.get("workspace_access_check", False)
    ):
        raise ValueError(
            "_workspaces_rule requires deny=True, apply_workspace_filter=True, or "
            "workspace_access_check=True when verb is set."
        )
    return AuthorizationRule(verb, resource=None, **kwargs)


def _normalize_resource_name(resource: str | None) -> str | None:
    if resource is None:
        return None
    normalized = resource.replace("_", "")
    if normalized not in ALLOWED_RESOURCES:
        raise ValueError(f"Unsupported RBAC resource '{resource}'")
    return normalized


REQUEST_AUTHORIZATION_RULES: dict[type, AuthorizationRule | tuple[AuthorizationRule, ...]] = {
    # Experiments
    CreateExperiment: _experiments_rule("create"),
    GetExperiment: _experiments_rule("get"),
    GetExperimentByName: _experiments_rule("get"),
    DeleteExperiment: _experiments_rule("delete"),
    RestoreExperiment: _experiments_rule("update"),
    UpdateExperiment: _experiments_rule("update"),
    SetExperimentTag: _experiments_rule("update"),
    DeleteExperimentTag: _experiments_rule("update"),
    SearchExperiments: _experiments_rule("list"),
    # Datasets
    AddDatasetToExperiments: (_datasets_rule("update"), _experiments_rule("update")),
    CreateDataset: _datasets_rule("create"),
    DeleteDataset: _datasets_rule("delete"),
    DeleteDatasetRecords: _datasets_rule("update"),
    DeleteDatasetTag: _datasets_rule("update"),
    RemoveDatasetFromExperiments: (_datasets_rule("update"), _experiments_rule("update")),
    SetDatasetTags: _datasets_rule("update"),
    UpsertDatasetRecords: _datasets_rule("update"),
    # Experiment child resources (write operations)
    CreateAssessment: _experiments_rule("update"),
    UpdateAssessment: _experiments_rule("update"),
    CreateRun: _experiments_rule("update", override_run_user=True),
    DeleteRun: _experiments_rule("update"),
    RestoreRun: _experiments_rule("update"),
    UpdateRun: _experiments_rule("update"),
    LogMetric: _experiments_rule("update"),
    LogBatch: _experiments_rule("update"),
    LogModel: _experiments_rule("update"),
    SetTag: _experiments_rule("update"),
    DeleteTag: _experiments_rule("update"),
    LogParam: _experiments_rule("update"),
    CreateLoggedModel: _experiments_rule("update"),
    DeleteLoggedModel: _experiments_rule("update"),
    FinalizeLoggedModel: _experiments_rule("update"),
    DeleteLoggedModelTag: _experiments_rule("update"),
    SetLoggedModelTags: _experiments_rule("update"),
    LogLoggedModelParamsRequest: _experiments_rule("update"),
    RegisterScorer: _experiments_rule("update"),
    DeleteScorer: _experiments_rule("update"),
    EndTrace: _experiments_rule("update"),
    LinkPromptsToTrace: _experiments_rule("update"),
    LinkTracesToRun: _experiments_rule("update"),
    DeleteTraceTag: _experiments_rule("update"),
    DeleteTraceTagV3: _experiments_rule("update"),
    DeleteTraces: _experiments_rule("update"),
    DeleteTracesV3: _experiments_rule("update"),
    SetTraceTag: _experiments_rule("update"),
    SetTraceTagV3: _experiments_rule("update"),
    StartTrace: _experiments_rule("update"),
    StartTraceV3: _experiments_rule("update"),
    LogInputs: _experiments_rule("update"),
    LogOutputs: _experiments_rule("update"),
    CompleteMultipartUpload: _experiments_rule("update"),
    CreateMultipartUpload: _experiments_rule("update"),
    AbortMultipartUpload: _experiments_rule("update"),
    DeleteArtifact: _experiments_rule("update"),
    UploadArtifact: _experiments_rule("update"),
    # Experiment child resources (single-experiment reads)
    GetAssessmentRequest: _experiments_rule("get"),
    GetRun: _experiments_rule("get"),
    GetMetricHistory: _experiments_rule("get"),
    ListArtifacts: _experiments_rule("get"),
    GetLoggedModel: _experiments_rule("get"),
    ListLoggedModelArtifacts: _experiments_rule("get"),
    ListScorers: _experiments_rule("get"),
    GetScorer: _experiments_rule("get"),
    ListScorerVersions: _experiments_rule("get"),
    GetTraceInfo: _experiments_rule("get"),
    GetTraceInfoV3: _experiments_rule("get"),
    DownloadArtifact: _experiments_rule("get"),
    ListArtifactsMlflowArtifacts: _experiments_rule("get"),
    # Dataset reads
    GetDataset: _datasets_rule("get"),
    GetDatasetExperimentIds: _datasets_rule("list"),
    GetDatasetRecords: _datasets_rule("list"),
    SearchDatasets: _datasets_rule("list"),
    SearchEvaluationDatasets: _datasets_rule("list"),
    # Experiment child resources (multi-experiment reads)
    SearchLoggedModels: _experiments_rule("list"),
    BatchGetTraces: _experiments_rule("list"),
    CalculateTraceFilterCorrelation: _experiments_rule("list"),
    QueryTraceMetrics: _experiments_rule("list"),
    SearchTraces: _experiments_rule("list"),
    SearchTracesV3: _experiments_rule("list"),
    GetMetricHistoryBulkInterval: _experiments_rule("list"),
    SearchRuns: _experiments_rule("list"),
    # Model registry
    # Registered models
    CreateRegisteredModel: _registered_models_rule("create"),
    GetRegisteredModel: _registered_models_rule("get"),
    DeleteRegisteredModel: _registered_models_rule("delete"),
    UpdateRegisteredModel: _registered_models_rule("update"),
    RenameRegisteredModel: _registered_models_rule("update"),
    SearchRegisteredModels: _registered_models_rule("list"),
    # Registered model child resources (writes)
    CreateModelVersion: _registered_models_rule("update"),
    DeleteModelVersion: _registered_models_rule("update"),
    UpdateModelVersion: _registered_models_rule("update"),
    TransitionModelVersionStage: _registered_models_rule("update"),
    SetRegisteredModelTag: _registered_models_rule("update"),
    DeleteRegisteredModelTag: _registered_models_rule("update"),
    SetModelVersionTag: _registered_models_rule("update"),
    DeleteModelVersionTag: _registered_models_rule("update"),
    SetRegisteredModelAlias: _registered_models_rule("update"),
    DeleteRegisteredModelAlias: _registered_models_rule("update"),
    CreateWebhook: _registered_models_rule("update"),
    DeleteWebhook: _registered_models_rule("update"),
    TestWebhook: _registered_models_rule("update"),
    UpdateWebhook: _registered_models_rule("update"),
    # Registered model child resources (reads)
    GetModelVersion: _registered_models_rule("get"),
    GetModelVersionDownloadUri: _registered_models_rule("get"),
    GetModelVersionByAlias: _registered_models_rule("get"),
    GetWebhook: _registered_models_rule("get"),
    # Registered model child resources (lists)
    GetLatestVersions: _registered_models_rule("list"),
    ListWebhooks: _registered_models_rule("list"),
    # Gateway
    CreateGatewaySecret: _gateway_secrets_rule("create"),
    GetGatewaySecretInfo: _gateway_secrets_rule("get"),
    UpdateGatewaySecret: _gateway_secrets_rule("update"),
    DeleteGatewaySecret: _gateway_secrets_rule("delete"),
    ListGatewaySecretInfos: _gateway_secrets_rule("list"),
    CreateGatewayEndpoint: _gateway_endpoints_rule("create"),
    GetGatewayEndpoint: _gateway_endpoints_rule("get"),
    UpdateGatewayEndpoint: _gateway_endpoints_rule("update"),
    DeleteGatewayEndpoint: _gateway_endpoints_rule("delete"),
    ListGatewayEndpoints: _gateway_endpoints_rule("list"),
    CreateGatewayModelDefinition: _gateway_model_definitions_rule("create"),
    GetGatewayModelDefinition: _gateway_model_definitions_rule("get"),
    UpdateGatewayModelDefinition: _gateway_model_definitions_rule("update"),
    DeleteGatewayModelDefinition: _gateway_model_definitions_rule("delete"),
    ListGatewayModelDefinitions: _gateway_model_definitions_rule("list"),
    AttachModelToGatewayEndpoint: _gateway_endpoints_rule("update"),
    DetachModelFromGatewayEndpoint: _gateway_endpoints_rule("update"),
    # Bindings and tags modify the endpoint, not create/delete it, so use update verb
    CreateGatewayEndpointBinding: _gateway_endpoints_rule("update"),
    DeleteGatewayEndpointBinding: _gateway_endpoints_rule("update"),
    ListGatewayEndpointBindings: _gateway_endpoints_rule("list"),
    SetGatewayEndpointTag: _gateway_endpoints_rule("update"),
    DeleteGatewayEndpointTag: _gateway_endpoints_rule("update"),
    # Prompt optimization jobs (experiment-scoped, matching upstream)
    CreatePromptOptimizationJob: _experiments_rule("update"),
    GetPromptOptimizationJob: _experiments_rule("get"),
    SearchPromptOptimizationJobs: _experiments_rule("list"),
    CancelPromptOptimizationJob: _experiments_rule("update"),
    DeletePromptOptimizationJob: _experiments_rule("update"),
    # Workspaces
    # ListWorkspaces omits a direct RBAC verb/namespace check because a single
    # SelfSubjectAccessReview cannot cover the full list. The response is instead filtered per
    # namespace via accessible_workspaces.
    ListWorkspaces: _workspaces_rule(None, apply_workspace_filter=True, requires_workspace=False),
    GetWorkspace: AuthorizationRule(None, requires_workspace=False, workspace_access_check=True),
    CreateWorkspace: _workspaces_rule("create", deny=True, requires_workspace=False),
    UpdateWorkspace: _workspaces_rule("update", deny=True, requires_workspace=False),
    DeleteWorkspace: _workspaces_rule("delete", deny=True, requires_workspace=False),
}


PATH_AUTHORIZATION_RULES: dict[
    tuple[str, str], AuthorizationRule | tuple[AuthorizationRule, ...]
] = {
    # Unprotected endpoints (no authorization required)
    ("/version", "GET"): AuthorizationRule(None),
    ("/server-info", "GET"): AuthorizationRule(None),
    ("/api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule("list"),
    ("/ajax-api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule("list"),
    ("/graphql", "GET"): _experiments_rule("get"),
    ("/graphql", "POST"): _experiments_rule("get"),
    ("/api/2.0/mlflow/gateway-proxy", "GET"): _gateway_endpoints_use_rule(),
    ("/api/2.0/mlflow/gateway-proxy", "POST"): _gateway_endpoints_use_rule(),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "GET"): _gateway_endpoints_use_rule(),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "POST"): _gateway_endpoints_use_rule(),
    # Gateway invocation routes (FastAPI)
    ("/gateway/<endpoint_name>/mlflow/invocations", "POST"): _gateway_endpoints_use_rule(),
    ("/gateway/mlflow/v1/chat/completions", "POST"): _gateway_endpoints_use_rule(),
    ("/gateway/openai/v1/chat/completions", "POST"): _gateway_endpoints_use_rule(),
    ("/gateway/openai/v1/embeddings", "POST"): _gateway_endpoints_use_rule(),
    ("/gateway/openai/v1/responses", "POST"): _gateway_endpoints_use_rule(),
    ("/gateway/anthropic/v1/messages", "POST"): _gateway_endpoints_use_rule(),
    (
        "/gateway/gemini/v1beta/models/<endpoint_name>:generateContent",
        "POST",
    ): _gateway_endpoints_use_rule(),
    (
        "/gateway/gemini/v1beta/models/<endpoint_name>:streamGenerateContent",
        "POST",
    ): _gateway_endpoints_use_rule(),
    ("/get-artifact", "GET"): _experiments_rule("get"),
    ("/model-versions/get-artifact", "GET"): _registered_models_rule("get"),
    ("/ajax-api/2.0/mlflow/upload-artifact", "POST"): _experiments_rule("update"),
    ("/ajax-api/2.0/mlflow/get-trace-artifact", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/get-trace-artifact", "GET"): _experiments_rule("get"),
    ("/ajax-api/2.0/mlflow/metrics/get-history-bulk", "GET"): _experiments_rule("list"),
    (
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
        "GET",
    ): _experiments_rule("list"),
    ("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "POST"): _experiments_rule("update"),
    (
        "/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files",
        "GET",
    ): _experiments_rule("get"),
    ("/api/2.0/mlflow/experiments/search-datasets", "POST"): _datasets_rule("list"),
    ("/ajax-api/2.0/mlflow/experiments/search-datasets", "POST"): _datasets_rule("list"),
    # Assessment deletion (path-parameterized, not matched by handler rules)
    ("/api/3.0/mlflow/traces/<trace_id>/assessments/<assessment_id>", "DELETE"): _experiments_rule(
        "update"
    ),
    ("/ajax-api/3.0/mlflow/traces/<trace_id>/assessments/<assessment_id>", "DELETE"): (
        _experiments_rule("update")
    ),
    # Trace retrieval endpoints (REST v3)
    ("/api/3.0/mlflow/traces/get", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/traces/get", "GET"): _experiments_rule("get"),
    # OTEL trace ingestion endpoint
    (OTLP_TRACES_PATH, "POST"): _experiments_rule("update"),
    # Job API endpoints (FastAPI router, experiment-scoped, matching upstream)
    ("/ajax-api/3.0/jobs", "POST"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/", "POST"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/<job_id>", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/jobs/<job_id>/", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/jobs/cancel/<job_id>", "PATCH"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/cancel/<job_id>/", "PATCH"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/search", "POST"): _experiments_rule("list"),
    ("/ajax-api/3.0/jobs/search/", "POST"): _experiments_rule("list"),
    # Assistant API endpoints (FastAPI router, currently localhost-only)
    ("/ajax-api/3.0/mlflow/assistant/message", "POST"): _assistants_rule("create"),
    (
        "/ajax-api/3.0/mlflow/assistant/sessions/<session_id>/stream",
        "GET",
    ): _assistants_rule("get"),
    ("/ajax-api/3.0/mlflow/assistant/status", "GET"): _assistants_rule("get"),
    ("/ajax-api/3.0/mlflow/assistant/sessions/<session_id>", "PATCH"): _assistants_rule("update"),
    (
        "/ajax-api/3.0/mlflow/assistant/providers/<provider>/health",
        "GET",
    ): _assistants_rule("get"),
    ("/ajax-api/3.0/mlflow/assistant/config", "GET"): _assistants_rule("get"),
    ("/ajax-api/3.0/mlflow/assistant/config", "PUT"): _assistants_rule("update"),
    ("/ajax-api/3.0/mlflow/assistant/skills/install", "POST"): _assistants_rule("update"),
    # Gateway discovery/config endpoints
    ("/ajax-api/3.0/mlflow/gateway/supported-providers", "GET"): _gateway_model_definitions_rule(
        "list"
    ),
    ("/ajax-api/3.0/mlflow/gateway/supported-models", "GET"): _gateway_model_definitions_rule(
        "list"
    ),
    ("/ajax-api/3.0/mlflow/gateway/provider-config", "GET"): _gateway_model_definitions_rule("get"),
    ("/ajax-api/3.0/mlflow/gateway/secrets/config", "GET"): _gateway_secrets_rule("get"),
    # Scorers (online config + invocation) - experiment-scoped
    ("/ajax-api/3.0/mlflow/scorers/online-configs", "GET"): _experiments_rule("get"),
    ("/api/3.0/mlflow/scorers/online-configs", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/scorers/online-config", "PUT"): _experiments_rule("update"),
    ("/api/3.0/mlflow/scorers/online-config", "PUT"): _experiments_rule("update"),
    ("/ajax-api/3.0/mlflow/scorer/invoke", "POST"): _experiments_rule("update"),
    # Demo data generation and deletion
    ("/ajax-api/3.0/mlflow/demo/generate", "POST"): (
        _experiments_rule("create"),
        _datasets_rule("create"),
        _registered_models_rule("create"),
    ),
    ("/ajax-api/3.0/mlflow/demo/delete", "POST"): (
        _experiments_rule("delete"),
        _datasets_rule("delete"),
        _registered_models_rule("delete"),
    ),
}


GRAPHQL_OPERATION_RULES: dict[str, AuthorizationRule] = _build_graphql_operation_rules(
    AuthorizationRule, _normalize_resource_name
)


def _normalize_rules(
    value: AuthorizationRule | tuple[AuthorizationRule, ...],
) -> list[AuthorizationRule]:
    """Normalize a single rule or tuple of rules into a list."""
    if isinstance(value, AuthorizationRule):
        return [value]
    return list(value)


__all__ = [
    "AuthorizationRule",
    "GRAPHQL_OPERATION_RULES",
    "PATH_AUTHORIZATION_RULES",
    "REQUEST_AUTHORIZATION_RULES",
    "_normalize_rules",
]
