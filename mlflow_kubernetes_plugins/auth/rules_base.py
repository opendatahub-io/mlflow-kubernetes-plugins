"""Baseline authorization tables for the minimum supported MLflow contract."""

from __future__ import annotations

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

from mlflow_kubernetes_plugins.auth.collection_filters import (
    COLLECTION_POLICY_BROAD_ONLY,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    COLLECTION_POLICY_REQUEST_RUN_IDS,
    COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS,
    COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES,
    COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
    COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS,
    COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS,
    COLLECTION_POLICY_RESPONSE_TRACES,
)
from mlflow_kubernetes_plugins.auth.constants import (
    WORKSPACE_MUTATION_DENIED_MESSAGE,
)
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,
    RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,
    RESOURCE_NAME_PARSER_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,
    RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME,
    RESOURCE_NAME_PARSER_OTEL_EXPERIMENT_ID_HEADER_TO_NAME,
    RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,
)
from mlflow_kubernetes_plugins.auth.rules import (
    AuthorizationRule,
    _assistants_rule,
    _datasets_rule,
    _experiments_rule,
    _gateway_endpoints_rule,
    _gateway_endpoints_use_rule,
    _gateway_model_definitions_rule,
    _gateway_secrets_rule,
    _registered_models_rule,
    _workspaces_rule,
)

BASE_REQUEST_AUTHORIZATION_RULES: dict[type, AuthorizationRule | tuple[AuthorizationRule, ...]] = {
    # Experiments
    CreateExperiment: _experiments_rule("create"),
    GetExperiment: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    GetExperimentByName: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_NAME,),
    ),
    DeleteExperiment: _experiments_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    RestoreExperiment: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    UpdateExperiment: _experiments_rule(
        "update",
        resource_name_parsers=(
            RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
            RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME,
        ),
    ),
    SetExperimentTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    DeleteExperimentTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    SearchExperiments: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
    ),
    # Datasets
    AddDatasetToExperiments: (
        _datasets_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
        ),
        _experiments_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,),
        ),
    ),
    CreateDataset: _datasets_rule("create"),
    DeleteDataset: _datasets_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    DeleteDatasetRecords: _datasets_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    DeleteDatasetTag: _datasets_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    RemoveDatasetFromExperiments: (
        _datasets_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
        ),
        _experiments_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,),
        ),
    ),
    SetDatasetTags: _datasets_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    UpsertDatasetRecords: _datasets_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    # Experiment child resources (write operations)
    CreateAssessment: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    UpdateAssessment: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    CreateRun: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
        override_run_user=True,
    ),
    DeleteRun: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    RestoreRun: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    UpdateRun: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    LogMetric: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    LogBatch: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    LogModel: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    SetTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    DeleteTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    LogParam: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    CreateLoggedModel: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    DeleteLoggedModel: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    FinalizeLoggedModel: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    DeleteLoggedModelTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    SetLoggedModelTags: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    LogLoggedModelParamsRequest: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    RegisterScorer: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    DeleteScorer: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    EndTrace: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    LinkPromptsToTrace: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    LinkTracesToRun: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    DeleteTraceTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    DeleteTraceTagV3: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    DeleteTraces: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    DeleteTracesV3: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    SetTraceTag: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    SetTraceTagV3: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    StartTrace: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    StartTraceV3: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,),
    ),
    LogInputs: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    LogOutputs: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    CompleteMultipartUpload: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    CreateMultipartUpload: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    AbortMultipartUpload: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    DeleteArtifact: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    UploadArtifact: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    # Experiment child resources (single-experiment reads)
    GetAssessmentRequest: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    GetRun: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    GetMetricHistory: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    ListArtifacts: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    GetLoggedModel: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    ListLoggedModelArtifacts: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    ListScorers: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    GetScorer: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    ListScorerVersions: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    GetTraceInfo: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    GetTraceInfoV3: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    DownloadArtifact: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    ListArtifactsMlflowArtifacts: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
    ),
    # Dataset reads
    GetDataset: _datasets_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    GetDatasetExperimentIds: _datasets_rule(
        "list",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    GetDatasetRecords: _datasets_rule(
        "list",
        resource_name_parsers=(RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    ),
    SearchDatasets: _datasets_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    SearchEvaluationDatasets: _datasets_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_DATASET_SUMMARIES,
    ),
    # Experiment child resources (multi-experiment reads)
    SearchLoggedModels: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    BatchGetTraces: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_TRACES,
    ),
    CalculateTraceFilterCorrelation: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    QueryTraceMetrics: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    SearchTraces: _experiments_rule("list", collection_policy=COLLECTION_POLICY_RESPONSE_TRACES),
    SearchTracesV3: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_TRACE_LOCATIONS,
    ),
    GetMetricHistoryBulkInterval: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_RUN_IDS,
    ),
    SearchRuns: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    # Model registry
    CreateRegisteredModel: _registered_models_rule("create"),
    GetRegisteredModel: _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteRegisteredModel: _registered_models_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    UpdateRegisteredModel: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    RenameRegisteredModel: _registered_models_rule(
        "update",
        resource_name_parsers=(
            RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
            RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME,
        ),
    ),
    SearchRegisteredModels: _registered_models_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_REGISTERED_MODELS,
    ),
    CreateModelVersion: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteModelVersion: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    UpdateModelVersion: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    TransitionModelVersionStage: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    SetRegisteredModelTag: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteRegisteredModelTag: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    SetModelVersionTag: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteModelVersionTag: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    SetRegisteredModelAlias: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteRegisteredModelAlias: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    CreateWebhook: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    DeleteWebhook: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,),
    ),
    TestWebhook: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,),
    ),
    UpdateWebhook: _registered_models_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,),
    ),
    GetModelVersion: _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    GetModelVersionDownloadUri: _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    GetModelVersionByAlias: _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    GetWebhook: _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,),
    ),
    GetLatestVersions: _registered_models_rule(
        "list",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    ListWebhooks: _registered_models_rule(
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    # Gateway
    CreateGatewaySecret: _gateway_secrets_rule("create"),
    GetGatewaySecretInfo: _gateway_secrets_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,),
    ),
    UpdateGatewaySecret: _gateway_secrets_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,),
    ),
    DeleteGatewaySecret: _gateway_secrets_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,),
    ),
    ListGatewaySecretInfos: _gateway_secrets_rule(
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    CreateGatewayEndpoint: _gateway_endpoints_rule("create"),
    GetGatewayEndpoint: _gateway_endpoints_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    UpdateGatewayEndpoint: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    DeleteGatewayEndpoint: _gateway_endpoints_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    ListGatewayEndpoints: _gateway_endpoints_rule(
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    CreateGatewayModelDefinition: _gateway_model_definitions_rule("create"),
    GetGatewayModelDefinition: _gateway_model_definitions_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,),
    ),
    UpdateGatewayModelDefinition: _gateway_model_definitions_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,),
    ),
    DeleteGatewayModelDefinition: _gateway_model_definitions_rule(
        "delete",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,),
    ),
    ListGatewayModelDefinitions: _gateway_model_definitions_rule(
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    AttachModelToGatewayEndpoint: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    DetachModelFromGatewayEndpoint: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    CreateGatewayEndpointBinding: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    DeleteGatewayEndpointBinding: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    ListGatewayEndpointBindings: _gateway_endpoints_rule(
        "list",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    SetGatewayEndpointTag: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    DeleteGatewayEndpointTag: _gateway_endpoints_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
    ),
    # Prompt optimization jobs (experiment-scoped, matching upstream)
    CreatePromptOptimizationJob: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    GetPromptOptimizationJob: _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME,),
    ),
    SearchPromptOptimizationJobs: _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    ),
    CancelPromptOptimizationJob: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME,),
    ),
    DeletePromptOptimizationJob: _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_JOB_ID_TO_EXPERIMENT_NAME,),
    ),
    # Workspaces
    ListWorkspaces: _workspaces_rule(None, apply_workspace_filter=True, requires_workspace=False),
    GetWorkspace: AuthorizationRule(None, requires_workspace=False, workspace_access_check=True),
    CreateWorkspace: _workspaces_rule(
        "create",
        deny=True,
        deny_message=WORKSPACE_MUTATION_DENIED_MESSAGE,
        requires_workspace=False,
    ),
    UpdateWorkspace: _workspaces_rule(
        "update",
        deny=True,
        deny_message=WORKSPACE_MUTATION_DENIED_MESSAGE,
        requires_workspace=False,
    ),
    DeleteWorkspace: _workspaces_rule(
        "delete",
        deny=True,
        deny_message=WORKSPACE_MUTATION_DENIED_MESSAGE,
        requires_workspace=False,
    ),
}

BASE_PATH_AUTHORIZATION_RULES: dict[
    tuple[str, str], AuthorizationRule | tuple[AuthorizationRule, ...]
] = {
    # Unprotected endpoints (no authorization required)
    ("/version", "GET"): AuthorizationRule(None),
    ("/server-info", "GET"): AuthorizationRule(None),
    ("/api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS,
    ),
    ("/ajax-api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule(
        "list",
        collection_policy=COLLECTION_POLICY_RESPONSE_MODEL_VERSIONS,
    ),
    ("/graphql", "GET"): _experiments_rule("get"),
    ("/graphql", "POST"): _experiments_rule("get"),
    ("/api/2.0/mlflow/gateway-proxy", "GET"): _gateway_endpoints_use_rule(),
    ("/api/2.0/mlflow/gateway-proxy", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "GET"): _gateway_endpoints_use_rule(),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    # Gateway invocation routes (FastAPI)
    ("/gateway/<endpoint_name>/mlflow/invocations", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/gateway/mlflow/v1/chat/completions", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/gateway/openai/v1/chat/completions", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/gateway/openai/v1/embeddings", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/gateway/openai/v1/responses", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/gateway/anthropic/v1/messages", "POST"): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    (
        "/gateway/gemini/v1beta/models/<endpoint_name>:generateContent",
        "POST",
    ): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    (
        "/gateway/gemini/v1beta/models/<endpoint_name>:streamGenerateContent",
        "POST",
    ): _gateway_endpoints_use_rule(
        resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    ),
    ("/get-artifact", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/model-versions/get-artifact", "GET"): _registered_models_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,),
    ),
    ("/ajax-api/2.0/mlflow/upload-artifact", "POST"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/ajax-api/2.0/mlflow/get-trace-artifact", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/ajax-api/3.0/mlflow/get-trace-artifact", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_REQUEST_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/ajax-api/2.0/mlflow/metrics/get-history-bulk", "GET"): _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_RUN_IDS,
    ),
    (
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
        "GET",
    ): _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_RUN_IDS,
    ),
    ("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "POST"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    (
        "/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files",
        "GET",
    ): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_MODEL_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/api/2.0/mlflow/experiments/search-datasets", "POST"): _datasets_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    ("/ajax-api/2.0/mlflow/experiments/search-datasets", "POST"): _datasets_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    ),
    # Assessment deletion (path-parameterized, not matched by handler rules)
    ("/api/3.0/mlflow/traces/<trace_id>/assessments/<assessment_id>", "DELETE"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/ajax-api/3.0/mlflow/traces/<trace_id>/assessments/<assessment_id>", "DELETE"): (
        _experiments_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
        )
    ),
    # Trace retrieval endpoints (REST v3)
    ("/api/3.0/mlflow/traces/get", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    ("/ajax-api/3.0/mlflow/traces/get", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_ID_TO_EXPERIMENT_NAME,),
    ),
    # OTEL trace ingestion endpoint
    (OTLP_TRACES_PATH, "POST"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_OTEL_EXPERIMENT_ID_HEADER_TO_NAME,),
    ),
    # Generic Job API endpoints (FastAPI router) stay workspace-scoped. The plugin does not try to
    # infer every resource a generic job may touch from the job metadata.
    ("/ajax-api/3.0/jobs", "POST"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/", "POST"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/<job_id>", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/jobs/<job_id>/", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/jobs/cancel/<job_id>", "PATCH"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/cancel/<job_id>/", "PATCH"): _experiments_rule("update"),
    ("/ajax-api/3.0/jobs/search", "POST"): _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    ),
    ("/ajax-api/3.0/jobs/search/", "POST"): _experiments_rule(
        "list",
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    ),
    ("/ajax-api/3.0/mlflow/jobs/<job_id>", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/jobs/<job_id>/", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/jobs/cancel/<job_id>", "PATCH"): _experiments_rule("update"),
    ("/ajax-api/3.0/mlflow/jobs/cancel/<job_id>/", "PATCH"): _experiments_rule("update"),
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
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    ("/ajax-api/3.0/mlflow/gateway/supported-models", "GET"): _gateway_model_definitions_rule(
        "list",
        collection_policy=COLLECTION_POLICY_BROAD_ONLY,
    ),
    ("/ajax-api/3.0/mlflow/gateway/provider-config", "GET"): _gateway_model_definitions_rule("get"),
    ("/ajax-api/3.0/mlflow/gateway/secrets/config", "GET"): _gateway_secrets_rule("get"),
    # Scorers (online config + invocation) - experiment-scoped
    ("/ajax-api/3.0/mlflow/scorers/online-configs", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    ("/api/3.0/mlflow/scorers/online-configs", "GET"): _experiments_rule(
        "get",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    ("/ajax-api/3.0/mlflow/scorers/online-config", "PUT"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    ("/api/3.0/mlflow/scorers/online-config", "PUT"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
    ("/ajax-api/3.0/mlflow/scorer/invoke", "POST"): _experiments_rule(
        "update",
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    ),
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
