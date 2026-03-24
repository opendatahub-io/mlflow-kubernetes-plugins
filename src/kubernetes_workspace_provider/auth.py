"""Kubernetes-backed authorization plugin for the MLflow tracking server."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from typing import Iterable, NamedTuple

import werkzeug
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from flask import Flask, Response, g, has_request_context, request
from kubernetes import client, config
from kubernetes.client import AuthorizationV1Api
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mlflow.environment_variables import _MLFLOW_SGI_NAME
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.mlflow_artifacts_pb2 import (
    AbortMultipartUpload,
    CompleteMultipartUpload,
    CreateMultipartUpload,
    DeleteArtifact,
    DownloadArtifact,
    UploadArtifact,
)
from mlflow.protos.mlflow_artifacts_pb2 import (
    ListArtifacts as ListArtifactsMlflowArtifacts,
)
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
from mlflow.server import app as mlflow_app
from mlflow.server import handlers as mlflow_handlers
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, get_endpoints
from mlflow.server.workspace_helpers import WORKSPACE_HEADER_NAME, resolve_workspace_from_header
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils import workspace_context

if not hasattr(werkzeug, "__version__"):  # pragma: no cover - compatibility shim
    werkzeug.__version__ = "werkzeug"

DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_USERNAME_CLAIM = "sub"
DEFAULT_AUTH_GROUP = "mlflow.kubeflow.org"
DEFAULT_REMOTE_USER_HEADER = "x-remote-user"
DEFAULT_REMOTE_GROUPS_HEADER = "x-remote-groups"
DEFAULT_REMOTE_GROUPS_SEPARATOR = "|"

CACHE_TTL_ENV = "MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS"
USERNAME_CLAIM_ENV = "MLFLOW_K8S_AUTH_USERNAME_CLAIM"
AUTHORIZATION_MODE_ENV = "MLFLOW_K8S_AUTH_AUTHORIZATION_MODE"
REMOTE_USER_HEADER_ENV = "MLFLOW_K8S_AUTH_REMOTE_USER_HEADER"
REMOTE_GROUPS_HEADER_ENV = "MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER"
REMOTE_GROUPS_SEPARATOR_ENV = "MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR"

_UNPROTECTED_PATH_PREFIXES = ("/static", "/favicon.ico", "/health", "/build")
_UNPROTECTED_PATHS = {
    "/",
    "/health",
    "/version",
    "/metrics",
    "/api/3.0/mlflow/server-info",
    "/ajax-api/3.0/mlflow/server-info",
    "/ajax-api/3.0/mlflow/ui-telemetry",
}
RESOURCE_ASSISTANTS = "assistants"
RESOURCE_DATASETS = "datasets"
RESOURCE_EXPERIMENTS = "experiments"
RESOURCE_REGISTERED_MODELS = "registeredmodels"
RESOURCE_GATEWAY_SECRETS = "gatewaysecrets"
RESOURCE_GATEWAY_ENDPOINTS = "gatewayendpoints"
RESOURCE_GATEWAY_MODEL_DEFINITIONS = "gatewaymodeldefinitions"
_ALLOWED_RESOURCES = {
    RESOURCE_ASSISTANTS,
    RESOURCE_DATASETS,
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
    RESOURCE_GATEWAY_SECRETS,
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
}
_WORKSPACE_PERMISSION_RESOURCE_PRIORITY = (
    RESOURCE_EXPERIMENTS,
    RESOURCE_DATASETS,
    RESOURCE_REGISTERED_MODELS,
    RESOURCE_GATEWAY_SECRETS,
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
)

_WORKSPACE_REQUIRED_ERROR_MESSAGE = "Workspace context is required for this request."
_WORKSPACE_MUTATION_DENIED_MESSAGE = (
    "Workspace create, update, and delete operations are not supported when MLflow "
    "workspaces map to Kubernetes namespaces."
)

# Re-export GraphQL constants from auth_graphql module
from kubernetes_workspace_provider.auth_graphql import (
    GRAPHQL_FIELD_RESOURCE_MAP,
    GRAPHQL_FIELD_VERB_MAP,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    _build_graphql_operation_rules,
)
from kubernetes_workspace_provider.auth_graphql import (
    determine_graphql_rules as _determine_graphql_rules,
)
from kubernetes_workspace_provider.auth_graphql import (
    extract_graphql_query_info as _extract_graphql_query_info,
)
from kubernetes_workspace_provider.auth_graphql import (
    validate_graphql_field_authorization as _validate_graphql_field_authorization,
)

_logger = logging.getLogger(__name__)
_AUTHORIZATION_HANDLED: ContextVar[_AuthorizationResult | None] = ContextVar(
    "_AUTHORIZATION_HANDLED", default=None
)


class AuthorizationMode(str, Enum):
    SELF_SUBJECT_ACCESS_REVIEW = "self_subject_access_review"
    SUBJECT_ACCESS_REVIEW = "subject_access_review"


class _ReadWriteLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        with self._condition:
            self._writer = False
            self._condition.notify_all()


class _CacheEntry(NamedTuple):
    allowed: bool
    expires_at: float


_AuthorizationCacheKey = tuple[str, str, str, str | None, str]


class _AuthorizationCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[_AuthorizationCacheKey, _CacheEntry] = {}
        self._lock = _ReadWriteLock()

    def get(self, key: _AuthorizationCacheKey) -> bool | None:
        observed_expiration: float | None = None

        self._lock.acquire_read()
        try:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at > time.time():
                return entry.allowed
            observed_expiration = entry.expires_at
        finally:
            self._lock.release_read()

        self._lock.acquire_write()
        try:
            current = self._entries.get(key)
            if current is not None:
                if observed_expiration is None or current.expires_at <= observed_expiration:
                    self._entries.pop(key, None)
        finally:
            self._lock.release_write()
        return None

    def set(self, key: _AuthorizationCacheKey, allowed: bool) -> None:
        self._lock.acquire_write()
        try:
            self._entries[key] = _CacheEntry(
                allowed=allowed, expires_at=time.time() + self._ttl_seconds
            )
        finally:
            self._lock.release_write()


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
    return AuthorizationRule(verb, resource=None, **kwargs)


def _normalize_resource_name(resource: str | None) -> str | None:
    if resource is None:
        return None
    normalized = resource.replace("_", "")
    if normalized not in _ALLOWED_RESOURCES:
        raise ValueError(f"Unsupported RBAC resource '{resource}'")
    return normalized


_AUTH_RULES: dict[tuple[str, str], list[AuthorizationRule]] = {}
_AUTH_REGEX_RULES: list[tuple[re.Pattern[str], str, list[AuthorizationRule]]] = []
_HANDLER_RULES: dict[object, list[AuthorizationRule]] = {}
_RULES_COMPILED = False


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


def _unwrap_handler(handler):
    while hasattr(handler, "__wrapped__"):
        handler = handler.__wrapped__
    return handler


def _load_kubernetes_configuration() -> client.Configuration:
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    try:
        return client.Configuration.get_default_copy()
    except AttributeError:  # pragma: no cover - fallback for older client versions
        return client.Configuration()


def _create_api_client_for_subject_access_reviews() -> client.ApiClient:
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException as exc:  # pragma: no cover - depends on env
            raise MlflowException(
                "Failed to load Kubernetes configuration for authorization plugin",
                error_code=databricks_pb2.INVALID_STATE,
            ) from exc
    return client.ApiClient()


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
        token = _resolve_bearer_token(authorization_header, forwarded_access_token)
        identity = _RequestIdentity(token=token)
        username = _parse_jwt_subject(token, config_values.username_claim)
    else:
        remote_user = (remote_user_header_value or "").strip()
        if not remote_user:
            raise MlflowException(
                f"Missing required '{config_values.user_header}' header for "
                "SubjectAccessReview mode.",
                error_code=databricks_pb2.UNAUTHENTICATED,
            )
        groups = _parse_remote_groups(remote_groups_header_value, config_values.groups_separator)
        identity = _RequestIdentity(user=remote_user, groups=groups)
        username = remote_user

    workspace_name = None
    if isinstance(workspace, str):
        workspace_name = workspace.strip() or None

    rules = _find_authorization_rules(path, method, graphql_payload=graphql_payload)
    if rules is None or len(rules) == 0:
        _logger.warning(
            "No Kubernetes authorization rule matched request %s %s; returning 404.",
            method,
            path,
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
                    f"Authorization rule for '{method} {path}' is missing an RBAC resource "
                    "mapping.",
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
                f"Authorization rule for '{method} {path}' is missing a verb or other "
                "required configuration.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )

    return _AuthorizationResult(identity=identity, rules=rules, username=username)


class KubernetesAuthorizer:
    def __init__(
        self, config_values: KubernetesAuthConfig, group: str = DEFAULT_AUTH_GROUP
    ) -> None:
        self._group = group
        self._cache = _AuthorizationCache(config_values.cache_ttl_seconds)
        self._mode = config_values.authorization_mode
        self._base_configuration: client.Configuration | None = None
        self._sar_api_client: client.ApiClient | None = None
        self._user_header_label = (
            f"Header '{config_values.user_header}'"
            if config_values.user_header
            else "Remote user header"
        )

        if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
            self._base_configuration = _load_kubernetes_configuration()
        else:
            self._sar_api_client = _create_api_client_for_subject_access_reviews()

    def _build_api_client_with_token(self, token: str) -> client.ApiClient:
        if self._base_configuration is None:
            raise MlflowException(
                "SelfSubjectAccessReview mode is not initialized with base configuration.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        base = self._base_configuration
        configuration = client.Configuration()
        # Make a copy of the Kubernetes client without credential information
        configuration.host = base.host
        configuration.ssl_ca_cert = base.ssl_ca_cert
        configuration.verify_ssl = base.verify_ssl
        configuration.proxy = base.proxy
        configuration.no_proxy = base.no_proxy
        configuration.proxy_headers = base.proxy_headers
        configuration.safe_chars_for_path_param = base.safe_chars_for_path_param
        configuration.connection_pool_maxsize = base.connection_pool_maxsize
        configuration.assert_hostname = getattr(base, "assert_hostname", None)
        configuration.retries = getattr(base, "retries", None)
        configuration.cert_file = None
        configuration.key_file = None
        configuration.username = None
        configuration.password = None
        configuration.refresh_api_key_hook = None
        configuration.api_key = {"authorization": token}
        configuration.api_key_prefix = {"authorization": "Bearer"}
        return client.ApiClient(configuration)

    def _submit_self_subject_access_review(
        self,
        token: str,
        resource: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        body = client.V1SelfSubjectAccessReview(
            spec=client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                    subresource=subresource,
                )
            )
        )

        api_client = self._build_api_client_with_token(token)
        try:
            authorization_api = AuthorizationV1Api(api_client)
            response = authorization_api.create_self_subject_access_review(body)  # type: ignore[call-arg]
        finally:
            api_client.close()

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SelfSubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def _submit_subject_access_review(
        self,
        user: str,
        groups: tuple[str, ...],
        resource: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        body = client.V1SubjectAccessReview(
            spec=client.V1SubjectAccessReviewSpec(
                user=user,
                groups=list(groups) if groups else None,
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                    subresource=subresource,
                ),
            )
        )

        if self._sar_api_client is None:
            raise MlflowException(
                "SubjectAccessReview mode requires a Kubernetes client.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )

        authorization_api = AuthorizationV1Api(self._sar_api_client)
        response = authorization_api.create_subject_access_review(body)  # type: ignore[call-arg]

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def is_allowed(
        self,
        identity: _RequestIdentity,
        resource_type: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        resource = resource_type.replace("_", "")
        identity_hash = identity.subject_hash(
            self._mode, missing_user_label=self._user_header_label
        )
        cache_key = (identity_hash, namespace, resource, subresource, verb)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                allowed = self._submit_self_subject_access_review(
                    identity.token or "", resource, verb, namespace, subresource
                )
            else:
                allowed = self._submit_subject_access_review(
                    identity.user or "",
                    identity.groups,
                    resource,
                    verb,
                    namespace,
                    subresource,
                )
        except ApiException as exc:  # pragma: no cover - depends on live cluster
            if exc.status == 401:
                raise MlflowException(
                    "Authentication with the Kubernetes API failed. The provided token may be "
                    "invalid or expired.",
                    error_code=databricks_pb2.UNAUTHENTICATED,
                ) from exc
            if exc.status == 403:
                if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SelfSubjectAccessReview. Grant the required authorization."
                    )
                else:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SubjectAccessReview. Grant 'create' on subjectaccessreviews."
                    )
                raise MlflowException(
                    message,
                    error_code=databricks_pb2.PERMISSION_DENIED,
                ) from exc
            raise MlflowException(
                f"Failed to perform Kubernetes authorization check: {exc}",
                error_code=databricks_pb2.INTERNAL_ERROR,
            ) from exc

        self._cache.set(cache_key, allowed)
        _logger.debug(
            "Access review evaluated subject_hash=%s resource=%s subresource=%s namespace=%s "
            + "verb=%s allowed=%s",
            identity_hash,
            resource,
            subresource,
            namespace,
            verb,
            allowed,
        )
        return allowed

    def accessible_workspaces(self, identity: _RequestIdentity, names: Iterable[str]) -> set[str]:
        accessible: set[str] = set()
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for workspace_name in names:
            if self.can_access_workspace(identity, workspace_name, verb="list"):
                accessible.add(workspace_name)
            else:
                _logger.debug(
                    "Workspace %s excluded for subject_hash=%s; no list permission detected",
                    workspace_name,
                    subject_hash,
                )
        return accessible

    def can_access_workspace(
        self, identity: _RequestIdentity, workspace_name: str, verb: str = "get"
    ) -> bool:
        """Check if the identity can access the workspace via any priority resource.

        Iterates through experiments, registeredmodels, and gateway resources to find if
        the identity has the specified permission on any of them for the given workspace
        (namespace).

        Args:
            identity: The request identity containing token or user/groups.
            workspace_name: The workspace (namespace) to check access for.
            verb: The Kubernetes RBAC verb to check (default: "get").

        Returns:
            True if the identity has the permission on any priority resource.
        """
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for resource in _WORKSPACE_PERMISSION_RESOURCE_PRIORITY:
            if self.is_allowed(identity, resource, verb, workspace_name):
                _logger.debug(
                    "Workspace %s accessible for subject_hash=%s via resource=%s verb=%s",
                    workspace_name,
                    subject_hash,
                    resource,
                    verb,
                )
                return True
        return False


@dataclass(frozen=True)
class KubernetesAuthConfig:
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    username_claim: str = DEFAULT_USERNAME_CLAIM
    authorization_mode: AuthorizationMode = AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW
    user_header: str = DEFAULT_REMOTE_USER_HEADER
    groups_header: str = DEFAULT_REMOTE_GROUPS_HEADER
    groups_separator: str = DEFAULT_REMOTE_GROUPS_SEPARATOR

    @classmethod
    def from_env(cls) -> "KubernetesAuthConfig":
        ttl_env = os.environ.get(CACHE_TTL_ENV)
        username_claim = os.environ.get(USERNAME_CLAIM_ENV, DEFAULT_USERNAME_CLAIM)
        mode_env = os.environ.get(
            AUTHORIZATION_MODE_ENV, AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW.value
        )
        user_header = os.environ.get(REMOTE_USER_HEADER_ENV, DEFAULT_REMOTE_USER_HEADER)
        groups_header = os.environ.get(REMOTE_GROUPS_HEADER_ENV, DEFAULT_REMOTE_GROUPS_HEADER)
        groups_separator = os.environ.get(
            REMOTE_GROUPS_SEPARATOR_ENV, DEFAULT_REMOTE_GROUPS_SEPARATOR
        )

        cache_ttl_seconds = DEFAULT_CACHE_TTL_SECONDS
        if ttl_env:
            try:
                cache_ttl_seconds = float(ttl_env)
                if cache_ttl_seconds <= 0:
                    raise ValueError
            except ValueError as exc:
                raise MlflowException(
                    f"Environment variable {CACHE_TTL_ENV} must be a positive number if set",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                ) from exc

        try:
            authorization_mode = AuthorizationMode(mode_env.strip().lower())
        except ValueError as exc:
            valid_modes = ", ".join(mode.value for mode in AuthorizationMode)
            raise MlflowException(
                f"Environment variable {AUTHORIZATION_MODE_ENV} must be one of: {valid_modes}",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            ) from exc

        user_header = user_header.strip()
        groups_header = groups_header.strip()
        if not user_header:
            raise MlflowException(
                f"Environment variable {REMOTE_USER_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        if not groups_header:
            raise MlflowException(
                f"Environment variable {REMOTE_GROUPS_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        return cls(
            cache_ttl_seconds=cache_ttl_seconds,
            username_claim=username_claim,
            authorization_mode=authorization_mode,
            user_header=user_header,
            groups_header=groups_header,
            groups_separator=groups_separator or DEFAULT_REMOTE_GROUPS_SEPARATOR,
        )


def _normalize_rules(
    value: AuthorizationRule | tuple[AuthorizationRule, ...],
) -> list[AuthorizationRule]:
    """Normalize a single rule or tuple of rules into a list."""
    if isinstance(value, AuthorizationRule):
        return [value]
    return list(value)


def _compile_authorization_rules() -> None:
    global _RULES_COMPILED
    if _RULES_COMPILED:
        return

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
            _HANDLER_RULES[_unwrap_handler(handler)] = _normalize_rules(value)
        return handler

    # Inspect the protobuf-driven Flask routes and copy over authorization metadata.
    for path, handler, methods in get_endpoints(_get_request_authorization_handler):
        if not path:
            continue

        canonical_path = _canonicalize_path(raw_path=path)
        if _is_unprotected_path(canonical_path):
            continue

        base_handler = _unwrap_handler(handler)
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
                regex_rules.append((_re_compile_path(canonical_path), method, rules))
            else:
                exact_rules[(canonical_path, method)] = rules

    # Include custom Flask routes (e.g., get-artifact) that aren't part of the protobuf services.
    for rule in mlflow_app.url_map.iter_rules():
        view_func = mlflow_app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue

        canonical_path = _canonicalize_path(raw_path=rule.rule)
        if _is_unprotected_path(canonical_path):
            continue

        base_handler = _unwrap_handler(view_func)
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
            regex_rules.append((_re_compile_path(path), method, normalized))
        else:
            exact_rules[(path, method)] = normalized

    if uncovered:
        formatted = ", ".join(f"{method} {path}" for path, method in uncovered)
        raise MlflowException(
            "Kubernetes auth plugin cannot determine authorization mapping for endpoints: "
            f"{formatted}. Update the plugin allow list or verb mapping.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )

    # Persist the computed lookup tables so _find_authorization_rules can use them.
    _AUTH_RULES.update(exact_rules)
    _AUTH_REGEX_RULES.extend(regex_rules)
    _RULES_COMPILED = True


def _validate_fastapi_route_authorization(fastapi_app: FastAPI) -> None:
    """Ensure all protected FastAPI routes are covered by authorization rules."""
    missing: list[tuple[str, str]] = []

    for route in getattr(fastapi_app, "routes", []):
        if not isinstance(route, APIRoute):
            continue
        methods = getattr(route, "methods", set()) or set()
        canonical_path = _canonicalize_path(raw_path=route.path or "")
        if not canonical_path or _is_unprotected_path(canonical_path):
            continue
        template_path = _fastapi_path_to_template(canonical_path)
        # Use a concrete probe path so _find_authorization_rules follows the same regex path
        # matching logic that real requests do.
        probe_path = _templated_path_to_probe(template_path)

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
    """Find authorization rules for a request.

    For most endpoints, returns a single-element list. For GraphQL endpoints,
    may return multiple rules (one per resource type accessed by the query).

    Returns None if the path is not covered or if authorization cannot be
    determined (e.g., unknown GraphQL fields).
    """
    canonical_path = _canonicalize_path(raw_path=request_path or "")

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
                _logger.error("Could not determine GraphQL authorization: no query provided.")
                return None

            query_info = _extract_graphql_query_info(query_string)
            if not query_info.root_fields and not query_info.has_nested_model_registry_access:
                _logger.error(
                    "Could not determine GraphQL authorization: query could not be "
                    "parsed or contained no recognized fields."
                )
                return None

            # _determine_graphql_rules returns None if unknown fields are present
            return _determine_graphql_rules(query_info, AuthorizationRule)
        return rules

    for pattern, pattern_method, candidate in _AUTH_REGEX_RULES:
        if pattern_method == method and pattern.fullmatch(canonical_path):
            return candidate

    return None


# MLflow has some APIs that are through Flask and some through FastAPI. When MLflow is running
# under uvicorn, the FastAPI app wraps the entire Flask app, so we rely on a context variable to
# ensure the Flask middleware can skip duplicate authorization checks.
class KubernetesAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Kubernetes-based authorization."""

    def __init__(self, app, authorizer: KubernetesAuthorizer, config_values: KubernetesAuthConfig):
        super().__init__(app)
        self.authorizer = authorizer
        self.config_values = config_values

    async def dispatch(self, request: Request, call_next):
        """Process each request through the authorization pipeline."""
        authorization_token = _AUTHORIZATION_HANDLED.set(None)
        try:
            canonical_path = _canonicalize_path(
                raw_path=str(request.url.path or ""),
                scope_path=request.scope.get("path"),
                root_path=request.scope.get("root_path"),
            )
            fastapi_app = request.scope.get("app")
            if fastapi_app is None:
                exc = MlflowException(
                    "FastAPI app missing from request scope.",
                    error_code=databricks_pb2.INTERNAL_ERROR,
                )
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )

            # Skip authentication for unprotected paths
            if _is_unprotected_path(canonical_path):
                return await call_next(request)

            workspace_name = workspace_context.get_request_workspace()
            workspace_set = False

            if workspace_name is None:
                # FastAPI executes middlewares in reverse order, so this auth middleware can run
                # before the MLflow workspace middleware. Resolve here using the same helper, which
                # also falls back to the configured default workspace when the header is missing
                # or empty.
                try:
                    workspace = resolve_workspace_from_header(
                        request.headers.get(WORKSPACE_HEADER_NAME)
                    )
                except MlflowException as exc:
                    return JSONResponse(
                        status_code=exc.get_http_status_code(),
                        content=json.loads(exc.serialize_as_json()),
                    )

                if workspace is not None:
                    workspace_name = workspace.name
                    workspace_context.set_server_request_workspace(workspace_name)
                    workspace_set = True

            if canonical_path.endswith("/graphql"):
                # Let Flask authorize GraphQL to avoid consuming/rebuffering the ASGI body.
                try:
                    return await call_next(request)
                finally:
                    if workspace_set:
                        workspace_context.clear_server_request_workspace()

            try:
                auth_result = _authorize_request(
                    authorization_header=request.headers.get("Authorization"),
                    forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
                    remote_user_header_value=request.headers.get(self.config_values.user_header),
                    remote_groups_header_value=request.headers.get(
                        self.config_values.groups_header
                    ),
                    path=canonical_path,
                    method=request.method,
                    authorizer=self.authorizer,
                    config_values=self.config_values,
                    workspace=workspace_name,
                )
                _AUTHORIZATION_HANDLED.set(auth_result)
            except MlflowException as exc:
                if workspace_set:
                    workspace_context.clear_server_request_workspace()
                if (
                    workspace_name is None
                    and exc.error_code
                    == databricks_pb2.ErrorCode.Name(databricks_pb2.INVALID_PARAMETER_VALUE)
                    and exc.message == _WORKSPACE_REQUIRED_ERROR_MESSAGE
                ):
                    exc = MlflowException(
                        _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                        error_code=databricks_pb2.INTERNAL_ERROR,
                    )
                return JSONResponse(
                    status_code=exc.get_http_status_code(),
                    content={"error": {"code": exc.error_code, "message": exc.message}},
                )

            # Continue with the request, clearing any temporary workspace context.
            try:
                response = await call_next(request)
            finally:
                if workspace_set:
                    workspace_context.clear_server_request_workspace()
            return response
        finally:
            _AUTHORIZATION_HANDLED.reset(authorization_token)


def _override_run_user(username: str) -> None:
    """Rewrite the request payload so MLflow sees the authenticated user as run owner."""
    if not request.mimetype or "json" not in request.mimetype.lower():
        return

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return

    payload["user_id"] = username
    data = json.dumps(payload).encode("utf-8")

    # Reset cached JSON with proper structure expected by Werkzeug
    # The keys are boolean values for the 'silent' parameter
    request._cached_json = {True: payload, False: payload}  # type: ignore[attr-defined]
    request._cached_data = data  # type: ignore[attr-defined]
    request.environ["wsgi.input"] = io.BytesIO(data)
    request.environ["CONTENT_LENGTH"] = str(len(data))
    request.environ["CONTENT_TYPE"] = "application/json"


def _record_authorization_metadata(auth_result: _AuthorizationResult) -> None:
    """Record auth metadata on the Flask request context."""
    primary_rule = auth_result.rules[0]
    if auth_result.username and primary_rule.override_run_user:
        _override_run_user(auth_result.username)

    g.mlflow_k8s_identity = auth_result.identity
    g.mlflow_k8s_apply_workspace_filter = primary_rule.apply_workspace_filter


def create_app(app: Flask = mlflow_app) -> Flask:
    """Enable Kubernetes-based authorization for the MLflow tracking server."""

    global _logger
    parent_logger = getattr(app, "logger", logging.getLogger("mlflow"))
    _logger = parent_logger
    _logger.info("Kubernetes authorization plugin initialized")

    config_values = KubernetesAuthConfig.from_env()
    authorizer = KubernetesAuthorizer(config_values=config_values)

    _compile_authorization_rules()

    @app.before_request
    def _k8s_auth_before_request():
        auth_result = _AUTHORIZATION_HANDLED.get()
        if auth_result is not None:
            _record_authorization_metadata(auth_result)
            return None

        canonical_path = _canonicalize_path(
            raw_path=request.path or "",
            path_info=request.environ.get("PATH_INFO"),
            script_name=request.environ.get("SCRIPT_NAME"),
        )
        if _is_unprotected_path(canonical_path):
            return None

        graphql_payload: dict[str, object] | None = None
        if canonical_path.endswith("/graphql"):
            try:
                payload = request.get_json(silent=True) or {}
                if isinstance(payload, dict):
                    graphql_payload = payload
            except Exception:
                graphql_payload = None

        try:
            auth_result = _authorize_request(
                authorization_header=request.headers.get("Authorization"),
                forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
                remote_user_header_value=request.headers.get(config_values.user_header),
                remote_groups_header_value=request.headers.get(config_values.groups_header),
                path=canonical_path,
                method=request.method,
                authorizer=authorizer,
                config_values=config_values,
                workspace=workspace_context.get_request_workspace(),
                graphql_payload=graphql_payload,
            )
        except MlflowException as exc:
            response = Response(mimetype="application/json")
            response.set_data(exc.serialize_as_json())
            response.status_code = exc.get_http_status_code()
            return response

        # Use the first rule for metadata - these properties are consistent across all rules
        _record_authorization_metadata(auth_result)

        return None

    @app.after_request
    def _k8s_auth_after_request(response: Response):
        try:
            should_filter = getattr(g, "mlflow_k8s_apply_workspace_filter", False)
            identity = getattr(g, "mlflow_k8s_identity", None)
            can_filter_response = (
                response.mimetype == "application/json" and response.status_code < 400
            )
            if not (should_filter and identity and can_filter_response):
                return response

            try:
                payload = json.loads(response.get_data(as_text=True))
            except Exception:
                payload = None

            if not isinstance(payload, dict):
                return response

            workspaces = payload.get("workspaces")
            if not isinstance(workspaces, list):
                return response

            workspace_names = [ws.get("name") for ws in workspaces if isinstance(ws, dict)]
            accessible = authorizer.accessible_workspaces(
                identity, [name for name in workspace_names if isinstance(name, str)]
            )
            payload["workspaces"] = [
                ws for ws in workspaces if isinstance(ws, dict) and ws.get("name") in accessible
            ]
            response.set_data(json.dumps(payload))
            response.headers["Content-Length"] = str(len(response.get_data()))
        finally:
            for attr in (
                "mlflow_k8s_identity",
                "mlflow_k8s_apply_workspace_filter",
            ):
                if hasattr(g, attr):
                    delattr(g, attr)
            _AUTHORIZATION_HANDLED.set(None)

        return response

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        fastapi_app = create_fastapi_app(app)

        # Add Kubernetes auth middleware to FastAPI
        # Important: This must be added AFTER security middleware but BEFORE routes
        # to ensure proper middleware ordering
        #
        # Note: The KubernetesAuthMiddleware runs for all requests when the Flask app
        # is mounted under FastAPI. It sets a context variable so the Flask auth hooks
        # can skip duplicate authorization work when FastAPI already handled it.
        fastapi_app.add_middleware(
            KubernetesAuthMiddleware,
            authorizer=authorizer,
            config_values=config_values,
        )
        _validate_fastapi_route_authorization(fastapi_app)
        _validate_graphql_field_authorization()
        return fastapi_app
    return app


__all__ = [
    "create_app",
    "GRAPHQL_FIELD_RESOURCE_MAP",
    "GRAPHQL_FIELD_VERB_MAP",
    "K8S_GRAPHQL_OPERATION_RESOURCE_MAP",
    "K8S_GRAPHQL_OPERATION_VERB_MAP",
]
