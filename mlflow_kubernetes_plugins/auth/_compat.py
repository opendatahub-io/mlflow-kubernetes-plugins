"""Compatibility helpers for MLflow-version-dependent auth surfaces."""

from __future__ import annotations

import importlib

import mlflow
from mlflow.protos import mlflow_artifacts_pb2 as artifacts_pb2
from mlflow.protos import service_pb2 as service_pb2_mod
from packaging.version import Version

MLFLOW_VERSION = Version(mlflow.__version__)
HAS_MLFLOW_3_11_AUTH_SURFACE = MLFLOW_VERSION >= Version("3.11.0.dev0")

if HAS_MLFLOW_3_11_AUTH_SURFACE:
    GetPresignedDownloadUrl = artifacts_pb2.GetPresignedDownloadUrl
    # `issues_pb2` is absent on MLflow 3.10, so this module import must stay conditional even
    # though the service protobuf module itself exists across both supported minor lines.
    issues_pb2 = importlib.import_module("mlflow.protos.issues_pb2")
    CreateIssue = issues_pb2.CreateIssue
    GetIssue = issues_pb2.GetIssue
    SearchIssues = issues_pb2.SearchIssues
    UpdateIssue = issues_pb2.UpdateIssue
    BatchGetTraceInfos = service_pb2_mod.BatchGetTraceInfos
    CreateGatewayBudgetPolicy = service_pb2_mod.CreateGatewayBudgetPolicy
    DeleteGatewayBudgetPolicy = service_pb2_mod.DeleteGatewayBudgetPolicy
    GetGatewayBudgetPolicy = service_pb2_mod.GetGatewayBudgetPolicy
    ListGatewayBudgetPolicies = service_pb2_mod.ListGatewayBudgetPolicies
    ListGatewayBudgetWindows = service_pb2_mod.ListGatewayBudgetWindows
    UpdateGatewayBudgetPolicy = service_pb2_mod.UpdateGatewayBudgetPolicy
else:  # pragma: no cover - exercised via MLflow version matrix
    GetPresignedDownloadUrl = None
    CreateIssue = GetIssue = SearchIssues = UpdateIssue = None
    BatchGetTraceInfos = None
    CreateGatewayBudgetPolicy = None
    DeleteGatewayBudgetPolicy = None
    GetGatewayBudgetPolicy = None
    ListGatewayBudgetPolicies = None
    ListGatewayBudgetWindows = None
    UpdateGatewayBudgetPolicy = None

__all__ = [
    "BatchGetTraceInfos",
    "CreateGatewayBudgetPolicy",
    "CreateIssue",
    "DeleteGatewayBudgetPolicy",
    "GetGatewayBudgetPolicy",
    "GetIssue",
    "GetPresignedDownloadUrl",
    "HAS_MLFLOW_3_11_AUTH_SURFACE",
    "ListGatewayBudgetPolicies",
    "ListGatewayBudgetWindows",
    "MLFLOW_VERSION",
    "SearchIssues",
    "UpdateGatewayBudgetPolicy",
    "UpdateIssue",
]
