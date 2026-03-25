"""Kubernetes-backed MLflow workspace and authorization plugins."""

from importlib.metadata import PackageNotFoundError, version

from mlflow_kubernetes_plugins.auth.middleware import create_app
from mlflow_kubernetes_plugins.workspace_plugin.provider import (
    KubernetesWorkspaceProvider,
    create_kubernetes_workspace_store,
)

try:
    __version__ = version("mlflow-kubernetes-plugins")
except PackageNotFoundError:  # pragma: no cover - local source tree without installed metadata
    __version__ = "0.0.0"

__all__ = [
    "KubernetesWorkspaceProvider",
    "__version__",
    "create_app",
    "create_kubernetes_workspace_store",
]
