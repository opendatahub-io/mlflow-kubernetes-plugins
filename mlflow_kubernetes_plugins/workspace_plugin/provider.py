from __future__ import annotations

import base64
import binascii
import logging
import os
import posixpath
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from kubernetes import client, config
from kubernetes.client import CoreV1Api, CustomObjectsApi
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.workspace.abstract_store import AbstractStore
from mlflow.utils.uri import append_to_uri_path

from mlflow_kubernetes_plugins.workspace_plugin.caches import (
    MlflowConfigCache,
    NamespaceCache,
)

_logger = logging.getLogger(__name__)
DEFAULT_NAMESPACE_EXCLUDE_GLOBS: tuple[str, ...] = (
    "dedicated-admin",
    "kube-*",
    "nvidia-gpu-operator",
    "open-cluster-management",
    "open-cluster-management-*",
    "openshift-*",
    "openshift",
    "redhat-ods-*",
)


def _parse_glob_input(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None

    candidates = value.split(",") if isinstance(value, str) else value
    return tuple(token.strip() for token in candidates if token and token.strip())


def _merge_globs(
    *glob_lists: tuple[str, ...] | None,
) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []

    for glob_list in glob_lists:
        if not glob_list:
            continue
        for pattern in glob_list:
            if pattern in seen:
                continue
            merged.append(pattern)
            seen.add(pattern)

    return tuple(merged)


@dataclass(frozen=True, slots=True)
class KubernetesWorkspaceProviderConfig:
    label_selector: str | None = None
    default_workspace: str | None = None
    namespace_exclude_globs: tuple[str, ...] = DEFAULT_NAMESPACE_EXCLUDE_GLOBS

    @classmethod
    def from_sources(
        cls,
        *,
        label_selector: str | None = None,
        default_workspace: str | None = None,
        namespace_exclude_globs: Iterable[str] | str | None = None,
    ) -> "KubernetesWorkspaceProviderConfig":
        env = os.environ

        env_globs = _parse_glob_input(env.get("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS"))

        if namespace_exclude_globs is not None:
            glob_source = _parse_glob_input(namespace_exclude_globs) or ()
        elif env_globs is not None:
            glob_source = env_globs
        else:
            glob_source = ()

        exclude_globs = _merge_globs(DEFAULT_NAMESPACE_EXCLUDE_GLOBS, glob_source)

        return cls(
            label_selector=label_selector or env.get("MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR"),
            default_workspace=default_workspace or env.get("MLFLOW_K8S_DEFAULT_WORKSPACE"),
            namespace_exclude_globs=exclude_globs,
        )


class KubernetesWorkspaceProvider(AbstractStore):
    """Workspace provider that maps MLflow workspaces to Kubernetes namespaces."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        label_selector: str | None = None,
        default_workspace: str | None = None,
        namespace_exclude_globs: Iterable[str] | str | None = None,
    ) -> None:
        del tracking_uri  # Unused but kept for compatibility with MLflow factory

        self._config = KubernetesWorkspaceProviderConfig.from_sources(
            label_selector=label_selector,
            default_workspace=default_workspace,
            namespace_exclude_globs=namespace_exclude_globs,
        )

        self._core_api, self._custom_api = self._create_api_clients()
        self._namespace_cache = NamespaceCache(
            self._core_api,
            self._config.label_selector,
            self._config.namespace_exclude_globs,
        )
        self._mlflow_config_cache = MlflowConfigCache(self._custom_api)

    def list_workspaces(self) -> Iterable[Workspace]:  # type: ignore[override]
        infos = self._namespace_cache.list_namespaces()
        return [Workspace(name=info.name, description=info.description) for info in infos]

    def get_workspace(self, workspace_name: str) -> Workspace:  # type: ignore[override]
        info = self._namespace_cache.get_namespace(workspace_name)
        if info is None:
            parts = [
                f"Workspace '{workspace_name}' was not found in the Kubernetes cluster.",
                "Each MLflow workspace maps 1:1 to a namespace.",
            ]
            if self._config.label_selector:
                parts.append(
                    f"Ensure the namespace exists and matches the configured selector "
                    f"('{self._config.label_selector}')."
                )
            raise MlflowException(" ".join(parts), RESOURCE_DOES_NOT_EXIST)

        return Workspace(name=info.name, description=info.description)

    def create_workspace(self, workspace: Workspace) -> Workspace:  # type: ignore[override]
        raise NotImplementedError("Namespace creation is not supported by this provider")

    def update_workspace(self, workspace: Workspace) -> Workspace:  # type: ignore[override]
        raise NotImplementedError("Namespace updates are not supported by this provider")

    def delete_workspace(self, workspace_name: str) -> None:  # type: ignore[override]
        raise NotImplementedError("Namespace deletion is not supported by this provider")

    def get_default_workspace(self) -> Workspace:  # type: ignore[override]
        if not self._config.default_workspace:
            raise NotImplementedError(
                "Active workspace is required. Specify one in the request path or set "
                "MLFLOW_K8S_DEFAULT_WORKSPACE."
            )
        return self.get_workspace(self._config.default_workspace)

    def resolve_artifact_root(
        self, default_artifact_root: str | None, workspace_name: str
    ) -> tuple[str | None, bool]:
        """
        Resolve the artifact root for a workspace.

        If an MLflowConfig CRD exists for the namespace with artifactRootSecret set,
        read the bucket information from the Secret and construct the artifact root.
        If artifactRootPath is also set, append it to the bucket URI.

        Returns:
            A tuple (artifact_root, should_append_workspace_prefix).
        """
        if not workspace_name:
            return default_artifact_root, True

        mlflow_config = self._mlflow_config_cache.get_config(workspace_name)
        if not mlflow_config or not mlflow_config.artifact_root_secret:
            # No override configured - use default behavior
            return default_artifact_root, True

        # Read the Secret to get bucket information
        try:
            bucket_uri = self._get_bucket_uri_from_secret(
                workspace_name, mlflow_config.artifact_root_secret
            )
        except (ApiException, KeyError, TypeError, ValueError):
            _logger.exception(
                "Failed to read Secret '%s' in namespace '%s'",
                mlflow_config.artifact_root_secret,
                workspace_name,
            )
            raise MlflowException(
                f"Failed to read Secret '{mlflow_config.artifact_root_secret}' "
                f"in namespace '{workspace_name}'",
                INTERNAL_ERROR,
            ) from None

        if not bucket_uri:
            raise MlflowException(
                f"Invalid artifact storage configuration in namespace '{workspace_name}'. "
                f"Secret '{mlflow_config.artifact_root_secret}' does not exist or is missing "
                "the 'AWS_S3_BUCKET' key.",
                INVALID_PARAMETER_VALUE,
            )

        # If artifactRootPath is set, validate and append it to the bucket URI
        if mlflow_config.artifact_root_path:
            path = mlflow_config.artifact_root_path.strip()

            # Validate the path
            if not path:
                _logger.debug(
                    f"MLflowConfig in namespace '{workspace_name}' has empty artifactRootPath. "
                    "Using bucket root."
                )
            elif validated_path := self._validate_artifact_path(path, workspace_name):
                bucket_uri = append_to_uri_path(bucket_uri, validated_path)

        return bucket_uri, False

    def _validate_artifact_path(self, path: str, workspace_name: str) -> str | None:
        """
        Validate and normalize an artifact path.

        Args:
            path: The raw artifact path from MLflowConfig.
            workspace_name: The namespace name (for error messages).

        Returns:
            The validated and normalized path, or None if path is empty after normalization.

        Raises:
            MlflowException: If path contains invalid characters or traversal attempts.
        """
        # Reject backslashes
        if "\\" in path:
            raise MlflowException(
                f"Invalid artifactRootPath '{path}' in MLflowConfig for namespace "
                f"'{workspace_name}'. Backslashes are not allowed; use forward slashes.",
                INVALID_PARAMETER_VALUE,
            )

        # Normalize the path to resolve . and .. segments
        normalized = posixpath.normpath(path)

        # Reject absolute paths
        if normalized.startswith("/"):
            raise MlflowException(
                f"Invalid artifactRootPath '{path}' in MLflowConfig for namespace "
                f"'{workspace_name}'. Absolute paths are not allowed.",
                INVALID_PARAMETER_VALUE,
            )

        # This catches cases like "../foo" that normalize but still traverse up
        if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
            raise MlflowException(
                f"Invalid artifactRootPath '{path}' in MLflowConfig for namespace "
                f"'{workspace_name}'. Path traversal ('..') is not allowed.",
                INVALID_PARAMETER_VALUE,
            )

        if normalized == ".":
            return None

        return normalized

    def _get_bucket_uri_from_secret(self, namespace: str, secret_name: str) -> str | None:
        """
        Read bucket information from a Secret and construct the bucket URI.

        Returns:
            The bucket URI in s3:// format (e.g., 's3://bucket-name'),
            or None if the Secret doesn't contain valid bucket information.
        """
        try:
            secret = self._core_api.read_namespaced_secret(
                name=secret_name,
                namespace=namespace,
            )
        except ApiException as exc:
            if exc.status == 404:
                _logger.warning(f"Secret '{secret_name}' not found in namespace '{namespace}'")
                return None
            raise

        data = secret.data or {}

        def decode(key: str) -> str | None:
            if value := data.get(key):
                try:
                    return base64.b64decode(value).decode("utf-8")
                except (binascii.Error, UnicodeDecodeError):
                    _logger.warning(
                        "Invalid base64 data for %s in Secret '%s' (namespace '%s')",
                        key,
                        secret_name,
                        namespace,
                    )
                    return None
            return None

        if not (bucket := decode("AWS_S3_BUCKET")):
            return None

        return f"s3://{bucket}"

    @staticmethod
    def _create_api_clients() -> tuple[CoreV1Api, CustomObjectsApi]:
        """Load Kubernetes config and create API clients."""
        try:
            config.load_incluster_config()
        except ConfigException:
            try:
                config.load_kube_config()
            except ConfigException as exc:  # pragma: no cover - depends on env
                raise MlflowException(
                    "Failed to load Kubernetes configuration for workspace provider",
                    error_code=INVALID_STATE,
                ) from exc
        return client.CoreV1Api(), client.CustomObjectsApi()


def _parse_workspace_uri_options(
    workspace_uri: str | None,
) -> dict[str, str | tuple[str, ...] | None]:
    """
    Parse query parameters from the workspace URI to configure the provider.
    Supported parameters:
      - label_selector
      - default_workspace
      - namespace_exclude_globs (comma-separated glob patterns)
    """

    if not workspace_uri:
        return {}

    parsed = urlparse(workspace_uri)
    query = parse_qs(parsed.query, keep_blank_values=True)

    def _get_param(name: str) -> str | None:
        values = query.get(name) or []
        value = values[-1] if values else None
        if value is None:
            return None
        value = value.strip()
        return value or None

    options: dict[str, str | tuple[str, ...] | None] = {
        "label_selector": _get_param("label_selector"),
        "default_workspace": _get_param("default_workspace"),
    }

    exclude_param = _get_param("namespace_exclude_globs")
    if exclude_param is not None:
        parsed_excludes = _parse_glob_input(exclude_param) or ()
        options["namespace_exclude_globs"] = parsed_excludes
    else:
        options["namespace_exclude_globs"] = None

    return options


def create_kubernetes_workspace_store(workspace_uri: str, **_kwargs) -> KubernetesWorkspaceProvider:
    """
    Entry point factory that instantiates the Kubernetes workspace provider.

    Args:
        workspace_uri: The resolved workspace store URI. Query parameters are used
            to override provider configuration.
        _kwargs: Additional keyword arguments ignored for forward compatibility.
    """

    options = _parse_workspace_uri_options(workspace_uri)
    return KubernetesWorkspaceProvider(
        tracking_uri=workspace_uri,
        label_selector=options.get("label_selector"),
        default_workspace=options.get("default_workspace"),
        namespace_exclude_globs=options.get("namespace_exclude_globs"),
    )
