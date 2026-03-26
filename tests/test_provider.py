from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from mlflow.exceptions import MlflowException
from mlflow_kubernetes_plugins.workspace_plugin.caches import (
    ARTIFACT_CONNECTION_SECRET_NAME,
    MlflowConfigCache,
    SecretCache,
)
from mlflow_kubernetes_plugins.workspace_plugin.provider import (
    KubernetesWorkspaceProvider,
    create_kubernetes_workspace_store,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR", raising=False)
    monkeypatch.delenv("MLFLOW_K8S_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS", raising=False)


class _FakeWatch:
    def stream(self, *args, **kwargs):
        return iter(())

    def stop(self):
        return None


@pytest.fixture
def mock_apis(monkeypatch):
    """Fixture that mocks both CoreV1Api and CustomObjectsApi."""
    mock_core = MagicMock()
    mock_custom = MagicMock()

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.provider.config.load_kube_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.provider.config.load_incluster_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.provider.client.CoreV1Api",
        lambda: mock_core,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.provider.client.CustomObjectsApi",
        lambda: mock_custom,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    # Default: No MLflowConfig CRDs
    mock_custom.list_cluster_custom_object.return_value = {
        "items": [],
        "metadata": {"resourceVersion": "1"},
    }

    # Default: No artifact connection secrets
    mock_core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[],
        metadata=SimpleNamespace(resource_version="1"),
    )

    return SimpleNamespace(core=mock_core, custom=mock_custom)


@pytest.fixture
def core_api(mock_apis):
    """Backward-compatible fixture for existing tests."""
    return mock_apis.core


def _namespace(name: str, description: str | None = None):
    annotations = {"mlflow.kubeflow.org/workspace-description": description} if description else {}
    metadata = SimpleNamespace(name=name, annotations=annotations, resource_version="1")
    return SimpleNamespace(metadata=metadata)


def _secret(namespace: str, bucket: str | None = None):
    data = {}
    if bucket is not None:
        data["AWS_S3_BUCKET"] = base64.b64encode(bucket.encode()).decode()
    metadata = SimpleNamespace(
        name=ARTIFACT_CONNECTION_SECRET_NAME,
        namespace=namespace,
        resource_version="1",
    )
    return SimpleNamespace(metadata=metadata, data=data)


def test_list_workspaces_uses_cache(core_api):
    namespaces = [_namespace("team-a", "Team A"), _namespace("team-b")]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="5"),
    )

    provider = KubernetesWorkspaceProvider()

    first = provider.list_workspaces()
    second = provider.list_workspaces()

    assert core_api.list_namespace.call_args_list[0][1]["label_selector"] is None
    assert [ws.name for ws in first] == ["team-a", "team-b"]
    assert [ws.description for ws in first] == ["Team A", None]
    assert [ws.default_artifact_root for ws in first] == [None, None]
    assert [ws.name for ws in second] == ["team-a", "team-b"]


def test_system_namespaces_are_filtered(core_api):
    namespaces = [
        _namespace("kube-system"),
        _namespace("openshift-config"),
        _namespace("team-a"),
    ]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="6"),
    )

    provider = KubernetesWorkspaceProvider()

    assert [ws.name for ws in provider.list_workspaces()] == ["team-a"]


def test_custom_namespace_filter_from_env(core_api, monkeypatch):
    monkeypatch.setenv("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS", "secret-*,*-internal")
    namespaces = [
        _namespace("team-a"),
        _namespace("secret-workspace"),
        _namespace("ml-internal"),
    ]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="7"),
    )

    provider = KubernetesWorkspaceProvider()

    assert [ws.name for ws in provider.list_workspaces()] == ["team-a"]


def test_get_workspace_reads_namespace(core_api):
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("analytics", "Analytics Workspace")],
        metadata=SimpleNamespace(resource_version="9"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_workspace("analytics")

    assert not core_api.read_namespace.called
    assert workspace.name == "analytics"
    assert workspace.description == "Analytics Workspace"
    assert workspace.default_artifact_root is None


def test_get_default_workspace_env(core_api, monkeypatch):
    monkeypatch.setenv("MLFLOW_K8S_DEFAULT_WORKSPACE", "shared")
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("shared", "Shared")],
        metadata=SimpleNamespace(resource_version="17"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_default_workspace()

    assert workspace.name == "shared"


def test_get_default_workspace_requires_selection(core_api):
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("alpha")],
        metadata=SimpleNamespace(resource_version="21"),
    )

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(NotImplementedError, match="Active workspace is required"):
        provider.get_default_workspace()


def test_create_workspace_store_parses_uri_options(core_api):
    store = create_kubernetes_workspace_store(
        "kubernetes://?label_selector=team%3Dmlflow&default_workspace=shared"
        "&namespace_exclude_globs=team-secret-*,%20extra"
    )

    assert isinstance(store, KubernetesWorkspaceProvider)
    assert store._config.label_selector == "team=mlflow"
    assert store._config.default_workspace == "shared"
    assert store._config.namespace_exclude_globs == (
        "dedicated-admin",
        "kube-*",
        "nvidia-gpu-operator",
        "open-cluster-management",
        "open-cluster-management-*",
        "openshift-*",
        "openshift",
        "redhat-ods-*",
        "team-secret-*",
        "extra",
    )


def _mlflow_config(namespace: str, secret: str, path: str | None = None):
    """Helper to create a mock MLflowConfig CRD response."""
    spec = {"artifactRootSecret": secret}
    if path is not None:
        spec["artifactRootPath"] = path
    return {
        "metadata": {"namespace": namespace, "resourceVersion": "1"},
        "spec": spec,
    }


def test_mlflow_config_cache_loads_configs(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.return_value = {
        "items": [
            _mlflow_config("team-a", "team-a-secret", "experiments"),
            _mlflow_config("team-b", "team-b-secret"),
        ],
        "metadata": {"resourceVersion": "10"},
    }

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    config_a = cache.get_config("team-a")
    assert config_a is not None
    assert config_a.namespace == "team-a"
    assert config_a.artifact_root_secret == "team-a-secret"
    assert config_a.artifact_root_path == "experiments"

    config_b = cache.get_config("team-b")
    assert config_b is not None
    assert config_b.artifact_root_secret == "team-b-secret"
    assert config_b.artifact_root_path is None


def test_mlflow_config_cache_returns_none_for_unknown_namespace(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.return_value = {
        "items": [],
        "metadata": {"resourceVersion": "1"},
    }

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    assert cache.get_config("unknown-namespace") is None


def test_mlflow_config_cache_handles_crd_not_installed(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.side_effect = ApiException(status=404)

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    assert cache.get_config("any-namespace") is None
    assert cache._crd_available is False


def test_mlflow_config_cache_handles_permission_denied(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.side_effect = ApiException(status=403)

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    assert cache.get_config("any-namespace") is None
    assert cache._crd_available is False


def test_mlflow_config_cache_reloads_immediately_when_crd_installed(monkeypatch):
    """When the CRD is initially missing and then installed, get_config triggers
    an immediate blocking reload instead of waiting for the background retry.
    """
    mock_api = MagicMock()
    # First call: CRD not installed
    mock_api.list_cluster_custom_object.side_effect = ApiException(status=404)

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)
    assert cache._crd_available is False
    assert cache.get_config("team-a") is None

    # Simulate CRD being installed: next API call returns data
    mock_api.list_cluster_custom_object.side_effect = None
    mock_api.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "team-a-secret", "data")],
        "metadata": {"resourceVersion": "5"},
    }

    # get_config should trigger _try_reload and pick up the new config immediately
    config = cache.get_config("team-a")

    assert config is not None
    assert config.artifact_root_secret == "team-a-secret"
    assert config.artifact_root_path == "data"
    assert cache._crd_available is True


def test_mlflow_config_cache_ensures_secret_cache_for_supported_secret(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME, "data")],
        "metadata": {"resourceVersion": "5"},
    }
    ensure_secret_cache = MagicMock()

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    MlflowConfigCache(
        mock_api,
        ensure_artifact_connection_secret_cache=ensure_secret_cache,
    )

    ensure_secret_cache.assert_called_once_with()


def test_mlflow_config_cache_does_not_ensure_secret_cache_for_other_secret(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "other-secret", "data")],
        "metadata": {"resourceVersion": "5"},
    }
    ensure_secret_cache = MagicMock()

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    MlflowConfigCache(
        mock_api,
        ensure_artifact_connection_secret_cache=ensure_secret_cache,
    )

    ensure_secret_cache.assert_not_called()


def test_resolve_artifact_root_returns_default_when_no_workspace(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    root, should_append = provider.resolve_artifact_root("s3://default-bucket", "")

    assert root == "s3://default-bucket"
    assert should_append is True


def test_resolve_artifact_root_returns_default_when_no_config(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    # No MLflowConfig CRDs (default from fixture)

    provider = KubernetesWorkspaceProvider()

    root, should_append = provider.resolve_artifact_root("s3://default-bucket", "team-a")

    assert root == "s3://default-bucket"
    assert should_append is True
    mock_apis.core.list_secret_for_all_namespaces.assert_not_called()


def test_resolve_artifact_root_uses_secret_bucket(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    root, should_append = provider.resolve_artifact_root("s3://default-bucket", "team-a")

    assert root == "s3://team-a-bucket"
    assert should_append is False


def test_resolve_artifact_root_appends_path(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME, "experiments/data")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    root, should_append = provider.resolve_artifact_root("s3://default-bucket", "team-a")

    assert root == "s3://team-a-bucket/experiments/data"
    assert should_append is False


def test_resolve_artifact_root_handles_empty_path(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME, "")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    root, should_append = provider.resolve_artifact_root("s3://default", "team-a")

    assert root == "s3://team-a-bucket"
    assert should_append is False


def test_resolve_artifact_root_raises_on_secret_not_found(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(
        MlflowException, match="does not exist or is missing the 'AWS_S3_BUCKET' key"
    ):
        provider.resolve_artifact_root("s3://default", "team-a")


def test_resolve_artifact_root_raises_on_missing_bucket_key(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(
        MlflowException, match="does not exist or is missing the 'AWS_S3_BUCKET' key"
    ):
        provider.resolve_artifact_root("s3://default", "team-a")


def test_secret_cache_starts_lazily(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    assert provider._secret_cache is None
    mock_apis.core.list_secret_for_all_namespaces.assert_not_called()


def test_secret_cache_starts_when_mlflow_config_seen(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()

    assert provider._secret_cache is not None
    mock_apis.core.list_secret_for_all_namespaces.assert_called_once()


def test_secret_cache_raises_on_transient_error(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.side_effect = ApiException(status=500)

    with pytest.raises(MlflowException, match="Failed to list Kubernetes secrets"):
        KubernetesWorkspaceProvider()


def _make_provider_with_path(mock_apis, path):
    """Helper: set up a provider whose MLflowConfig has the given artifactRootPath."""
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME, path)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    return KubernetesWorkspaceProvider()


@pytest.mark.parametrize(
    "malicious_path",
    [
        "..",
        "../foo",
        "foo/../../bar",
        "foo/../../../etc/passwd",
        "a/b/../../../secret",
    ],
)
def test_validate_artifact_path_rejects_traversal(mock_apis, malicious_path):
    provider = _make_provider_with_path(mock_apis, malicious_path)

    with pytest.raises(MlflowException, match="Path traversal"):
        provider.resolve_artifact_root("s3://default", "team-a")


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/etc/passwd",
        "/foo",
        "/absolute/path/to/data",
    ],
)
def test_validate_artifact_path_rejects_absolute(mock_apis, absolute_path):
    provider = _make_provider_with_path(mock_apis, absolute_path)

    with pytest.raises(MlflowException, match="Absolute paths are not allowed"):
        provider.resolve_artifact_root("s3://default", "team-a")


@pytest.mark.parametrize(
    "backslash_path",
    [
        "foo\\bar",
        "experiments\\data",
        "a\\b\\c",
    ],
)
def test_validate_artifact_path_rejects_backslashes(mock_apis, backslash_path):
    provider = _make_provider_with_path(mock_apis, backslash_path)

    with pytest.raises(MlflowException, match="Backslashes are not allowed"):
        provider.resolve_artifact_root("s3://default", "team-a")


def test_validate_artifact_path_normalizes_dot_to_bucket_root(mock_apis):
    provider = _make_provider_with_path(mock_apis, ".")

    root, should_append = provider.resolve_artifact_root("s3://default", "team-a")

    assert root == "s3://team-a-bucket"
    assert should_append is False


def test_validate_artifact_path_normalizes_dot_slash_prefix(mock_apis):
    provider = _make_provider_with_path(mock_apis, "./foo")

    root, should_append = provider.resolve_artifact_root("s3://default", "team-a")

    assert root == "s3://team-a-bucket/foo"
    assert should_append is False


def test_validate_artifact_path_normalizes_redundant_slashes(mock_apis):
    provider = _make_provider_with_path(mock_apis, "a//b///c")

    root, should_append = provider.resolve_artifact_root("s3://default", "team-a")

    assert root == "s3://team-a-bucket/a/b/c"
    assert should_append is False


def test_get_workspace_includes_artifact_root(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a", "Team A")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME, "experiments")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_workspace("team-a")

    assert workspace.name == "team-a"
    assert workspace.description == "Team A"
    assert workspace.default_artifact_root == "s3://team-a-bucket/experiments"


def test_list_workspaces_includes_artifact_root(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a"), _namespace("team-b")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()
    workspaces = provider.list_workspaces()

    by_name = {ws.name: ws for ws in workspaces}
    assert by_name["team-a"].default_artifact_root == "s3://team-a-bucket"
    assert by_name["team-b"].default_artifact_root is None


def test_get_workspace_ignores_config_error(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_workspace("team-a")

    assert workspace.name == "team-a"
    assert workspace.default_artifact_root is None


def test_resolve_artifact_root_rejects_wrong_secret_name(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "some-other-secret")],
        "metadata": {"resourceVersion": "1"},
    }

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(MlflowException, match="only 'mlflow-artifact-connection' is supported"):
        provider.resolve_artifact_root("s3://default", "team-a")
    mock_apis.core.list_secret_for_all_namespaces.assert_not_called()


def test_secret_cache_handles_permission_denied(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", ARTIFACT_CONNECTION_SECRET_NAME)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.list_secret_for_all_namespaces.side_effect = ApiException(status=403)

    provider = KubernetesWorkspaceProvider()
    secret_info = provider._ensure_secret_cache().get_secret("any-namespace")

    assert secret_info is None
    assert provider._secret_cache is not None
    assert provider._secret_cache._available is False


def test_secret_cache_does_not_reload_from_request_path(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_secret_for_all_namespaces.side_effect = ApiException(status=403)

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = SecretCache(mock_api)

    assert cache.get_secret("team-a") is None
    assert cache.get_secret("team-a") is None
    assert mock_api.list_secret_for_all_namespaces.call_count == 1


def test_secret_cache_loads_secrets(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a", "team-a-bucket"), _secret("team-b", "team-b-bucket")],
        metadata=SimpleNamespace(resource_version="10"),
    )

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = SecretCache(mock_api)

    info_a = cache.get_secret("team-a")
    assert info_a is not None
    assert info_a.bucket_uri == "s3://team-a-bucket"

    info_b = cache.get_secret("team-b")
    assert info_b is not None
    assert info_b.bucket_uri == "s3://team-b-bucket"

    assert cache.get_secret("unknown") is None


def test_secret_cache_handles_missing_bucket_key(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_secret_for_all_namespaces.return_value = SimpleNamespace(
        items=[_secret("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = SecretCache(mock_api)

    info = cache.get_secret("team-a")
    assert info is not None
    assert info.bucket_uri is None
