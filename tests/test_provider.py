from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from kubernetes_workspace_provider.provider import (
    KubernetesWorkspaceProvider,
    MlflowConfigCache,
    create_kubernetes_workspace_store,
)

from mlflow.exceptions import MlflowException


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
        "kubernetes_workspace_provider.provider.config.load_kube_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.config.load_incluster_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.client.CoreV1Api",
        lambda: mock_core,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.client.CustomObjectsApi",
        lambda: mock_custom,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.watch.Watch",
        lambda: _FakeWatch(),
    )

    # Default: No MLflowConfig CRDs
    mock_custom.list_cluster_custom_object.return_value = {
        "items": [],
        "metadata": {"resourceVersion": "1"},
    }

    return SimpleNamespace(core=mock_core, custom=mock_custom)


@pytest.fixture
def core_api(mock_apis):
    """Backward-compatible fixture for existing tests."""
    return mock_apis.core


def _namespace(name: str, description: str | None = None):
    annotations = {"mlflow.kubeflow.org/workspace-description": description} if description else {}
    metadata = SimpleNamespace(name=name, annotations=annotations, resource_version="1")
    return SimpleNamespace(metadata=metadata)


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
        "kubernetes_workspace_provider.provider.watch.Watch",
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
        "kubernetes_workspace_provider.provider.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    assert cache.get_config("unknown-namespace") is None


def test_mlflow_config_cache_handles_crd_not_installed(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.side_effect = ApiException(status=404)

    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.watch.Watch",
        lambda: _FakeWatch(),
    )

    cache = MlflowConfigCache(mock_api)

    assert cache.get_config("any-namespace") is None
    assert cache._crd_available is False


def test_mlflow_config_cache_handles_permission_denied(monkeypatch):
    mock_api = MagicMock()
    mock_api.list_cluster_custom_object.side_effect = ApiException(status=403)

    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.watch.Watch",
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
        "kubernetes_workspace_provider.provider.watch.Watch",
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


def test_resolve_artifact_root_uses_secret_bucket(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "team-a-secret")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.return_value = SimpleNamespace(
        data={
            "AWS_S3_BUCKET": base64.b64encode(b"team-a-bucket").decode(),
            "AWS_ACCESS_KEY_ID": base64.b64encode(b"key").decode(),
            "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"secret").decode(),
        }
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
        "items": [_mlflow_config("team-a", "team-a-secret", "experiments/data")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.return_value = SimpleNamespace(
        data={"AWS_S3_BUCKET": base64.b64encode(b"team-a-bucket").decode()}
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
        "items": [_mlflow_config("team-a", "team-a-secret", "")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.return_value = SimpleNamespace(
        data={"AWS_S3_BUCKET": base64.b64encode(b"team-a-bucket").decode()}
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
        "items": [_mlflow_config("team-a", "nonexistent-secret")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.side_effect = ApiException(status=404)

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
        "items": [_mlflow_config("team-a", "team-a-secret")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.return_value = SimpleNamespace(
        data={"AWS_ACCESS_KEY_ID": base64.b64encode(b"key").decode()}
    )

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(
        MlflowException, match="does not exist or is missing the 'AWS_S3_BUCKET' key"
    ):
        provider.resolve_artifact_root("s3://default", "team-a")


def test_resolve_artifact_root_raises_on_transient_error(mock_apis):
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "team-a-secret")],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.side_effect = ApiException(status=500)

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(MlflowException, match="Failed to read Secret"):
        provider.resolve_artifact_root("s3://default", "team-a")


def _make_provider_with_path(mock_apis, path):
    """Helper: set up a provider whose MLflowConfig has the given artifactRootPath."""
    mock_apis.core.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("team-a")],
        metadata=SimpleNamespace(resource_version="1"),
    )
    mock_apis.custom.list_cluster_custom_object.return_value = {
        "items": [_mlflow_config("team-a", "team-a-secret", path)],
        "metadata": {"resourceVersion": "1"},
    }
    mock_apis.core.read_namespaced_secret.return_value = SimpleNamespace(
        data={"AWS_S3_BUCKET": base64.b64encode(b"team-a-bucket").decode()}
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
