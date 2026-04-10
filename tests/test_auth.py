"""Tests for Kubernetes auth behavior with Flask and workspace helpers.

This suite exercises request overrides, authorization rules, and caching logic.
"""

import base64
import json
import time
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow_kubernetes_plugins.auth.resource_names as resource_names_mod
import pytest
from fastapi.testclient import TestClient
from flask import Flask, request
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    GetLatestVersions,
    GetModelVersion,
    GetRegisteredModel,
    RenameRegisteredModel,
)
from mlflow.protos.service_pb2 import (
    AddDatasetToExperiments,
    AttachModelToGatewayEndpoint,
    CancelPromptOptimizationJob,
    CreateDataset,
    CreateGatewayEndpoint,
    CreateGatewayEndpointBinding,
    CreateGatewayModelDefinition,
    CreateGatewaySecret,
    CreatePromptOptimizationJob,
    CreateRun,
    CreateWorkspace,
    DeleteDataset,
    DeleteDatasetRecords,
    DeleteGatewayEndpointBinding,
    DeleteGatewayEndpointTag,
    DeletePromptOptimizationJob,
    DeleteWorkspace,
    DetachModelFromGatewayEndpoint,
    GetDataset,
    GetGatewayEndpoint,
    GetGatewayModelDefinition,
    GetGatewaySecretInfo,
    GetMetricHistoryBulkInterval,
    GetPromptOptimizationJob,
    GetWorkspace,
    ListGatewayEndpointBindings,
    ListWorkspaces,
    RemoveDatasetFromExperiments,
    SearchPromptOptimizationJobs,
    SetDatasetTags,
    SetGatewayEndpointTag,
    StartTraceV3,
    UpdateGatewaySecret,
    UpdateWorkspace,
    UpsertDatasetRecords,
)
from mlflow.protos.webhooks_pb2 import DeleteWebhook, GetWebhook, UpdateWebhook
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow_kubernetes_plugins.auth.authorizer import (
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
    _AuthorizationCache,
    _AuthorizationCacheKey,
    _CacheEntry,
)
from mlflow_kubernetes_plugins.auth.collection_filters import (
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    COLLECTION_POLICY_REQUEST_RUN_IDS,
    COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
    apply_request_collection_filter,
    apply_response_collection_filters,
)
from mlflow_kubernetes_plugins.auth.compiler import (
    _compile_named_path_pattern,
    _extract_path_params,
    _find_authorization_rules,
    _reset_compiled_rules,
)
from mlflow_kubernetes_plugins.auth.constants import (
    AUTHORIZATION_MODE_ENV,
    REMOTE_GROUPS_HEADER_ENV,
    REMOTE_USER_HEADER_ENV,
    RESOURCE_ASSISTANTS,
    RESOURCE_DATASETS,
    RESOURCE_EXPERIMENTS,
    RESOURCE_GATEWAY_ENDPOINTS,
    RESOURCE_GATEWAY_MODEL_DEFINITIONS,
    RESOURCE_GATEWAY_SECRETS,
    RESOURCE_REGISTERED_MODELS,
)
from mlflow_kubernetes_plugins.auth.core import (
    _canonicalize_path,
    _is_unprotected_path,
    _parse_jwt_subject,
    _parse_remote_groups,
    _RequestIdentity,
)
from mlflow_kubernetes_plugins.auth.request_context import AuthorizationRequest
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,
    RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,
    RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,
    RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,
    RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME,
    RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,
    _NameLookupCache,
    apply_response_cache_updates,
    resolve_resource_names,
)
from mlflow_kubernetes_plugins.auth.rules import (
    PATH_AUTHORIZATION_RULES,
    REQUEST_AUTHORIZATION_RULES,
    AuthorizationRule,
    _normalize_rules,
)

from conftest import _authorize_request


@pytest.fixture(autouse=True)
def _compile_rules(compile_auth_rules):
    """Ensure authorization rules are populated before each test."""
    compile_auth_rules(
        [
            ("/api/2.0/mlflow/runs/create", CreateRun, ["POST"]),
            ("/api/3.0/mlflow/workspaces", ListWorkspaces, ["GET"]),
            ("/api/3.0/mlflow/workspaces", CreateWorkspace, ["POST"]),
            ("/api/3.0/mlflow/workspaces/<workspace_name>", GetWorkspace, ["GET"]),
            ("/api/3.0/mlflow/workspaces/<workspace_name>", UpdateWorkspace, ["PATCH"]),
            ("/api/3.0/mlflow/workspaces/<workspace_name>", DeleteWorkspace, ["DELETE"]),
        ]
    )


def _make_jwt_token(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.signature"


def _invalidate_experiment_lookup_cache(experiment_id: str) -> None:
    resource_names_mod._experiment_name_cache.invalidate(experiment_id)


def _invalidate_run_lookup_cache(run_id: str) -> None:
    resource_names_mod._run_experiment_name_cache.invalidate(run_id)


def test_parse_jwt_subject_returns_claim_value_when_present():
    token = _make_jwt_token({"sub": "alice"})
    assert _parse_jwt_subject(token, "sub") == "alice"


def test_parse_jwt_subject_missing_claim_returns_none():
    token = _make_jwt_token({"sub": "alice"})
    assert _parse_jwt_subject(token, "email") is None


def test_parse_jwt_subject_empty_or_non_string_claim_returns_none():
    assert _parse_jwt_subject(_make_jwt_token({"sub": ""}), "sub") is None
    assert _parse_jwt_subject(_make_jwt_token({"sub": {"nested": True}}), "sub") is None


def test_parse_jwt_subject_token_without_payload_segment_returns_none():
    assert _parse_jwt_subject("malformed-token", "sub") is None


def test_parse_jwt_subject_invalid_base64_payload_returns_none():
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    token = f"{header}.@@not-base64@@.signature"
    assert _parse_jwt_subject(token, "sub") is None


def test_parse_jwt_subject_invalid_json_payload_returns_none():
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    body = base64.urlsafe_b64encode(b"not json").decode().rstrip("=")
    token = f"{header}.{body}.signature"
    assert _parse_jwt_subject(token, "sub") is None


def test_request_identity_subject_hash_self_subject_access_review():
    identity = _RequestIdentity(token="token-value")
    digest = identity.subject_hash(AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW)
    assert digest == sha256(b"token-value").hexdigest()


def test_request_identity_subject_hash_subject_access_review_normalizes(monkeypatch):
    identity = _RequestIdentity(user="  alice  ", groups=("group-b", "group-a"))
    digest = identity.subject_hash(AuthorizationMode.SUBJECT_ACCESS_REVIEW)
    expected = sha256(b"alice\x00group-a\x00group-b").hexdigest()
    assert digest == expected


def test_request_identity_subject_hash_missing_user_raises():
    identity = _RequestIdentity(user=None)
    with pytest.raises(MlflowException, match="X-Remote-User header required"):
        identity.subject_hash(
            AuthorizationMode.SUBJECT_ACCESS_REVIEW,
            missing_user_label="X-Remote-User header required",
        )


@pytest.mark.parametrize(
    ("header_value", "separator", "expected"),
    [
        (None, "|", ()),
        ("", "|", ()),
        ("group-a|group-b", "", ("group-a|group-b",)),
        (" group-a | group-b ", "|", ("group-a", "group-b")),
        ("one,two", ",", ("one", "two")),
    ],
)
def test_parse_remote_groups_variations(header_value, separator, expected):
    assert _parse_remote_groups(header_value, separator) == expected


def test_canonicalize_path_prefers_scope_and_path_info(monkeypatch):
    path = _canonicalize_path(
        raw_path="/mlflow/api/2.0/mlflow/runs/create",
        scope_path="/api/2.0/mlflow/runs/create",
        root_path="/mlflow",
    )
    assert path == "/api/2.0/mlflow/runs/create"

    path = _canonicalize_path(
        raw_path="/prefix/api/2.0/mlflow/runs/create",
        path_info="/api/2.0/mlflow/runs/create",
        script_name="/prefix",
    )
    assert path == "/api/2.0/mlflow/runs/create"


def test_canonicalize_path_static_prefix_applies_to_static_routes_only(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")

    ajax_path = "/mlflow/ajax-api/2.0/mlflow/runs/create"
    api_path = "/mlflow/api/2.0/mlflow/runs/create"
    health_path = "/mlflow/health"
    metrics_path = "/mlflow/metrics"
    version_path = "/mlflow/version"

    assert _canonicalize_path(raw_path=ajax_path) == "/ajax-api/2.0/mlflow/runs/create"
    assert _canonicalize_path(raw_path=api_path) == "/api/2.0/mlflow/runs/create"
    assert _canonicalize_path(raw_path=health_path) == "/health"
    assert _canonicalize_path(raw_path=metrics_path) == "/metrics"
    assert _canonicalize_path(raw_path=version_path) == "/version"


def test_compile_named_path_pattern_escapes_literal_segments():
    pattern = _compile_named_path_pattern("/api/test.v1/<resource_id>")
    assert pattern.fullmatch("/api/test.v1/123")
    assert not pattern.fullmatch("/api/testXv1/123")


def test_request_identity_subject_hash_missing_token_in_ssar():
    identity = _RequestIdentity(token=None)
    with pytest.raises(
        MlflowException,
        match="Bearer token is required for SelfSubjectAccessReview mode.",
    ) as exc:
        identity.subject_hash(AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW)

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.UNAUTHENTICATED)


def test_kubernetes_auth_config_invalid_mode(monkeypatch):
    monkeypatch.setenv(AUTHORIZATION_MODE_ENV, "invalid-mode")
    with pytest.raises(MlflowException, match="must be one of"):
        KubernetesAuthConfig.from_env()


def test_kubernetes_auth_config_empty_user_header(monkeypatch):
    monkeypatch.setenv(REMOTE_USER_HEADER_ENV, "   ")
    with pytest.raises(MlflowException, match="cannot be empty") as exc:
        KubernetesAuthConfig.from_env()

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(
        databricks_pb2.INVALID_PARAMETER_VALUE
    )


def test_kubernetes_auth_config_empty_groups_header(monkeypatch):
    monkeypatch.setenv(REMOTE_GROUPS_HEADER_ENV, "")
    with pytest.raises(MlflowException, match="cannot be empty"):
        KubernetesAuthConfig.from_env()


def test_flask_create_run_request_processing(monkeypatch):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")
    fake_config = SimpleNamespace(
        host="https://cluster.local",
        ssl_ca_cert=None,
        verify_ssl=True,
        proxy=None,
        no_proxy=None,
        proxy_headers=None,
        safe_chars_for_path_param=None,
        connection_pool_maxsize=10,
    )
    with (
        patch(
            "mlflow_kubernetes_plugins.auth.authorizer.KubernetesAuthorizer.is_allowed",
            return_value=True,
        ),
        patch(
            "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
            return_value=fake_config,
        ),
        patch(
            "mlflow_kubernetes_plugins.auth.middleware.resolve_workspace_from_header",
            return_value=SimpleNamespace(name="default"),
        ),
    ):
        client = TestClient(create_app(app))

        with patch(
            "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"
        ):
            response = client.post(
                "/api/2.0/mlflow/runs/create",
                json={"experiment_id": "0"},
                headers={
                    "Authorization": "Bearer test-token",
                    "Host": "localhost",
                },
            )

    assert response.status_code == 200
    assert response.json()["run"]["info"]["user_id"] == "k8s-user"


def _build_workspace_app(monkeypatch):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/3.0/mlflow/workspaces", methods=["GET"])
    def list_workspaces():
        return {"workspaces": [{"name": "team-a"}]}

    @app.route("/api/3.0/mlflow/workspaces", methods=["POST"])
    def create_workspace_endpoint():
        payload = request.get_json()
        return (
            {"workspace": {"name": payload.get("name")}},
            201,
        )

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer.KubernetesAuthorizer.is_allowed",
        lambda self, identity, resource, verb, namespace: True,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: None,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.resolve_workspace_from_header",
        lambda _header: None,
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    return TestClient(create_app(app))


def test_list_workspaces_without_context(monkeypatch):
    client = _build_workspace_app(monkeypatch)

    with patch(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.get(
            "/api/3.0/mlflow/workspaces",
            headers={
                "Authorization": "Bearer list-token",
                "Host": "localhost",
            },
        )

    assert response.status_code == 200
    assert response.json()["workspaces"] == [{"name": "team-a"}]


def test_create_workspace_requests_are_denied(monkeypatch):
    mock_is_allowed = Mock(return_value=True)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer.KubernetesAuthorizer.is_allowed",
        mock_is_allowed,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: None,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.resolve_workspace_from_header",
        lambda _header: None,
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/3.0/mlflow/workspaces", methods=["POST"])
    def create_workspace_endpoint():
        payload = request.get_json()
        return {"workspace": {"name": payload.get("name")}}, 201

    client = TestClient(create_app(app))

    with patch(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-new"},
            headers={
                "Authorization": "Bearer create-token",
                "Host": "localhost",
            },
        )

    assert response.status_code == 403
    assert "Workspace create" in response.json()["error"]["message"]
    mock_is_allowed.assert_not_called()


def _build_flask_auth_app(monkeypatch, *, is_allowed=True):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    mock_is_allowed = Mock(return_value=is_allowed)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer.KubernetesAuthorizer.is_allowed", mock_is_allowed
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration", lambda: None
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.resolve_workspace_from_header",
        lambda _header: SimpleNamespace(name="default"),
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    return TestClient(create_app(app)), mock_is_allowed


def test_missing_authorization_header_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post(
        "/api/2.0/mlflow/runs/create",
        json={"experiment_id": "0"},
        headers={"Host": "localhost"},
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header" in response.json()["error"]["message"]
    )
    mock_is_allowed.assert_not_called()


def test_invalid_bearer_scheme_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post(
        "/api/2.0/mlflow/runs/create",
        json={"experiment_id": "0"},
        headers={
            "Authorization": "Token bad-token",
            "Host": "localhost",
        },
    )
    assert response.status_code == 401
    assert "Bearer" in response.json()["error"]["message"]
    mock_is_allowed.assert_not_called()


def test_forwarded_access_token_header_allows_request(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={
                "X-Forwarded-Access-Token": "test-token",
                "Host": "localhost",
            },
        )

    assert response.status_code == 200
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_is_allowed.call_args[0]
    assert identity.token == "test-token"
    assert subresource is None


def test_invalid_authorization_header_with_forwarded_token(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={
                "Authorization": "Basic bad-token",
                "X-Forwarded-Access-Token": "forwarded-token",
                "Host": "localhost",
            },
        )

    assert response.status_code == 200
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert subresource is None


def test_permission_denied_returns_403(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch, is_allowed=False)

    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={
                "Authorization": "Bearer test-token",
                "Host": "localhost",
            },
        )

    assert response.status_code == 403
    assert "Permission denied" in response.json()["error"]["message"]
    assert mock_is_allowed.call_count == 1
    identity, resource, verb, namespace, subresource = mock_is_allowed.call_args_list[0][0]
    assert identity.token == "test-token"
    assert (resource, verb, namespace) == ("experiments", "update", "default")
    assert subresource is None


def test_workspace_scope_string_is_normalized(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = True

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [AuthorizationRule("list", resource="experiments")],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    config = KubernetesAuthConfig()
    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/ajax-api/2.0/mlflow/experiments/search",
            method="GET",
            workspace=" team-a ",
        ),
        authorizer=authorizer,
        config_values=config,
    )

    call_args = authorizer.is_allowed.call_args[0]
    identity_arg = call_args[0]
    assert identity_arg.token == "valid-token"
    assert call_args[1:] == ("experiments", "list", "team-a", None)
    assert result.username == "k8s-user"


def test_workspace_listing_allows_missing_context(monkeypatch):
    authorizer = Mock()
    rule = AuthorizationRule(
        None,
        apply_workspace_filter=True,
        requires_workspace=False,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    config = KubernetesAuthConfig()
    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer list-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/3.0/mlflow/workspaces",
            method="GET",
            workspace=None,
        ),
        authorizer=authorizer,
        config_values=config,
    )

    assert result.username == "k8s-user"
    assert result.rules[0].apply_workspace_filter
    authorizer.is_allowed.assert_not_called()


def test_unmapped_endpoint_returns_not_found(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = True
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: None,
    )

    config = KubernetesAuthConfig()

    with pytest.raises(MlflowException, match=r"Endpoint not found\.") as exc:
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer missing-rule-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/unknown",
                method="GET",
                workspace="default",
            ),
            authorizer=authorizer,
            config_values=config,
        )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND)
    authorizer.is_allowed.assert_not_called()


def test_subject_access_review_mode_uses_remote_headers(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = True
    rule = AuthorizationRule("list", resource=RESOURCE_EXPERIMENTS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )

    config = KubernetesAuthConfig(authorization_mode=AuthorizationMode.SUBJECT_ACCESS_REVIEW)

    result = _authorize_request(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value="proxy-user",
            remote_groups_header_value="group-a|group-b",
            path="/ajax-api/2.0/mlflow/experiments/search",
            method="GET",
            workspace="team-a",
        ),
        authorizer=authorizer,
        config_values=config,
    )

    identity, resource, verb, namespace, subresource = authorizer.is_allowed.call_args[0]
    assert identity.token is None
    assert identity.user == "proxy-user"
    assert identity.groups == ("group-a", "group-b")
    assert (resource, verb, namespace) == ("experiments", "list", "team-a")
    assert subresource is None
    assert result.username == "proxy-user"


def test_subject_access_review_mode_requires_user_header(monkeypatch):
    authorizer = Mock()
    rule = AuthorizationRule("get", resource=RESOURCE_EXPERIMENTS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )

    config = KubernetesAuthConfig(authorization_mode=AuthorizationMode.SUBJECT_ACCESS_REVIEW)

    with pytest.raises(MlflowException, match="Missing required") as exc:
        _authorize_request(
            AuthorizationRequest(
                authorization_header=None,
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value="group-a|group-b",
                path="/ajax-api/2.0/mlflow/experiments/get",
                method="GET",
                workspace="team-a",
            ),
            authorizer=authorizer,
            config_values=config,
        )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.UNAUTHENTICATED)
    authorizer.is_allowed.assert_not_called()


def test_gateway_endpoint_create_requires_model_definition_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    # First call: endpoint create allowed, second call: model definitions use denied
    authorizer.is_allowed.side_effect = [True, False]
    rule = AuthorizationRule("create", resource=RESOURCE_GATEWAY_ENDPOINTS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/endpoints/create",
        method="POST",
        data=json.dumps({"name": "endpoint-1"}),
        content_type="application/json",
    ):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/api/2.0/mlflow/gateway/endpoints/create",
                    method="POST",
                    workspace="team-a",
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    assert "'use' permission on gateway model definitions" in exc.value.message
    assert authorizer.is_allowed.call_count == 2
    # Verify 'create' verb on 'gatewaymodeldefinitions/use' subresource
    assert authorizer.is_allowed.call_args_list[1][0][1:] == (
        RESOURCE_GATEWAY_MODEL_DEFINITIONS,
        "create",
        "team-a",
        "use",
    )


def test_gateway_endpoint_update_requires_model_definition_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    # First call: endpoint update allowed, second call: model definitions use denied
    authorizer.is_allowed.side_effect = [True, False]
    rule = AuthorizationRule("update", resource=RESOURCE_GATEWAY_ENDPOINTS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/endpoints/update",
        method="PATCH",
        data=json.dumps({"endpoint_id": "ep-1"}),
        content_type="application/json",
    ):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/api/2.0/mlflow/gateway/endpoints/update",
                    method="PATCH",
                    workspace="team-a",
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    assert "'use' permission on gateway model definitions" in exc.value.message
    # Verify 'create' verb on 'gatewaymodeldefinitions/use' subresource
    assert authorizer.is_allowed.call_args_list[1][0][1:] == (
        RESOURCE_GATEWAY_MODEL_DEFINITIONS,
        "create",
        "team-a",
        "use",
    )


def test_gateway_model_definition_create_requires_secret_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    # First call: model definition create allowed, second call: secrets use denied
    authorizer.is_allowed.side_effect = [True, False]
    rule = AuthorizationRule("create", resource=RESOURCE_GATEWAY_MODEL_DEFINITIONS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/model-definitions/create",
        method="POST",
        data=json.dumps({"name": "model-def-1"}),
        content_type="application/json",
    ):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/api/2.0/mlflow/gateway/model-definitions/create",
                    method="POST",
                    workspace="team-a",
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    assert "'use' permission on gateway secrets" in exc.value.message
    # Verify 'create' verb on 'gatewaysecrets/use' subresource
    assert authorizer.is_allowed.call_args_list[1][0][1:] == (
        RESOURCE_GATEWAY_SECRETS,
        "create",
        "team-a",
        "use",
    )


def test_gateway_model_definition_update_requires_secret_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    # First call: model definition update allowed, second call: secrets use denied
    authorizer.is_allowed.side_effect = [True, False]
    rule = AuthorizationRule("update", resource=RESOURCE_GATEWAY_MODEL_DEFINITIONS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/model-definitions/update",
        method="PATCH",
        data=json.dumps({"model_definition_id": "md-1"}),
        content_type="application/json",
    ):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/api/2.0/mlflow/gateway/model-definitions/update",
                    method="PATCH",
                    workspace="team-a",
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    assert "'use' permission on gateway secrets" in exc.value.message
    # Verify 'create' verb on 'gatewaysecrets/use' subresource
    assert authorizer.is_allowed.call_args_list[1][0][1:] == (
        RESOURCE_GATEWAY_SECRETS,
        "create",
        "team-a",
        "use",
    )


def test_gateway_endpoint_create_allows_named_model_definition_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()

    def _is_allowed(identity, resource, verb, namespace, subresource=None, resource_name=None):
        if resource == RESOURCE_GATEWAY_ENDPOINTS:
            return True
        if (
            resource == RESOURCE_GATEWAY_MODEL_DEFINITIONS
            and subresource == "use"
            and resource_name == "model-def-a"
        ):
            return True
        return False

    authorizer.is_allowed.side_effect = _is_allowed
    rule = AuthorizationRule("create", resource=RESOURCE_GATEWAY_ENDPOINTS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_gateway_model_definition=lambda model_definition_id: SimpleNamespace(name="model-def-a")
        ),
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/endpoints/create",
        method="POST",
        data=json.dumps(
            {"name": "endpoint-a", "model_configs": [{"model_definition_id": "model-def-id-a"}]}
        ),
        content_type="application/json",
    ):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/gateway/endpoints/create",
                method="POST",
                workspace="team-a",
                json_body={"name": "endpoint-a", "model_configs": [{"model_definition_id": "model-def-id-a"}]},
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_args_list[-1].kwargs == {"resource_name": "model-def-a"}


def test_gateway_model_definition_create_allows_named_secret_use(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()

    def _is_allowed(identity, resource, verb, namespace, subresource=None, resource_name=None):
        if resource == RESOURCE_GATEWAY_MODEL_DEFINITIONS:
            return True
        if resource == RESOURCE_GATEWAY_SECRETS and subresource == "use" and resource_name == "secret-a":
            return True
        return False

    authorizer.is_allowed.side_effect = _is_allowed
    rule = AuthorizationRule("create", resource=RESOURCE_GATEWAY_MODEL_DEFINITIONS)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_secret_info=lambda secret_id: SimpleNamespace(secret_name="secret-a")),
    )

    with app.test_request_context(
        "/api/2.0/mlflow/gateway/model-definitions/create",
        method="POST",
        data=json.dumps(
            {
                "name": "model-def-a",
                "secret_id": "secret-id-a",
                "provider": "openai",
                "model_name": "gpt-4o",
            }
        ),
        content_type="application/json",
    ):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/gateway/model-definitions/create",
                method="POST",
                workspace="team-a",
                json_body={
                    "name": "model-def-a",
                    "secret_id": "secret-id-a",
                    "provider": "openai",
                    "model_name": "gpt-4o",
                },
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_args_list[-1].kwargs == {"resource_name": "secret-a"}


def test_workspace_scope_falls_back_to_path_params(monkeypatch):
    authorizer = Mock()
    authorizer.can_access_workspace.return_value = True
    rule = AuthorizationRule(None, requires_workspace=False, workspace_access_check=True)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer scope-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/3.0/mlflow/workspaces/team-a",
            method="GET",
            workspace=None,
            path_params={"workspace_name": "team-a"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    args = authorizer.can_access_workspace.call_args[0]
    assert args[0].token == "scope-token"
    assert args[1:] == ("team-a",)
    kwargs = authorizer.can_access_workspace.call_args[1]
    assert kwargs == {"verb": "get"}


def test_authorize_request_persists_recovered_workspace_in_request_context(monkeypatch):
    authorizer = Mock()
    authorizer.can_access_workspace.return_value = True
    rule = AuthorizationRule(None, requires_workspace=False, workspace_access_check=True)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer scope-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/3.0/mlflow/workspaces/team-a",
            method="GET",
            workspace=None,
            path_params={"workspace_name": "team-a"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert result.request_context.workspace == "team-a"


def test_extract_path_params_includes_compiled_handler_routes():
    assert _extract_path_params("/api/3.0/mlflow/workspaces/team-a", "GET") == {
        "workspace_name": "team-a"
    }


def test_workspace_create_requests_are_denied(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    rule = AuthorizationRule("create", deny=True, requires_workspace=False)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/3.0/mlflow/workspaces",
        method="POST",
        data=json.dumps({"name": "team-new"}),
        content_type="application/json",
    ):
        with pytest.raises(
            MlflowException, match="Workspace create, update, and delete operations"
        ) as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer create-token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/api/3.0/mlflow/workspaces",
                    method="POST",
                    workspace=None,
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    authorizer.is_allowed.assert_not_called()


def test_authorize_request_retries_with_resource_name_after_broad_denial(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    rule = AuthorizationRule(
        "get",
        resource=RESOURCE_EXPERIMENTS,
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_experiment=lambda experiment_id: SimpleNamespace(name="exp-a")),
    )

    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer lookup-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/experiments/get",
            method="GET",
            workspace="team-a",
            query_params={"experiment_id": "123"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    first_call = authorizer.is_allowed.call_args_list[0]
    assert first_call.args[1:] == (RESOURCE_EXPERIMENTS, "get", "team-a", None)
    assert first_call.kwargs == {}

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.args[1:] == (RESOURCE_EXPERIMENTS, "get", "team-a", None)
    assert second_call.kwargs == {"resource_name": "exp-a"}


def test_resolve_resource_names_resolves_experiment_ids_to_names(monkeypatch):
    experiment_names = {
        "dataset-exp-1": "exp-a",
        "dataset-exp-2": "exp-b",
    }
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(name=experiment_names[experiment_id])
        ),
    )

    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/datasets/add-to-experiments",
            method="POST",
            workspace="team-a",
            json_body={
                "dataset_id": "dataset-1",
                "experiment_ids": ["dataset-exp-1", "dataset-exp-2", "dataset-exp-1"],
            },
        ),
        (RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,),
    )

    assert names == ("exp-a", "exp-b")


def test_resolve_resource_names_reads_run_id_from_post_query_params(monkeypatch):
    _invalidate_run_lookup_cache("run-query")
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="exp-id-1")),
            get_experiment=lambda experiment_id: SimpleNamespace(name="exp-a"),
        ),
    )

    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/ajax-api/2.0/mlflow/upload-artifact",
            method="POST",
            workspace="team-a",
            query_params={"run_id": "run-query"},
            json_body={"path": "artifact.txt"},
        ),
        (RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
    )

    assert names == ("exp-a",)


def test_resolve_resource_names_reads_experiment_id_from_post_body(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_experiment=lambda experiment_id: SimpleNamespace(name="exp-a")),
    )

    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/ajax-api/3.0/jobs/search",
            method="POST",
            workspace="team-a",
            json_body={"experiment_id": "body-exp"},
        ),
        (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    )

    assert names == ("exp-a",)


def test_resolve_resource_names_rejects_conflicting_single_value_query_params():
    with pytest.raises(RuntimeError, match="Missing required parameter 'experiment_id'"):
        resolve_resource_names(
            AuthorizationRequest(
                authorization_header=None,
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/ajax-api/3.0/jobs/search",
                method="POST",
                workspace="team-a",
                query_params={"experiment_id": ["query-exp-a", "query-exp-b"]},
                json_body={"job_name": "search"},
            ),
            (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
        )


def test_resolve_resource_names_post_ignores_query_params(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_experiment=lambda experiment_id: SimpleNamespace(name=f"exp-{experiment_id}")),
    )

    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/ajax-api/3.0/jobs/search",
            method="POST",
            workspace="team-a",
            query_params={"experiment_id": "post-ignore-query"},
            json_body={"experiment_id": "post-ignore-body"},
        ),
        (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    )

    assert names == ("exp-post-ignore-body",)


def test_resolve_resource_names_delete_json_body_does_not_fallback_to_query_params():
    with pytest.raises(RuntimeError, match="Missing required parameter 'experiment_id'"):
        resolve_resource_names(
            AuthorizationRequest(
                authorization_header=None,
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/experiments/delete",
                method="DELETE",
                workspace="team-a",
                query_params={"experiment_id": "query-exp"},
                json_body={},
            ),
            (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
        )


def test_resolve_resource_names_resolves_dataset_id_to_name(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_dataset=lambda dataset_id: SimpleNamespace(name="dataset-a")),
    )

    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/datasets/get",
            method="GET",
            workspace="team-a",
            query_params={"dataset_id": "dataset-1"},
        ),
        (RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,),
    )

    assert names == ("dataset-a",)


def test_resolve_gateway_model_definition_names_for_use_deduplicates_across_sources(monkeypatch):
    store = SimpleNamespace(
        get_gateway_model_definition=lambda model_definition_id: SimpleNamespace(
            name={
                "md-body": "model-def-body",
                "md-query": "model-def-query",
                "md-extra": "model-def-extra",
            }[model_definition_id]
        )
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: store,
    )

    names = resource_names_mod.resolve_gateway_model_definition_names_for_use(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/gateway/endpoints/create",
            method="POST",
            workspace="team-a",
            query_params={"model_definition_id": ["md-query", "md-body"]},
            json_body={
                "model_definition_id": "md-body",
                "model_config": {"model_definition_id": "md-query"},
                "model_configs": [
                    {"model_definition_id": "md-extra"},
                    {"model_definition_id": "md-body"},
                ],
            },
        )
    )

    assert names == ("model-def-body", "model-def-query", "model-def-extra")


def test_resolve_gateway_secret_names_for_use_patch_reads_body_only(monkeypatch):
    store = SimpleNamespace(
        get_secret_info=lambda secret_id: SimpleNamespace(secret_name=f"name-{secret_id}")
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: store,
    )

    names = resource_names_mod.resolve_gateway_secret_names_for_use(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/gateway/model-definitions/update",
            method="PATCH",
            workspace="team-a",
            query_params={"secret_id": "query-secret"},
            json_body={"secret_id": "body-secret"},
        )
    )

    assert names == ("name-body-secret",)


def test_resolve_experiment_name_from_run_id_uses_cache(monkeypatch):
    _invalidate_run_lookup_cache("run-cache")
    store = Mock()
    store.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id="exp-1"))
    store.get_experiment.return_value = SimpleNamespace(name="exp-a")
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: store,
    )

    first = resource_names_mod._resolve_experiment_name_from_run_id("run-cache")
    second = resource_names_mod._resolve_experiment_name_from_run_id("run-cache")

    assert first == "exp-a"
    assert second == "exp-a"
    store.get_run.assert_called_once_with("run-cache")


def test_authorize_request_retries_dataset_linking_with_experiment_names(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()

    def _is_allowed(identity, resource, verb, namespace, subresource=None, resource_name=None):
        if resource == RESOURCE_DATASETS:
            return True
        if resource == RESOURCE_EXPERIMENTS and resource_name in {"exp-a", "exp-b"}:
            return True
        return False

    authorizer.is_allowed.side_effect = _is_allowed
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [
            AuthorizationRule("update", resource=RESOURCE_DATASETS),
            AuthorizationRule(
                "update",
                resource=RESOURCE_EXPERIMENTS,
                resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,),
            ),
        ],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(
                name={"dataset-exp-1": "exp-a", "dataset-exp-2": "exp-b"}[experiment_id]
            )
        ),
    )

    with app.test_request_context(
        "/api/2.0/mlflow/datasets/add-to-experiments",
        method="POST",
        data=json.dumps(
            {
                "dataset_id": "dataset-1",
                "experiment_ids": ["dataset-exp-1", "dataset-exp-2"],
            }
        ),
        content_type="application/json",
    ):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer dataset-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/datasets/add-to-experiments",
                method="POST",
                workspace="team-a",
                json_body={
                    "dataset_id": "dataset-1",
                    "experiment_ids": ["dataset-exp-1", "dataset-exp-2"],
                },
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_args_list[0].args[1:] == (
        RESOURCE_DATASETS,
        "update",
        "team-a",
        None,
    )
    assert authorizer.is_allowed.call_args_list[1].args[1:] == (
        RESOURCE_EXPERIMENTS,
        "update",
        "team-a",
        None,
    )
    assert authorizer.is_allowed.call_args_list[2].kwargs == {"resource_name": "exp-a"}
    assert authorizer.is_allowed.call_args_list[3].kwargs == {"resource_name": "exp-b"}


def test_authorize_request_retries_with_resource_name_for_put_requests(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    rule = AuthorizationRule(
        "update",
        resource=RESOURCE_EXPERIMENTS,
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(name="put-exp")
        ),
    )

    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer lookup-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/3.0/mlflow/scorers/online-config",
            method="PUT",
            workspace="team-a",
            json_body={"experiment_id": "put-exp-123"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.kwargs == {"resource_name": "put-exp"}


def test_authorize_request_denies_when_resource_name_lookup_fails(monkeypatch):
    experiment_id = "fail-lookup-exp-123"
    _invalidate_experiment_lookup_cache(experiment_id)
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    rule = AuthorizationRule(
        "get",
        resource=RESOURCE_EXPERIMENTS,
        resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: (_ for _ in ()).throw(
                MlflowException("missing experiment")
            )
        ),
    )

    with pytest.raises(MlflowException, match="Permission denied") as exc:
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer lookup-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/experiments/get",
                method="GET",
                workspace="team-a",
                query_params={"experiment_id": experiment_id},
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    assert authorizer.is_allowed.call_count == 1


def test_authorize_request_prefilters_experiment_ids_after_broad_denial(monkeypatch):
    authorizer = Mock()

    def _is_allowed(identity, resource, verb, namespace, subresource=None, resource_name=None):
        if resource_name is None:
            return False
        return resource_name == "exp-a"

    authorizer.is_allowed.side_effect = _is_allowed
    rule = AuthorizationRule(
        "list",
        resource=RESOURCE_EXPERIMENTS,
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {"1": "exp-a", "2": "exp-b"}[experiment_id],
    )

    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/runs/search",
            method="POST",
            workspace="team-a",
            json_body={"experiment_ids": ["1", "2"], "filter": ""},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert result.request_context.json_body == {"experiment_ids": ["1"], "filter": ""}


def test_apply_request_collection_filter_keeps_experiment_id_sources_separate(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-body"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {
            "body-allowed": "exp-body",
            "query-denied": "exp-denied",
        }[experiment_id],
    )

    updated_request_context, allowed = apply_request_collection_filter(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/runs/search",
            method="POST",
            workspace="team-a",
            query_params={"experiment_ids": ["query-denied"]},
            json_body={"experiment_ids": ["body-allowed"]},
        ),
        COLLECTION_POLICY_REQUEST_EXPERIMENT_IDS,
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert allowed is True
    assert updated_request_context.json_body == {"experiment_ids": ["body-allowed"]}
    assert updated_request_context.query_params == {}


def test_apply_request_collection_filter_denies_single_experiment_id_when_query_source_denied(
    monkeypatch,
):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-body"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {
            "body-allowed": "exp-body",
            "query-denied": "exp-denied",
        }[experiment_id],
    )

    _, allowed = apply_request_collection_filter(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/jobs/search",
            method="POST",
            workspace="team-a",
            query_params={"experiment_id": "query-denied"},
            json_body={"experiment_id": "body-allowed"},
        ),
        COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert allowed is False


def test_authorize_request_prefilters_run_ids_after_broad_denial(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    rule = AuthorizationRule(
        "list",
        resource=RESOURCE_EXPERIMENTS,
        collection_policy=COLLECTION_POLICY_REQUEST_RUN_IDS,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_run_id",
        lambda run_id: {"run-1": "exp-a", "run-2": "exp-b"}[run_id],
    )

    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/ajax-api/2.0/mlflow/metrics/get-history-bulk",
            method="GET",
            workspace="team-a",
            query_params={"run_ids": ["run-1", "run-2"], "metric_key": "loss"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert result.request_context.query_params == {
        "run_ids": ["run-1"],
        "metric_key": "loss",
    }


def test_apply_request_collection_filter_keeps_run_id_sources_separate(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") in {
        "exp-body",
        "exp-query",
    }
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_run_id",
        lambda run_id: {
            "run-body-allowed": "exp-body",
            "run-body-denied": "exp-denied",
            "run-query-allowed": "exp-query",
            "run-query-denied": "exp-denied",
        }[run_id],
    )

    updated_request_context, allowed = apply_request_collection_filter(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/metric-history",
            method="GET",
            workspace="team-a",
            query_params={
                "run_ids": ["run-query-allowed", "run-query-denied"],
                "metric_key": "loss",
            },
            json_body={"run_ids": ["run-body-allowed", "run-body-denied"]},
        ),
        COLLECTION_POLICY_REQUEST_RUN_IDS,
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert allowed is True
    assert updated_request_context.json_body == {"run_ids": ["run-body-allowed"]}
    assert updated_request_context.query_params == {
        "run_ids": ["run-query-allowed"],
        "metric_key": "loss",
    }


def test_authorize_request_denies_when_all_run_ids_are_filtered(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    rule = AuthorizationRule(
        "list",
        resource=RESOURCE_EXPERIMENTS,
        collection_policy=COLLECTION_POLICY_REQUEST_RUN_IDS,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_run_id",
        lambda run_id: {"run-1": "exp-a"}[run_id],
    )

    with app.test_request_context(
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk?run_ids=run-1&metric_key=loss",
        method="GET",
    ):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            _authorize_request(
                AuthorizationRequest(
                    authorization_header="Bearer filter-token",
                    forwarded_access_token=None,
                    remote_user_header_value=None,
                    remote_groups_header_value=None,
                    path="/ajax-api/2.0/mlflow/metrics/get-history-bulk",
                    method="GET",
                    workspace="team-a",
                    query_params={"run_ids": ["run-1"], "metric_key": "loss"},
                ),
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)


def test_authorize_request_allows_single_experiment_collection_after_broad_denial(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    rule = AuthorizationRule(
        "list",
        resource=RESOURCE_EXPERIMENTS,
        collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_ID,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: "exp-a",
    )

    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer filter-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/3.0/mlflow/prompt-optimization/jobs/search",
            method="GET",
            workspace="team-a",
            query_params={"experiment_id": "1"},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.kwargs == {"resource_name": "exp-a"}


def test_authorize_request_retries_with_resource_name_for_start_trace_v3(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    rule = AuthorizationRule(
        "update",
        resource=RESOURCE_EXPERIMENTS,
        resource_name_parsers=(RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,),
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler._find_authorization_rules",
        lambda path, method, **kwargs: [rule],
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(name="trace-v3-exp")
        ),
    )

    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer trace-v3-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/traces",
            method="POST",
            workspace="team-a",
            json_body={
                "trace": {
                    "trace_info": {
                        "trace_location": {
                            "mlflow_experiment": {"experiment_id": "trace-v3-exp-123"}
                        }
                    }
                }
            },
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    first_call = authorizer.is_allowed.call_args_list[0]
    assert first_call.args[1:] == (RESOURCE_EXPERIMENTS, "update", "team-a", None)
    assert first_call.kwargs == {}

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.args[1:] == (RESOURCE_EXPERIMENTS, "update", "team-a", None)
    assert second_call.kwargs == {"resource_name": "trace-v3-exp"}


def test_resolve_resource_names_supports_start_trace_v3_camel_case(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(name="trace-v3-camel-exp")
        ),
    )

    request_context = AuthorizationRequest(
        authorization_header="Bearer trace-v3-token",
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/api/2.0/mlflow/traces",
        method="POST",
        workspace="team-a",
        json_body={
            "trace": {
                "traceInfo": {
                    "traceLocation": {
                        "mlflowExperiment": {"experimentId": "trace-v3-exp-456"}
                    }
                }
            }
        },
    )

    assert resolve_resource_names(
        request_context, (RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,)
    ) == ("trace-v3-camel-exp",)


def test_resolve_resource_names_rejects_start_trace_v3_without_nested_experiment_id():
    request_context = AuthorizationRequest(
        authorization_header="Bearer trace-v3-token",
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/api/2.0/mlflow/traces",
        method="POST",
        workspace="team-a",
        json_body={"trace": {"trace_info": {"trace_location": {"mlflow_experiment": {}}}}},
    )

    with pytest.raises(RuntimeError, match="MLflow experiment ID"):
        resolve_resource_names(
            request_context, (RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,)
        )


def test_compile_rules_raise_for_uncovered_endpoint(monkeypatch):
    import mlflow_kubernetes_plugins.auth.compiler as auth_mod

    monkeypatch.setenv("K8S_AUTH_TEST_SKIP_COMPILE", "1")

    _reset_compiled_rules()

    def _fake_endpoints(resolver):
        def _handler():
            return None

        return [("/api/2.0/mlflow/uncovered", _handler, ["GET"])]

    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.compiler.get_endpoints", _fake_endpoints)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.compiler.mlflow_app.url_map.iter_rules",
        lambda: [],
    )

    with pytest.raises(MlflowException, match="/api/2.0/mlflow/uncovered") as exc:
        auth_mod._compile_authorization_rules()

    assert "/api/2.0/mlflow/uncovered" in str(exc.value)


def test_gateway_proxy_routes_require_verbs():
    get_rule = PATH_AUTHORIZATION_RULES[("/api/2.0/mlflow/gateway-proxy", "GET")]
    post_rule = PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/gateway-proxy", "POST")]
    assert (get_rule.verb, get_rule.resource, get_rule.subresource) == (
        "create",
        RESOURCE_GATEWAY_ENDPOINTS,
        "use",
    )
    assert (post_rule.verb, post_rule.resource, post_rule.subresource) == (
        "create",
        RESOURCE_GATEWAY_ENDPOINTS,
        "use",
    )


def test_gateway_invocation_routes_require_use():
    routes = [
        ("/gateway/<endpoint_name>/mlflow/invocations", "POST"),
        ("/gateway/mlflow/v1/chat/completions", "POST"),
        ("/gateway/openai/v1/chat/completions", "POST"),
        ("/gateway/openai/v1/embeddings", "POST"),
        ("/gateway/openai/v1/responses", "POST"),
        ("/gateway/anthropic/v1/messages", "POST"),
        ("/gateway/gemini/v1beta/models/<endpoint_name>:generateContent", "POST"),
        ("/gateway/gemini/v1beta/models/<endpoint_name>:streamGenerateContent", "POST"),
    ]
    for route in routes:
        rule = PATH_AUTHORIZATION_RULES[route]
        assert (rule.verb, rule.resource, rule.subresource) == (
            "create",
            RESOURCE_GATEWAY_ENDPOINTS,
            "use",
        )


def test_misc_path_authorization_rules_cover_recent_endpoints():
    assert PATH_AUTHORIZATION_RULES[("/version", "GET")].verb is None
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/metrics/get-history-bulk", "GET")].verb
        == "list"
    )
    assert (
        PATH_AUTHORIZATION_RULES[
            ("/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval", "GET")
        ].verb
        == "list"
    )
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "POST")].verb
        == "update"
    )
    assert (
        PATH_AUTHORIZATION_RULES[
            ("/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files", "GET")
        ].verb
        == "get"
    )
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/get-trace-artifact", "GET")].verb == "get"
    )
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/scorers/online-configs", "GET")].verb
        == "get"
    )
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/scorers/online-config", "PUT")].verb
        == "update"
    )
    assert PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/scorer/invoke", "POST")].verb == "update"
    assert (
        PATH_AUTHORIZATION_RULES[
            ("/ajax-api/3.0/mlflow/gateway/supported-providers", "GET")
        ].resource
        == RESOURCE_GATEWAY_MODEL_DEFINITIONS
    )


def test_metric_history_bulk_rules_use_run_id_request_filter():
    request_rule = REQUEST_AUTHORIZATION_RULES[GetMetricHistoryBulkInterval]
    assert request_rule.collection_policy == COLLECTION_POLICY_REQUEST_RUN_IDS

    for path in (
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
    ):
        path_rule = PATH_AUTHORIZATION_RULES[(path, "GET")]
        assert path_rule.collection_policy == COLLECTION_POLICY_REQUEST_RUN_IDS


def test_server_info_endpoints_are_unprotected():
    assert _is_unprotected_path("/server-info")
    assert _is_unprotected_path("/api/3.0/mlflow/server-info")
    assert _is_unprotected_path("/ajax-api/3.0/mlflow/server-info")
    assert _is_unprotected_path("/ajax-api/3.0/mlflow/ui-telemetry")


def test_authorization_cache_does_not_drop_new_entries_during_cleanup():
    cache = _AuthorizationCache(ttl_seconds=0.1)
    key = _AuthorizationCacheKey(
        identity_hash="token",
        namespace="namespace",
        resource="resource",
        subresource=None,
        verb="verb",
    )

    class _InstrumentedLock:
        def __init__(self):
            self.on_release_read = None

        def acquire_read(self):
            return None

        def release_read(self):
            if self.on_release_read:
                callback = self.on_release_read
                self.on_release_read = None
                callback()

        def acquire_write(self):
            return None

        def release_write(self):
            return None

    cache._lock = _InstrumentedLock()  # type: ignore[assignment]
    cache._entries[key] = _CacheEntry(allowed=True, expires_at=time.time() - 10)

    def _insert_new_entry():
        cache._entries[key] = _CacheEntry(allowed=False, expires_at=time.time() + 100)

    cache._lock.on_release_read = _insert_new_entry  # type: ignore[attr-defined]

    # First call observes the expired entry and triggers cleanup
    assert cache.get(key) is None
    # New entry should remain available
    assert cache.get(key) is False


def test_name_lookup_cache_reaps_expired_entries_and_enforces_max_size(monkeypatch):
    now = 1_000.0
    monkeypatch.setattr(resource_names_mod.time, "time", lambda: now)
    cache = _NameLookupCache(ttl_seconds=0.01, max_entries=2)
    cache.set("stale", "exp-stale")
    now += 0.02
    cache.set("first", "exp-first")
    cache.set("second", "exp-second")
    cache.set("third", "exp-third")

    assert cache.get("stale") is None
    assert cache.get("first") is None
    assert cache.get("second") == "exp-second"
    assert cache.get("third") == "exp-third"


def test_authorization_cache_separates_resource_name_entries(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    authorizer = KubernetesAuthorizer(KubernetesAuthConfig(cache_ttl_seconds=60))
    submit_review = Mock(
        side_effect=lambda token, resource, verb, namespace, subresource=None, resource_name=None: (
            resource_name == "exp-a"
        )
    )
    monkeypatch.setattr(authorizer, "_submit_self_subject_access_review", submit_review)

    identity = _RequestIdentity(token="token")
    assert authorizer.is_allowed(identity, RESOURCE_EXPERIMENTS, "get", "team-a") is False
    assert (
        authorizer.is_allowed(
            identity,
            RESOURCE_EXPERIMENTS,
            "get",
            "team-a",
            resource_name="exp-a",
        )
        is True
    )
    assert (
        authorizer.is_allowed(
            identity,
            RESOURCE_EXPERIMENTS,
            "get",
            "team-a",
            resource_name="exp-a",
        )
        is True
    )
    assert submit_review.call_count == 2


def test_authorization_cache_ignores_resource_name_for_unscoped_verbs(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    authorizer = KubernetesAuthorizer(KubernetesAuthConfig(cache_ttl_seconds=60))
    submit_review = Mock(return_value=True)
    monkeypatch.setattr(authorizer, "_submit_self_subject_access_review", submit_review)

    identity = _RequestIdentity(token="token")
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "list", "team-a", resource_name="exp-a"
    )
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "list", "team-a", resource_name="exp-b"
    )
    assert authorizer.is_allowed(
        identity, RESOURCE_DATASETS, "create", "team-a", resource_name="dataset-a"
    )
    assert authorizer.is_allowed(
        identity, RESOURCE_DATASETS, "create", "team-a", resource_name="dataset-b"
    )
    assert authorizer.is_allowed(
        identity,
        RESOURCE_GATEWAY_ENDPOINTS,
        "create",
        "team-a",
        subresource="use",
        resource_name="endpoint-a",
    )

    assert submit_review.call_count == 4
    assert submit_review.call_args_list[0].args == (
        "token",
        "experiments",
        "list",
        "team-a",
        None,
        "exp-a",
    )
    assert submit_review.call_args_list[1].args == (
        "token",
        "experiments",
        "list",
        "team-a",
        None,
        "exp-b",
    )
    assert submit_review.call_args_list[2].args == (
        "token",
        "datasets",
        "create",
        "team-a",
        None,
        None,
    )
    assert submit_review.call_args_list[3].args == (
        "token",
        "gatewayendpoints",
        "create",
        "team-a",
        "use",
        "endpoint-a",
    )


def test_authorization_cache_evicts_oldest_named_entries(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    authorizer = KubernetesAuthorizer(KubernetesAuthConfig(cache_ttl_seconds=60))
    authorizer._cache = _AuthorizationCache(ttl_seconds=60, max_entries=2)
    submit_review = Mock(return_value=True)
    monkeypatch.setattr(authorizer, "_submit_self_subject_access_review", submit_review)

    identity = _RequestIdentity(token="token")
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "get", "team-a", resource_name="exp-a"
    )
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "get", "team-a", resource_name="exp-b"
    )
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "get", "team-a", resource_name="exp-a"
    )
    # FIFO eviction: exp-a is oldest by insertion order, so it gets evicted (not exp-b)
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "get", "team-a", resource_name="exp-c"
    )
    # exp-a was evicted, requires a new API call
    assert authorizer.is_allowed(
        identity, RESOURCE_EXPERIMENTS, "get", "team-a", resource_name="exp-a"
    )

    assert submit_review.call_count == 4


def test_apply_response_cache_updates_refreshes_experiment_name_cache(monkeypatch):
    experiment_id = "cache-exp-123"
    _invalidate_experiment_lookup_cache(experiment_id)
    store = Mock()
    store.get_experiment.return_value = SimpleNamespace(name="old-name")
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: store,
    )

    lookup_request = AuthorizationRequest(
        authorization_header="Bearer token",
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/api/2.0/mlflow/experiments/get",
        method="GET",
        workspace="team-a",
        query_params={"experiment_id": experiment_id},
    )
    assert resolve_resource_names(
        lookup_request, (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,)
    ) == ("old-name",)

    apply_response_cache_updates(
        AuthorizationRequest(
            authorization_header="Bearer token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/experiments/update",
            method="PATCH",
            workspace="team-a",
            json_body={"experiment_id": experiment_id, "new_name": "new-name"},
        ),
        [
            AuthorizationRule(
                "update",
                resource=RESOURCE_EXPERIMENTS,
                resource_name_parsers=(
                    RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
                    RESOURCE_NAME_PARSER_NEW_EXPERIMENT_NAME,
                ),
            )
        ],
        status_code=200,
    )

    store.reset_mock()
    assert resolve_resource_names(
        lookup_request, (RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,)
    ) == ("new-name",)
    store.get_experiment.assert_not_called()
    _invalidate_experiment_lookup_cache(experiment_id)


def test_run_id_cache_uses_stable_experiment_id_after_rename(monkeypatch):
    _invalidate_run_lookup_cache("run-123")
    _invalidate_experiment_lookup_cache("exp-123")
    store = Mock()
    store.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id="exp-123"))
    store.get_experiment.return_value = SimpleNamespace(name="old-name")
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: store,
    )

    assert resource_names_mod._resolve_experiment_name_from_run_id("run-123") == "old-name"
    resource_names_mod.update_experiment_name_cache("exp-123", "new-name")
    store.reset_mock()

    assert resource_names_mod._resolve_experiment_name_from_run_id("run-123") == "new-name"
    store.get_run.assert_not_called()
    store.get_experiment.assert_not_called()


def test_apply_response_collection_filters_filters_experiments():
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"

    filtered, enforceable = apply_response_collection_filters(
        {
            "experiments": [
                {"experiment_id": "1", "name": "exp-a"},
                {"experiment_id": "2", "name": "exp-b"},
            ]
        },
        [
            AuthorizationRule(
                "list",
                resource=RESOURCE_EXPERIMENTS,
                collection_policy=COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
            )
        ],
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert enforceable is True
    assert filtered == {"experiments": [{"experiment_id": "1", "name": "exp-a"}]}


def test_apply_response_collection_filters_not_enforceable_on_unexpected_shape():
    authorizer = Mock()
    authorizer.is_allowed.return_value = False

    _, enforceable = apply_response_collection_filters(
        {"experiments": "not-a-list"},
        [
            AuthorizationRule(
                "list",
                resource=RESOURCE_EXPERIMENTS,
                collection_policy=COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
            )
        ],
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert enforceable is False


def test_apply_response_collection_filters_enforceable_when_key_absent():
    authorizer = Mock()

    _, enforceable = apply_response_collection_filters(
        {"other_key": "value"},
        [
            AuthorizationRule(
                "list",
                resource=RESOURCE_EXPERIMENTS,
                collection_policy=COLLECTION_POLICY_RESPONSE_EXPERIMENTS,
            )
        ],
        authorizer=authorizer,
        identity=_RequestIdentity(token="token"),
        workspace_name="team-a",
    )

    assert enforceable is True


def test_apply_request_collection_filter_trace_locations(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda eid: {"1": "exp-a", "2": "exp-b"}[eid],
    )

    from mlflow_kubernetes_plugins.auth.collection_filters import _filter_request_trace_locations

    request_context = AuthorizationRequest(
        authorization_header=None,
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/api/3.0/mlflow/traces/search",
        method="POST",
        workspace="team-a",
        json_body={
            "locations": [
                {"mlflow_experiment": {"experiment_id": "1"}},
                {"mlflow_experiment": {"experiment_id": "2"}},
            ]
        },
    )

    updated, allowed = _filter_request_trace_locations(
        request_context,
        authorizer,
        _RequestIdentity(token="token"),
        "team-a",
    )

    assert allowed is True
    assert updated.json_body["locations"] == [{"mlflow_experiment": {"experiment_id": "1"}}]


def test_experiment_permissions_are_checked_first(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    config = KubernetesAuthConfig(cache_ttl_seconds=1)
    authorizer = KubernetesAuthorizer(config)

    def _fake_permission(identity, resource, verb, namespace):
        return resource == RESOURCE_EXPERIMENTS and namespace == "team-a"

    authorizer.is_allowed = Mock(side_effect=_fake_permission)  # type: ignore[method-assign]

    identity = _RequestIdentity(token="token")
    accessible = authorizer.accessible_workspaces(identity, ["team-a", "team-b"])

    assert accessible == {"team-a"}
    first_call = authorizer.is_allowed.call_args_list[0][0]
    assert first_call[1] == RESOURCE_EXPERIMENTS


def test_can_access_workspace_iterates_priority_resources(monkeypatch):
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    config = KubernetesAuthConfig(cache_ttl_seconds=1)
    authorizer = KubernetesAuthorizer(config)

    def _fake_permission(identity, resource, verb, namespace):
        return resource == RESOURCE_REGISTERED_MODELS and namespace == "team-a" and verb == "get"

    authorizer.is_allowed = Mock(side_effect=_fake_permission)  # type: ignore[method-assign]

    identity = _RequestIdentity(token="token")
    assert authorizer.can_access_workspace(identity, "team-a", verb="get") is True

    calls = authorizer.is_allowed.call_args_list
    assert calls[0][0][1] == RESOURCE_EXPERIMENTS
    assert calls[1][0][1] == RESOURCE_DATASETS
    assert calls[2][0][1] == RESOURCE_REGISTERED_MODELS

    authorizer.is_allowed.reset_mock()
    assert authorizer.can_access_workspace(identity, "team-b", verb="get") is False

    calls = authorizer.is_allowed.call_args_list
    assert len(calls) == 6
    assert [c[0][1] for c in calls] == [
        RESOURCE_EXPERIMENTS,
        RESOURCE_DATASETS,
        RESOURCE_REGISTERED_MODELS,
        RESOURCE_GATEWAY_SECRETS,
        RESOURCE_GATEWAY_ENDPOINTS,
        RESOURCE_GATEWAY_MODEL_DEFINITIONS,
    ]


def test_subject_access_review_authorizer_close_is_idempotent(monkeypatch):
    fake_client = Mock()
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._create_api_client_for_subject_access_reviews",
        lambda: fake_client,
    )

    authorizer = KubernetesAuthorizer(
        KubernetesAuthConfig(authorization_mode=AuthorizationMode.SUBJECT_ACCESS_REVIEW)
    )

    authorizer.close()
    authorizer.close()

    fake_client.close.assert_called_once()


def test_normalize_rules_single_rule():
    rule = AuthorizationRule("get", resource=RESOURCE_EXPERIMENTS)
    assert _normalize_rules(rule) == [rule]


def test_normalize_rules_tuple_of_rules():
    r1 = AuthorizationRule("update", resource=RESOURCE_DATASETS)
    r2 = AuthorizationRule("update", resource=RESOURCE_EXPERIMENTS)
    assert _normalize_rules((r1, r2)) == [r1, r2]


def test_start_trace_v3_uses_experiment_resource_name_parser():
    value = REQUEST_AUTHORIZATION_RULES[StartTraceV3]
    assert isinstance(value, AuthorizationRule)
    assert value.resource_name_parsers == (RESOURCE_NAME_PARSER_TRACE_V3_EXPERIMENT_ID_TO_NAME,)


def test_dataset_operations_use_datasets_resource():
    dataset_ops = {
        CreateDataset: ("create", RESOURCE_DATASETS),
        DeleteDataset: ("delete", RESOURCE_DATASETS),
        DeleteDatasetRecords: ("update", RESOURCE_DATASETS),
        SetDatasetTags: ("update", RESOURCE_DATASETS),
        UpsertDatasetRecords: ("update", RESOURCE_DATASETS),
        GetDataset: ("get", RESOURCE_DATASETS),
    }
    for msg_type, (expected_verb, expected_resource) in dataset_ops.items():
        value = REQUEST_AUTHORIZATION_RULES[msg_type]
        rule = value if isinstance(value, AuthorizationRule) else value[0]
        assert (rule.verb, rule.resource) == (expected_verb, expected_resource), msg_type.__name__


def test_dataset_request_rules_use_resource_name_parsers():
    assert REQUEST_AUTHORIZATION_RULES[CreateDataset].resource_name_parsers == ()
    assert REQUEST_AUTHORIZATION_RULES[GetDataset].resource_name_parsers == (
        RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,
    )
    add_dataset_rules = REQUEST_AUTHORIZATION_RULES[AddDatasetToExperiments]
    assert isinstance(add_dataset_rules, tuple)
    assert add_dataset_rules[0].resource_name_parsers == (RESOURCE_NAME_PARSER_DATASET_ID_TO_NAME,)


def test_search_datasets_effective_compiled_rule_uses_datasets_resource():
    """PATH_AUTHORIZATION_RULES overrides handler-derived rules at compile time.

    Verify the effective rule for the search-datasets endpoints resolves to
    datasets/list (not experiments/list) after compilation.
    """
    for path in (
        "/api/2.0/mlflow/experiments/search-datasets",
        "/ajax-api/2.0/mlflow/experiments/search-datasets",
    ):
        rules = _find_authorization_rules(path, "POST")
        assert rules is not None, f"No rule found for {path}"
        assert len(rules) == 1
        assert (rules[0].verb, rules[0].resource) == ("list", RESOURCE_DATASETS), path


def test_dataset_experiment_linking_requires_both_resources():
    for msg_type in (AddDatasetToExperiments, RemoveDatasetFromExperiments):
        value = REQUEST_AUTHORIZATION_RULES[msg_type]
        assert isinstance(value, tuple), f"{msg_type.__name__} should be a tuple"
        rules = list(value)
        assert len(rules) == 2, f"{msg_type.__name__} should have 2 rules"
        resources = {r.resource for r in rules}
        assert resources == {RESOURCE_DATASETS, RESOURCE_EXPERIMENTS}, msg_type.__name__
        assert all(r.verb == "update" for r in rules), msg_type.__name__
        experiment_rule = next(r for r in rules if r.resource == RESOURCE_EXPERIMENTS)
        assert experiment_rule.resource_name_parsers == (
            RESOURCE_NAME_PARSER_EXPERIMENT_IDS_TO_NAMES,
        )


def test_jobs_search_effective_compiled_rule_uses_experiment_filter():
    for path in ("/ajax-api/3.0/jobs/search", "/ajax-api/3.0/jobs/search/"):
        rules = _find_authorization_rules(path, "POST")
        assert rules is not None, f"No rule found for {path}"
        assert len(rules) == 1
        assert rules[0].collection_policy == COLLECTION_POLICY_REQUEST_EXPERIMENT_ID


def test_generic_job_routes_do_not_attach_experiment_name_parsers():
    routes = [
        ("/ajax-api/3.0/jobs", "POST"),
        ("/ajax-api/3.0/jobs/123", "GET"),
        ("/ajax-api/3.0/jobs/cancel/123", "PATCH"),
        ("/ajax-api/3.0/jobs/search", "POST"),
    ]
    for path, method in routes:
        rules = _find_authorization_rules(path, method)
        assert rules is not None, f"No rule found for {method} {path}"
        assert len(rules) == 1
        assert rules[0].resource_name_parsers == ()


def test_prompt_optimization_jobs_use_experiments_resource():
    job_ops = {
        CreatePromptOptimizationJob: "update",
        GetPromptOptimizationJob: "get",
        SearchPromptOptimizationJobs: "list",
        CancelPromptOptimizationJob: "update",
        DeletePromptOptimizationJob: "update",
    }
    for msg_type, expected_verb in job_ops.items():
        rule = REQUEST_AUTHORIZATION_RULES[msg_type]
        assert isinstance(rule, AuthorizationRule), msg_type.__name__
        assert (rule.verb, rule.resource) == (expected_verb, RESOURCE_EXPERIMENTS), (
            msg_type.__name__
        )


def test_registered_model_request_rules_use_resource_name_parsers():
    assert REQUEST_AUTHORIZATION_RULES[CreateRegisteredModel].resource_name_parsers == ()
    assert REQUEST_AUTHORIZATION_RULES[GetRegisteredModel].resource_name_parsers == (
        RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[RenameRegisteredModel].resource_name_parsers == (
        RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
        RESOURCE_NAME_PARSER_NEW_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[CreateModelVersion].resource_name_parsers == (
        RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[GetModelVersion].resource_name_parsers == (
        RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[DeleteWebhook].resource_name_parsers == (
        RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[UpdateWebhook].resource_name_parsers == (
        RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[GetWebhook].resource_name_parsers == (
        RESOURCE_NAME_PARSER_WEBHOOK_ID_TO_REGISTERED_MODEL_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[GetLatestVersions].resource_name_parsers == (
        RESOURCE_NAME_PARSER_REGISTERED_MODEL_NAME,
    )


def test_server_info_endpoint_is_unprotected():
    rule = PATH_AUTHORIZATION_RULES[("/server-info", "GET")]
    assert rule.verb is None
    assert rule.resource is None


def test_assessment_delete_path_rules():
    for prefix in ("/api/3.0", "/ajax-api/3.0"):
        path = f"{prefix}/mlflow/traces/<trace_id>/assessments/<assessment_id>"
        rule = PATH_AUTHORIZATION_RULES[(path, "DELETE")]
        assert (rule.verb, rule.resource) == ("update", RESOURCE_EXPERIMENTS)


def test_demo_generate_requires_multiple_resources():
    value = PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/demo/generate", "POST")]
    assert isinstance(value, tuple)
    resources = {r.resource for r in value}
    assert resources == {RESOURCE_EXPERIMENTS, RESOURCE_DATASETS, RESOURCE_REGISTERED_MODELS}
    assert all(r.verb == "create" for r in value)


def test_demo_delete_requires_multiple_resources():
    value = PATH_AUTHORIZATION_RULES[("/ajax-api/3.0/mlflow/demo/delete", "POST")]
    assert isinstance(value, tuple)
    resources = {r.resource for r in value}
    assert resources == {RESOURCE_EXPERIMENTS, RESOURCE_DATASETS, RESOURCE_REGISTERED_MODELS}


def test_assistant_endpoints_use_assistants_resource():
    assistant_routes = [
        (("/ajax-api/3.0/mlflow/assistant/message", "POST"), "create"),
        (("/ajax-api/3.0/mlflow/assistant/sessions/<session_id>/stream", "GET"), "get"),
        (("/ajax-api/3.0/mlflow/assistant/status", "GET"), "get"),
        (("/ajax-api/3.0/mlflow/assistant/sessions/<session_id>", "PATCH"), "update"),
        (("/ajax-api/3.0/mlflow/assistant/providers/<provider>/health", "GET"), "get"),
        (("/ajax-api/3.0/mlflow/assistant/config", "GET"), "get"),
        (("/ajax-api/3.0/mlflow/assistant/config", "PUT"), "update"),
        (("/ajax-api/3.0/mlflow/assistant/skills/install", "POST"), "update"),
    ]
    for route, expected_verb in assistant_routes:
        rule = PATH_AUTHORIZATION_RULES[route]
        assert isinstance(rule, AuthorizationRule), route
        assert (rule.verb, rule.resource) == (expected_verb, RESOURCE_ASSISTANTS), route


def test_gateway_request_rules_use_resource_name_parsers():
    assert REQUEST_AUTHORIZATION_RULES[CreateGatewaySecret].resource_name_parsers == ()
    assert REQUEST_AUTHORIZATION_RULES[GetGatewaySecretInfo].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[UpdateGatewaySecret].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_SECRET_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[CreateGatewayEndpoint].resource_name_parsers == ()
    assert REQUEST_AUTHORIZATION_RULES[GetGatewayEndpoint].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[CreateGatewayModelDefinition].resource_name_parsers == ()
    assert REQUEST_AUTHORIZATION_RULES[GetGatewayModelDefinition].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_MODEL_DEFINITION_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[AttachModelToGatewayEndpoint].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[DetachModelFromGatewayEndpoint].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[CreateGatewayEndpointBinding].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[DeleteGatewayEndpointBinding].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[ListGatewayEndpointBindings].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[SetGatewayEndpointTag].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )
    assert REQUEST_AUTHORIZATION_RULES[DeleteGatewayEndpointTag].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    )


def test_gateway_proxy_post_routes_use_endpoint_name_parser():
    for route in (
        ("/api/2.0/mlflow/gateway-proxy", "POST"),
        ("/ajax-api/2.0/mlflow/gateway-proxy", "POST"),
        ("/gateway/<endpoint_name>/mlflow/invocations", "POST"),
        ("/gateway/mlflow/v1/chat/completions", "POST"),
        ("/gateway/openai/v1/chat/completions", "POST"),
        ("/gateway/openai/v1/embeddings", "POST"),
        ("/gateway/openai/v1/responses", "POST"),
        ("/gateway/anthropic/v1/messages", "POST"),
        ("/gateway/gemini/v1beta/models/<endpoint_name>:generateContent", "POST"),
        ("/gateway/gemini/v1beta/models/<endpoint_name>:streamGenerateContent", "POST"),
    ):
        rule = PATH_AUTHORIZATION_RULES[route]
        assert rule.resource_name_parsers == (RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,)


def test_resolve_resource_names_reads_gateway_endpoint_name_from_model_field():
    names = resolve_resource_names(
        AuthorizationRequest(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/gateway/openai/v1/chat/completions",
            method="POST",
            workspace="team-a",
            json_body={"model": "endpoint-a"},
        ),
        (RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
    )

    assert names == ("endpoint-a",)


def test_resolve_resource_names_rejects_conflicting_gateway_path_and_model():
    with pytest.raises(RuntimeError, match="Conflicting endpoint identifiers"):
        resolve_resource_names(
            AuthorizationRequest(
                authorization_header=None,
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/gateway-proxy",
                method="POST",
                workspace="team-a",
                path_params={"gateway_path": "gateway/private-endpoint/invocations"},
                json_body={"model": "public-endpoint"},
            ),
            (RESOURCE_NAME_PARSER_GATEWAY_PROXY_ENDPOINT_NAME,),
        )
