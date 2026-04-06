"""Tests for Kubernetes auth behavior with Flask and workspace helpers.

This suite exercises request overrides, authorization rules, and caching logic.
"""

import base64
import json
import time
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flask import Flask, g, request
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import (
    AddDatasetToExperiments,
    CancelPromptOptimizationJob,
    CreateDataset,
    CreatePromptOptimizationJob,
    CreateRun,
    CreateWorkspace,
    DeleteDataset,
    DeleteDatasetRecords,
    DeletePromptOptimizationJob,
    DeleteWorkspace,
    GetDataset,
    GetPromptOptimizationJob,
    GetWorkspace,
    ListWorkspaces,
    RemoveDatasetFromExperiments,
    SearchPromptOptimizationJobs,
    SetDatasetTags,
    UpdateWorkspace,
    UpsertDatasetRecords,
)
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow.utils import workspace_context
from mlflow_kubernetes_plugins.auth.authorizer import (
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
    _AuthorizationCache,
    _CacheEntry,
)
from mlflow_kubernetes_plugins.auth.compiler import (
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
from mlflow_kubernetes_plugins.auth.middleware import _override_run_user
from mlflow_kubernetes_plugins.auth.request_context import AuthorizationRequest
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
    assert _canonicalize_path(raw_path=api_path) == api_path
    assert _canonicalize_path(raw_path=health_path) == "/health"
    assert _canonicalize_path(raw_path=metrics_path) == "/metrics"
    assert _canonicalize_path(raw_path=version_path) == "/version"


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


def test_override_run_user_with_json_request():
    app = Flask(__name__)

    @app.route("/test", methods=["POST"])
    def test_endpoint():
        # Get the modified JSON data
        data = request.get_json(silent=True)
        # Also test with silent=False to ensure both cached values work
        data2 = request.get_json(silent=False)
        assert data == data2
        return {"received": data, "user_id": data.get("user_id")}

    with app.test_request_context(
        "/test",
        method="POST",
        data=json.dumps({"experiment_id": "123"}),
        content_type="application/json",
    ):
        # Simulate the auth handler modifying the request
        _override_run_user("test-user")

        # Verify the request was modified correctly
        modified_data = request.get_json(silent=True)
        assert modified_data["experiment_id"] == "123"
        assert modified_data["user_id"] == "test-user"

        # Test that both silent=True and silent=False work
        data_silent_false = request.get_json(silent=False)
        assert data_silent_false == modified_data

        # Verify the raw data was updated
        request.environ["wsgi.input"].seek(0)
        raw_data = request.environ["wsgi.input"].read()
        parsed_raw = json.loads(raw_data)
        assert parsed_raw["user_id"] == "test-user"


def test_override_run_user_with_empty_request():
    app = Flask(__name__)

    with app.test_request_context(
        "/test",
        method="POST",
        data="{}",
        content_type="application/json",
    ):
        _override_run_user("test-user")

        modified_data = request.get_json(silent=True)
        assert modified_data == {"user_id": "test-user"}


def test_override_run_user_with_non_json_request():
    app = Flask(__name__)

    with app.test_request_context(
        "/test",
        method="POST",
        data="not json data",
        content_type="text/plain",
    ):
        original_data = request.data
        _override_run_user("test-user")

        # Data should not be modified for non-JSON requests
        assert request.data == original_data


def test_override_run_user_preserves_other_fields():
    app = Flask(__name__)

    original_payload = {
        "experiment_id": "exp123",
        "tags": [{"key": "tag1", "value": "val1"}],
        "nested": {"field": "value"},
        "user_id": "original-user",  # This should be overwritten
    }

    with app.test_request_context(
        "/test",
        method="POST",
        data=json.dumps(original_payload),
        content_type="application/json",
    ):
        _override_run_user("new-user")

        modified_data = request.get_json(silent=True)
        assert modified_data["user_id"] == "new-user"
        assert modified_data["experiment_id"] == "exp123"
        assert modified_data["tags"] == [{"key": "tag1", "value": "val1"}]
        assert modified_data["nested"] == {"field": "value"}


def test_flask_create_run_request_processing(monkeypatch):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    # Create a minimal Flask app with auth
    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        # This simulates the MLflow handler
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    @app.before_request
    def _set_rbac_context():
        g._workspace_set = True
        workspace_context.set_server_request_workspace("default")

    @app.teardown_request
    def _reset_rbac_context(_response):
        if getattr(g, "_workspace_set", False):
            workspace_context.clear_server_request_workspace()
            delattr(g, "_workspace_set")

    # Set up the app with Kubernetes auth
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
    ):
        create_app(app)

        # Test the request
        client = app.test_client()

        with patch(
            "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"
        ):
            response = client.post(
                "/api/2.0/mlflow/runs/create",
                json={"experiment_id": "0"},
                headers={"Authorization": "Bearer test-token"},
            )

    # The response should have the overridden user
    assert response.status_code == 200
    assert response.json["run"]["info"]["user_id"] == "k8s-user"


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
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    create_app(app)
    return app.test_client()


def test_list_workspaces_without_context(monkeypatch):
    client = _build_workspace_app(monkeypatch)

    with patch(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.get(
            "/api/3.0/mlflow/workspaces",
            headers={"Authorization": "Bearer list-token"},
        )

    assert response.status_code == 200
    assert response.json["workspaces"] == [{"name": "team-a"}]


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
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/3.0/mlflow/workspaces", methods=["POST"])
    def create_workspace_endpoint():
        payload = request.get_json()
        return {"workspace": {"name": payload.get("name")}}, 201

    create_app(app)
    client = app.test_client()

    with patch(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-new"},
            headers={"Authorization": "Bearer create-token"},
        )

    assert response.status_code == 403
    assert "Workspace create" in response.json["message"]
    mock_is_allowed.assert_not_called()


def _build_flask_auth_app(monkeypatch, *, is_allowed=True):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    @app.before_request
    def _set_rbac_context():
        g._workspace_set = True
        workspace_context.set_server_request_workspace("default")

    @app.teardown_request
    def _reset_rbac_context(_response):
        if getattr(g, "_workspace_set", False):
            workspace_context.clear_server_request_workspace()
            delattr(g, "_workspace_set")

    mock_is_allowed = Mock(return_value=is_allowed)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer.KubernetesAuthorizer.is_allowed", mock_is_allowed
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration", lambda: None
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    create_app(app)
    return app.test_client(), mock_is_allowed


def test_missing_authorization_header_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post("/api/2.0/mlflow/runs/create", json={"experiment_id": "0"})
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json["message"]
    )
    mock_is_allowed.assert_not_called()


def test_invalid_bearer_scheme_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post(
        "/api/2.0/mlflow/runs/create",
        json={"experiment_id": "0"},
        headers={"Authorization": "Token bad-token"},
    )
    assert response.status_code == 401
    assert "Bearer" in response.json["message"]
    mock_is_allowed.assert_not_called()


def test_forwarded_access_token_header_allows_request(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={"X-Forwarded-Access-Token": "test-token"},
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
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 403
    assert "Permission denied" in response.json["message"]
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_is_allowed.call_args[0]
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


def test_server_info_endpoints_are_unprotected():
    assert _is_unprotected_path("/server-info")
    assert _is_unprotected_path("/api/3.0/mlflow/server-info")
    assert _is_unprotected_path("/ajax-api/3.0/mlflow/server-info")
    assert _is_unprotected_path("/ajax-api/3.0/mlflow/ui-telemetry")


def test_flask_script_name_prefix_is_stripped_before_rule_matching(monkeypatch):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    app = Flask(__name__)

    @app.route("/mlflow", methods=["GET"])
    def prefixed_root():
        return {"status": "ok"}

    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
        lambda: None,
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    create_app(app)
    client = app.test_client()

    response = client.get(
        "/mlflow",
        environ_overrides={"SCRIPT_NAME": "/mlflow/"},
    )

    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_authorization_cache_does_not_drop_new_entries_during_cleanup():
    cache = _AuthorizationCache(ttl_seconds=0.1)
    key = ("token", "namespace", "resource", "verb")

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
