"""Integration tests for FastAPI authorization with the K8s workspace provider.

These tests ensure OTEL and job APIs enforce workspace-aware authentication.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.testclient import TestClient
from flask import Flask
from flask import request as flask_request
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils import workspace_context
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME
from mlflow_kubernetes_plugins.auth.authorizer import (
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
)
from mlflow_kubernetes_plugins.auth.compiler import _validate_fastapi_route_authorization
from mlflow_kubernetes_plugins.auth.constants import (
    DEFAULT_REMOTE_GROUPS_HEADER,
    DEFAULT_REMOTE_GROUPS_SEPARATOR,
    DEFAULT_REMOTE_USER_HEADER,
)
from mlflow_kubernetes_plugins.auth.middleware import KubernetesAuthMiddleware
from mlflow_kubernetes_plugins.auth.rules import (
    PATH_AUTHORIZATION_RULES,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


@pytest.fixture(autouse=True)
def _compile_rules(compile_auth_rules):
    """Ensure authorization rules are populated before each test."""
    compile_auth_rules([])


def test_otel_endpoints_in_auth_rules():
    # Check that OTEL endpoints are registered
    assert (OTLP_TRACES_PATH, "POST") in PATH_AUTHORIZATION_RULES

    # Verify they have the correct authorization rule
    rule = PATH_AUTHORIZATION_RULES[(OTLP_TRACES_PATH, "POST")]
    assert (rule.verb, rule.resource) == ("update", "experiments")


def test_trace_get_endpoints_in_auth_rules():
    paths = [
        "/api/3.0/mlflow/traces/get",
        "/ajax-api/3.0/mlflow/traces/get",
    ]

    for path in paths:
        rule = PATH_AUTHORIZATION_RULES[(path, "GET")]
        assert (rule.verb, rule.resource) == ("get", "experiments")


def test_job_api_endpoints_in_auth_rules():
    cases = [
        ("/ajax-api/3.0/jobs", "POST", "update"),
        ("/ajax-api/3.0/jobs/<job_id>", "GET", "get"),
        ("/ajax-api/3.0/jobs/cancel/<job_id>", "PATCH", "update"),
        ("/ajax-api/3.0/jobs/search", "POST", "list"),
    ]

    for path, method, verb in cases:
        rule = PATH_AUTHORIZATION_RULES[(path, method)]
        assert (rule.verb, rule.resource) == (verb, "experiments")


@pytest.fixture
def mock_authorizer():
    """Create a mock KubernetesAuthorizer."""
    authorizer = Mock(spec=KubernetesAuthorizer)
    authorizer.is_allowed.return_value = True  # Default to allowed
    return authorizer


@pytest.fixture
def mock_config():
    """Create a mock KubernetesAuthConfig."""
    config = Mock(spec=KubernetesAuthConfig)
    config.username_claim = "sub"
    config.cache_ttl_seconds = 300.0
    config.authorization_mode = AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW
    config.user_header = DEFAULT_REMOTE_USER_HEADER
    config.groups_header = DEFAULT_REMOTE_GROUPS_HEADER
    config.groups_separator = DEFAULT_REMOTE_GROUPS_SEPARATOR
    return config


@pytest.fixture
def fastapi_app_with_k8s_auth(mock_authorizer, mock_config):
    """Create a FastAPI app with Kubernetes auth middleware."""
    app = FastAPI()

    class _WorkspaceContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            workspace_header = request.headers.get(WORKSPACE_HEADER_NAME)
            if not workspace_header:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"Missing {WORKSPACE_HEADER_NAME} header"}},
                )
            if request.url.path == OTLP_TRACES_PATH and not request.headers.get(
                "X-MLflow-Experiment-Id"
            ):
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "Missing X-MLflow-Experiment-Id header"}},
                )
            workspace_context.set_server_request_workspace(workspace_header)
            try:
                return await call_next(request)
            finally:
                workspace_context.clear_server_request_workspace()

    # Add a mock OTEL endpoint
    @app.post(OTLP_TRACES_PATH)
    async def mock_otel_endpoint(request: Request):
        workspace_name = workspace_context.get_request_workspace()
        return {
            "status": "ok",
            "workspace": workspace_name,
        }

    # Add a mock Job API endpoint (should be processed by K8s middleware)
    @app.get("/ajax-api/3.0/jobs/123")
    async def mock_job_endpoint(request: Request):
        workspace_name = workspace_context.get_request_workspace()
        return {
            "status": "job_endpoint",
            "workspace": workspace_name,
        }

    # Add a mock Flask-style endpoint (should NOT be processed by K8s middleware)
    flask_app = Flask(__name__)

    @flask_app.post("/api/2.0/mlflow/experiments/create")
    def mock_flask_endpoint():
        # This should be called without auth processing
        return {"status": "flask_endpoint"}

    app.mount("/", WSGIMiddleware(flask_app))

    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )
    app.add_middleware(_WorkspaceContextMiddleware)

    return app


def test_otel_endpoint_requires_auth(fastapi_app_with_k8s_auth):
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )


def test_otel_endpoint_requires_bearer_token(fastapi_app_with_k8s_auth):
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Basic invalid",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 401
    assert "Bearer" in response.json()["error"]["message"]


def test_otel_endpoint_requires_experiment_id(fastapi_app_with_k8s_auth):
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 400
    assert "X-MLflow-Experiment-Id" in response.json()["error"]["message"]


def test_otel_endpoint_requires_workspace_header(fastapi_app_with_k8s_auth):
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            "X-MLflow-Experiment-Id": "exp123",
        },
    )

    assert response.status_code == 400
    assert WORKSPACE_HEADER_NAME in response.json()["error"]["message"]


def test_otel_endpoint_with_valid_auth(fastapi_app_with_k8s_auth, mock_authorizer):
    client = TestClient(fastapi_app_with_k8s_auth)

    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="test-user"):
        response = client.post(
            OTLP_TRACES_PATH,
            headers={
                "Authorization": "Bearer valid-token",
                "X-MLflow-Experiment-Id": "exp123",
                WORKSPACE_HEADER_NAME: "team-a",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["workspace"] == "team-a"

    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "valid-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")
    assert subresource is None


def test_otel_endpoint_with_root_path_requires_auth(mock_authorizer, mock_config) -> None:
    app = FastAPI(root_path="/mlflow")
    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )

    class _WorkspaceContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            workspace_header = request.headers.get(WORKSPACE_HEADER_NAME)
            if not workspace_header:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"Missing {WORKSPACE_HEADER_NAME} header"}},
                )
            workspace_context.set_server_request_workspace(workspace_header)
            try:
                return await call_next(request)
            finally:
                workspace_context.clear_server_request_workspace()

    # Ensure workspace context is set before Kubernetes auth middleware runs.
    app.add_middleware(_WorkspaceContextMiddleware)

    @app.post(OTLP_TRACES_PATH)
    async def mock_otel_endpoint(_request: Request):
        return {"status": "ok"}

    client = TestClient(app)

    response = client.post(
        f"/mlflow{OTLP_TRACES_PATH}",
        headers={
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )


def test_job_api_endpoints_with_root_path_require_auth(mock_authorizer, mock_config) -> None:
    app = FastAPI(root_path="/mlflow")
    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )

    class _WorkspaceContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            workspace_header = request.headers.get(WORKSPACE_HEADER_NAME)
            if not workspace_header:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"Missing {WORKSPACE_HEADER_NAME} header"}},
                )
            workspace_context.set_server_request_workspace(workspace_header)
            try:
                return await call_next(request)
            finally:
                workspace_context.clear_server_request_workspace()

    app.add_middleware(_WorkspaceContextMiddleware)

    @app.get("/ajax-api/3.0/jobs/123")
    async def mock_job_endpoint(_request: Request):
        return {"status": "job_endpoint"}

    client = TestClient(app)

    response = client.get(
        "/mlflow/ajax-api/3.0/jobs/123",
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )


def test_otel_endpoint_accepts_forwarded_access_token(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "X-Forwarded-Access-Token": "forwarded-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")
    assert subresource is None


def test_otel_endpoint_prefers_forwarded_token_on_invalid_authorization(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Token invalid",
            "X-Forwarded-Access-Token": "forwarded-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")
    assert subresource is None


def test_otel_endpoint_permission_denied(fastapi_app_with_k8s_auth, mock_authorizer):
    mock_authorizer.is_allowed.return_value = False
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 403
    assert "Permission denied" in response.json()["error"]["message"]


def test_flask_endpoints_require_fastapi_auth(fastapi_app_with_k8s_auth, mock_authorizer) -> None:
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        "/api/2.0/mlflow/experiments/create",
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )

    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )
    mock_authorizer.is_allowed.assert_not_called()


def test_job_api_endpoints_require_auth(fastapi_app_with_k8s_auth, mock_authorizer):
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )

    mock_authorizer.reset_mock()
    with patch("mlflow_kubernetes_plugins.auth.core._parse_jwt_subject", return_value="test-user"):
        response = client.get(
            "/ajax-api/3.0/jobs/123",
            headers={
                "Authorization": "Bearer valid-token",
                WORKSPACE_HEADER_NAME: "team-a",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "job_endpoint"
    assert response.json()["workspace"] == "team-a"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "valid-token"
    assert (resource, verb, namespace) == ("experiments", "get", "team-a")
    assert subresource is None


def test_job_api_endpoints_accept_forwarded_access_token(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "X-Forwarded-Access-Token": "forwarded-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "job_endpoint"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "get", "team-a")
    assert subresource is None


def test_job_api_endpoints_prefer_forwarded_token_on_invalid_authorization(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "Authorization": "Token invalid",
            "X-Forwarded-Access-Token": "forwarded-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace, subresource = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "get", "team-a")
    assert subresource is None


def test_graphql_flask_authorizes_when_fastapi_defers(
    mock_authorizer, mock_config, monkeypatch
) -> None:
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    flask_app = Flask(__name__)

    @flask_app.post("/graphql")
    def graphql_endpoint():
        payload = flask_request.get_json(silent=True) or {}
        return {"query": payload.get("query")}

    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")
    fake_k8s_config = SimpleNamespace(
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
        ) as flask_is_allowed,
        patch(
            "mlflow_kubernetes_plugins.auth.authorizer._load_kubernetes_configuration",
            return_value=fake_k8s_config,
        ),
    ):
        create_app(flask_app)

        fastapi_app = FastAPI()
        fastapi_app.mount("/", WSGIMiddleware(flask_app))

        class _WorkspaceContextMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                workspace_header = request.headers.get(WORKSPACE_HEADER_NAME)
                if not workspace_header:
                    return JSONResponse(
                        status_code=400,
                        content={"error": {"message": f"Missing {WORKSPACE_HEADER_NAME} header"}},
                    )
                workspace_context.set_server_request_workspace(workspace_header)
                try:
                    return await call_next(request)
                finally:
                    workspace_context.clear_server_request_workspace()

        fastapi_app.add_middleware(
            KubernetesAuthMiddleware,
            authorizer=mock_authorizer,
            config_values=mock_config,
        )
        fastapi_app.add_middleware(_WorkspaceContextMiddleware)

        client = TestClient(fastapi_app)
        query = '{ mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } } }'
        with patch(
            "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
            return_value="test-user",
        ):
            response = client.post(
                "/graphql",
                headers={
                    "Authorization": "Bearer valid-token",
                    WORKSPACE_HEADER_NAME: "team-a",
                },
                json={"query": query},
            )

    assert response.status_code == 200
    assert response.json()["query"] == query
    # FastAPI middleware should NOT have called its authorizer
    mock_authorizer.is_allowed.assert_not_called()
    # Flask's before_request handler SHOULD have called the authorizer
    flask_is_allowed.assert_called_once()


def test_job_api_missing_workspace_context_returns_error(
    mock_authorizer, mock_config, monkeypatch
) -> None:
    app = FastAPI()
    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.resolve_workspace_from_header",
        lambda _header: None,
    )

    @app.get("/ajax-api/3.0/jobs/123")
    async def job_endpoint():
        return {"status": "job_endpoint"}

    client = TestClient(app)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 500
    assert "Workspace context" in response.json()["error"]["message"]


def test_fastapi_route_validation_passes_for_known_routes(fastapi_app_with_k8s_auth):
    _validate_fastapi_route_authorization(fastapi_app_with_k8s_auth)


def test_fastapi_route_validation_fails_for_missing_rule():
    app = FastAPI()

    @app.get("/ajax-api/3.0/jobs-new")
    async def _missing():
        return {}

    with pytest.raises(MlflowException, match="FastAPI endpoints"):
        _validate_fastapi_route_authorization(app)


def test_create_app_wraps_flask_with_fastapi_under_uvicorn(monkeypatch):
    from mlflow_kubernetes_plugins.auth.middleware import create_app

    flask_app = Flask(__name__)
    fastapi_app = Mock()
    validate_fastapi_routes = Mock()
    validate_graphql = Mock()
    authorizer = Mock(spec=KubernetesAuthorizer)
    config_values = Mock(spec=KubernetesAuthConfig)

    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.middleware._compile_authorization_rules", lambda: None)
    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.middleware.create_fastapi_app", lambda app: fastapi_app)
    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.middleware._validate_fastapi_route_authorization", validate_fastapi_routes)
    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.middleware._validate_graphql_field_authorization", validate_graphql)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.KubernetesAuthConfig.from_env",
        lambda: config_values,
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.middleware.KubernetesAuthorizer",
        lambda config_values: authorizer,
    )

    with patch("mlflow_kubernetes_plugins.auth.middleware._MLFLOW_SGI_NAME.get", return_value="uvicorn"):
        result = create_app(flask_app)

    assert result is fastapi_app
    fastapi_app.add_middleware.assert_called_once()
    middleware_args, middleware_kwargs = fastapi_app.add_middleware.call_args
    assert middleware_args[0] is KubernetesAuthMiddleware
    assert middleware_kwargs["authorizer"] is authorizer
    assert middleware_kwargs["config_values"] is config_values
    validate_fastapi_routes.assert_called_once_with(fastapi_app)
    validate_graphql.assert_called_once_with()


# Example usage documentation
"""
Example: Using OTEL endpoints with Kubernetes authorization

1. Start MLflow server with the Kubernetes workspace provider (via workspace store URI) and auth:
   ```bash
   mlflow server \\
     --app-name kubernetes-auth \\
     --enable-workspaces \\
     --workspace-store-uri "kubernetes://?label_selector=mlflow-enabled%3Dtrue&default_workspace=team-a"
   ```

2. Configure OTEL exporter to send traces with authentication:
   ```python
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

   # Create exporter with Bearer token
   exporter = OTLPSpanExporter(
       endpoint="http://mlflow-server:5000/v1/traces",
      headers={
          "Authorization": "Bearer <k8s-service-account-token>",
          "X-MLflow-Experiment-Id": "experiment-123",
          "X-MLFLOW-WORKSPACE": "team-a",
      }
   )
   ```

3. To send to another workspace, update the header:
   ```python
   exporter = OTLPSpanExporter(
        endpoint="http://mlflow-server:5000/v1/traces",
      headers={
          "Authorization": "Bearer <k8s-service-account-token>",
          "X-MLflow-Experiment-Id": "experiment-123",
          "X-MLFLOW-WORKSPACE": "team-b",
      }
   )
   ```

4. Required Kubernetes RBAC permissions:
   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: mlflow-trace-ingester
     namespace: team-a  # workspace namespace
   rules:
     - apiGroups: ["mlflow.kubeflow.org"]
       resources: ["experiments"]
       verbs: ["create"]  # Required for trace ingestion
   ```
"""
