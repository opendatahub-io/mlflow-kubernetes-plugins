import asyncio
import os

import pytest
from mlflow_kubernetes_plugins.auth.core import _authorize_request_async


def _authorize_request(request_context, *, authorizer, config_values):
    """Synchronous wrapper for tests -- must not be called from a running event loop."""
    return asyncio.run(
        _authorize_request_async(
            request_context,
            authorizer=authorizer,
            config_values=config_values,
        )
    )


@pytest.fixture(scope="session", autouse=True)
def _mock_namespace_watch():
    patcher = pytest.MonkeyPatch()

    class _FakeWatch:
        def stream(self, *args, **kwargs):
            return iter(())

        def stop(self):
            return None

        def close(self):
            return None

    patcher.setattr(
        "mlflow_kubernetes_plugins.workspace_plugin.caches.watch.Watch",
        lambda: _FakeWatch(),
    )
    yield
    patcher.undo()


@pytest.fixture
def compile_auth_rules(monkeypatch):
    def _compile(endpoint_specs):
        if os.environ.get("K8S_AUTH_TEST_SKIP_COMPILE") == "1":
            return

        def _fake_get_endpoints(resolver):
            return [(path, resolver(request_class), methods) for path, request_class, methods in endpoint_specs]

        monkeypatch.setattr("mlflow_kubernetes_plugins.auth.compiler.get_endpoints", _fake_get_endpoints)
        monkeypatch.setattr(
            "mlflow_kubernetes_plugins.auth.compiler.mlflow_app.url_map.iter_rules",
            lambda: [],
        )

        from mlflow_kubernetes_plugins.auth.compiler import (
            _compile_authorization_rules,
            _reset_compiled_rules,
        )

        _reset_compiled_rules()
        _compile_authorization_rules()

    return _compile
