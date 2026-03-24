import pytest


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
        "kubernetes_workspace_provider.provider.watch.Watch",
        lambda: _FakeWatch(),
    )
    yield
    patcher.undo()
