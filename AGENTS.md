# AGENTS

- Python package code lives in `mlflow_kubernetes_plugins/`.
- User-facing docs live in `README.md` and `docs/`.
- Reusable Kubernetes manifests live in `examples/`.
- Preserve the MLflow entry point IDs `kubernetes` and `kubernetes-auth` unless you are intentionally making a breaking change.
- For auth coverage, preserve the explicit 1x1 mapping from MLflow endpoint to authorization requirement. Missing protected endpoint coverage should still fail at startup rather than falling back dynamically.
- In auth rule tables, readability and auditability are more important than deduplication. Keep duplicated literal route entries when that makes endpoint coverage clearer.
- GraphQL authorization must remain query-driven. Do not rely on `operationName` alone for authorization decisions.
- When refactoring auth internals, preserve compatibility through `mlflow_kubernetes_plugins.auth` unless you are intentionally making a breaking change.
- Before finishing, run `ruff check .`, `pytest`, and `python -m build` from the repository root when possible.
