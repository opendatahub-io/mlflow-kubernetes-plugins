# AGENTS

- Python package code lives in `mlflow_kubernetes_plugins/`.
- User-facing docs live in `README.md` and `docs/`.
- Reusable Kubernetes manifests live in `examples/`.
- Preserve the MLflow entry point IDs `kubernetes` and `kubernetes-auth` unless you are intentionally making a breaking change.
- Before finishing, run `ruff check .`, `pytest`, and `python -m build` from the repository root when possible.
