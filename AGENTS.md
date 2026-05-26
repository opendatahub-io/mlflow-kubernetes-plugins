# AGENTS

## Project Layout

- Python package code lives in `mlflow_kubernetes_plugins/`.
- User-facing docs live in `README.md` and `docs/`.
- Reusable Kubernetes manifests live in `examples/`.
- Architecture overview in `ARCHITECTURE.md`.

## Commands

```bash
# Setup
make install-dev                               # Sync development dependencies

# Lint
make python-lint                               # Lint entire project
uv run ruff check path/to/file.py              # Lint single file
uv run ruff check --fix path/to/file.py        # Lint and auto-fix single file
uv run ruff format path/to/file.py             # Format single file

# Type checking
make python-typecheck                          # Type-check entire project
uv run ty check path/to/file.py                # Type-check single file

# Testing
make python-test                               # Run all tests
uv run pytest tests/test_auth.py               # Run single test file
uv run pytest tests/test_auth.py -k "pattern"  # Run tests matching pattern

# Build
uv build                                       # Build distribution artifacts

# Pre-commit
uv run pre-commit run --all-files              # Run all hooks
```

## Key Conventions

- Preserve the MLflow entry point IDs `kubernetes` and `kubernetes-auth` unless you are intentionally making a breaking change.
- For auth coverage, preserve the explicit 1x1 mapping from MLflow endpoint to authorization requirement. Missing protected endpoint coverage should still fail at startup rather than falling back dynamically.
- In auth rule tables, readability and auditability are more important than deduplication. Keep duplicated literal route entries when that makes endpoint coverage clearer.
- GraphQL authorization must remain query-driven. Do not rely on `operationName` alone for authorization decisions.

## Patterns

- When adding auth rules for a new MLflow version, follow the pattern in `rules_v3_11.py` and `rules_v3_12.py`: create a new `rules_v3_XX.py` file, add the version-specific rules, and register them in `rules.py`.
- When adding a new collection filter, follow the existing filters in `collection_filters.py`.
- When adding a new resource type for fine-grained RBAC, add the resource name extraction logic in `resource_names.py`.

## Before Finishing

- Use `make install-dev` to sync development dependencies locally.
- Before finishing, run `make python-lint`, `make python-typecheck`, `make python-test`, and
  `uv build` from the repository root when possible.
