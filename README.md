# MLflow Kubernetes Plugins

This repository packages two MLflow extensions for Kubernetes-backed deployments:

- a workspace provider that maps MLflow workspaces to Kubernetes namespaces
- an optional authorization plugin that enforces Kubernetes RBAC for MLflow requests

These plugins build on top of MLflow's 3.10 workspace support. If you are new to MLflow workspaces, start with the official guide: <https://mlflow.org/docs/latest/self-hosting/workspaces/getting-started/>. It covers the core MLflow server requirements, how workspace context is set by clients, and the upstream workspace lifecycle model.

## Components

| Entry point | MLflow hook | Purpose |
| --- | --- | --- |
| [`kubernetes`](docs/workspace-provider.md) | `mlflow.workspace_provider` | Exposes Kubernetes namespaces as MLflow workspaces. |
| [`kubernetes-auth`](docs/authorization-plugin.md) | `mlflow.app` | Wraps the MLflow server with Kubernetes-based authorization checks. |

## Install

Install from PyPI:

```bash
pip install mlflow-kubernetes-plugins
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quick Start

1. Enable MLflow workspaces on an MLflow server backed by a SQL store.
2. Install this package into the same environment as the MLflow server.
3. Configure the workspace provider and, if needed, the auth plugin.

```bash
export MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR="mlflow-enabled=true"
export MLFLOW_K8S_DEFAULT_WORKSPACE="team-a"

mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://mlflow-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://" \
  --app-name kubernetes-auth
```

Use `--app-name kubernetes-auth` only when you want request authorization enforced by Kubernetes RBAC.

## Documentation

- [`docs/index.md`](docs/index.md): docs index
- [`docs/workspace-provider.md`](docs/workspace-provider.md): workspace provider behavior, configuration, and startup
- [`docs/authorization-plugin.md`](docs/authorization-plugin.md): auth modes, headers, and request handling
- [`docs/kubernetes-rbac.md`](docs/kubernetes-rbac.md): RBAC requirements and example manifests

## Development

Run the main local checks from the repository root:

```bash
pip install -e ".[dev]"
ruff check .
pytest
python -m build
```
