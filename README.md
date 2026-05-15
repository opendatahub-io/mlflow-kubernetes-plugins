# MLflow Kubeflow Integration

[![PyPI version](https://img.shields.io/pypi/v/mlflow-kubernetes-plugins?color=%2334D058&label=pypi%20package)](https://pypi.org/project/mlflow-kubernetes-plugins/)
[![License](https://img.shields.io/github/license/kubeflow/mlflow-integration)](https://github.com/kubeflow/mlflow-integration/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

This repository provides the integration layer between [MLflow](https://mlflow.org/) and
[Kubeflow](https://www.kubeflow.org/), making MLflow the first-class experiment tracking
experience for the Kubeflow platform
([KEP-897](https://github.com/kubeflow/community/pull/892)).

This repository packages two MLflow extensions for Kubernetes-backed deployments:

- a workspace provider that maps MLflow workspaces to Kubernetes namespaces
- an optional authorization plugin that enforces Kubernetes RBAC for MLflow requests

These plugins build on top of MLflow's workspace support. If you are new to MLflow workspaces,
start with the official guide: <https://mlflow.org/docs/latest/self-hosting/workspaces/getting-started/>.

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
- `config/crd/bases/mlflow.kubeflow.org_mlflowconfigs.yaml`: generated `MLflowConfig` CRD manifest

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, and testing instructions.
