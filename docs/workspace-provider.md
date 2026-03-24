# Workspace Provider

The `kubernetes` workspace provider exposes Kubernetes namespaces as MLflow workspaces. Each MLflow workspace maps 1:1 to a namespace, so workspace lifecycle stays external to MLflow.

If you need the upstream MLflow workspace concepts first, read the official guide: <https://mlflow.org/docs/latest/self-hosting/workspaces/getting-started/>.

## What It Does

- lists Kubernetes namespaces as workspaces
- watches namespaces in the background so listings stay warm
- filters built-in system namespaces such as `kube-*` and `openshift-*`
- optionally filters namespaces with a label selector
- reads workspace descriptions from the `mlflow.kubeflow.org/workspace-description` annotation
- supports per-namespace artifact root overrides through the optional `MLflowConfig` CRD
- keeps workspace CRUD read-only because namespace management belongs to Kubernetes

## Installation

```bash
pip install mlflow-kubernetes-plugins
```

## Configuration

The provider reads configuration from environment variables, constructor arguments, and `kubernetes://` URI query parameters. URI parameters win over environment variables.

### Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR` | unset | Limits visible namespaces to those matching a Kubernetes label selector. |
| `MLFLOW_K8S_DEFAULT_WORKSPACE` | unset | Workspace to use when a request omits explicit workspace context. |
| `MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS` | built-in exclusions | Extra comma-separated glob patterns to hide. |

### Workspace URI Parameters

Pass the same values through the workspace store URI when you want per-deployment overrides:

- `label_selector`
- `default_workspace`
- `namespace_exclude_globs`

Example:

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://mlflow-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://?label_selector=mlflow-enabled%3Dtrue&default_workspace=team-a"
```

## Running The Server

The provider loads in-cluster Kubernetes credentials first and falls back to the local kubeconfig. That allows the same plugin to work in a cluster or from a development workstation.

```bash
export MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR="mlflow-enabled=true"
export MLFLOW_K8S_DEFAULT_WORKSPACE="team-a"

mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://mlflow-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://"
```

## Client Usage

Clients still use standard MLflow workspace APIs and headers:

- call `mlflow.set_workspace("team-a")`
- set `MLFLOW_WORKSPACE=team-a`
- or send `X-MLFLOW-WORKSPACE: team-a`

If `MLFLOW_K8S_DEFAULT_WORKSPACE` is unset and the client does not specify a workspace, the server returns an "Active workspace is required" error.

## Artifact Root Overrides

If the optional `MLflowConfig` CRD is installed, a namespace can override the server's default artifact root for the MLflow workspace. The plugin reads:

- `spec.artifactRootSecret` for the secret containing `AWS_S3_BUCKET`
- `spec.artifactRootPath` for an optional path suffix under that bucket

This lets each namespace point to a different object store location without changing MLflow server startup flags.
