# Authorization Plugin

The `kubernetes-auth` plugin wraps the MLflow server with authorization checks backed by Kubernetes RBAC. It can protect both Flask and FastAPI routes exposed by MLflow.

## What It Does

- accepts bearer tokens from `Authorization` or `X-Forwarded-Access-Token`
- supports direct `SelfSubjectAccessReview` checks against the caller token
- supports trusted-proxy `SubjectAccessReview` mode using forwarded user and group headers
- checks MLflow operations against resources in the `mlflow.kubeflow.org` API group
- filters workspace lists down to namespaces the caller can access
- rewrites run ownership so the authenticated caller becomes the MLflow run owner
- blocks workspace create, update, and delete operations because namespaces remain externally managed

## Enable It

Install the package and add the MLflow app entry point at startup:

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://mlflow-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://" \
  --app-name kubernetes-auth
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS` | `300` | TTL for cached authorization decisions. |
| `MLFLOW_K8S_AUTH_USERNAME_CLAIM` | `sub` | JWT claim used for the MLflow run owner. |
| `MLFLOW_K8S_AUTH_AUTHORIZATION_MODE` | `self_subject_access_review` | Selects direct-token or trusted-proxy authorization mode. |
| `MLFLOW_K8S_AUTH_REMOTE_USER_HEADER` | `x-remote-user` | Username header used in trusted-proxy mode. |
| `MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER` | `x-remote-groups` | Groups header used in trusted-proxy mode. |
| `MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR` | `|` | Separator used to split the groups header. |

## Authorization Modes

### SelfSubjectAccessReview

This is the default mode. The plugin sends the caller's bearer token to the Kubernetes API and asks whether that identity can perform the requested action in the requested namespace.

Use this mode when MLflow receives end-user or workload tokens directly.

### SubjectAccessReview

In trusted-proxy mode, the plugin trusts proxy-provided user and group headers and asks Kubernetes to authorize that subject.

```bash
export MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=subject_access_review
export MLFLOW_K8S_AUTH_REMOTE_USER_HEADER=x-remote-user
export MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER=x-remote-groups
export MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR="|"
```

Use this mode only behind a trusted proxy such as `kube-rbac-proxy` that authenticates callers before forwarding requests to MLflow.

## Request Requirements

For authenticated requests, clients typically need:

- workspace context through `X-MLFLOW-WORKSPACE`, `mlflow.set_workspace()`, or `MLFLOW_WORKSPACE`
- a bearer token in `Authorization: Bearer <token>` or `X-Forwarded-Access-Token`

Example:

```bash
TOKEN=$(kubectl -n team-a create token mlflow-writer)
curl \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-MLFLOW-WORKSPACE: team-a" \
  http://mlflow.example/api/2.0/mlflow/experiments/search
```

## Notes

- workspace listings are filtered by the caller's visible namespaces
- GraphQL and FastAPI routes are covered in addition to the classic Flask endpoints
- gateway resource access uses `use` subresources for fine-grained RBAC checks

For the Kubernetes permissions required by the server and callers, see [`kubernetes-rbac.md`](kubernetes-rbac.md).
