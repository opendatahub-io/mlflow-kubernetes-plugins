# Kubernetes RBAC

Both plugins talk to the Kubernetes API, but they need different permissions depending on whether they are acting as the MLflow server or as the authenticated caller.

## MLflow Server Permissions

The workspace provider needs to list and watch namespaces. If you use the optional `MLflowConfig` CRD for artifact overrides, the server also needs access to that CRD and to the per-namespace secret named `mlflow-artifact-connection`.

Reusable example:

- [`examples/mlflow-server-rbac.yaml`](../examples/mlflow-server-rbac.yaml)

## Caller Permissions

The authorization plugin checks access against resources in the `mlflow.kubeflow.org` API group.

| Resource | Common verbs |
| --- | --- |
| `assistants` | `get`, `create`, `update` |
| `datasets` | `get`, `list`, `create`, `update`, `delete` |
| `experiments` | `get`, `list`, `create`, `update`, `delete` |
| `registeredmodels` | `get`, `list`, `create`, `update`, `delete` |
| `gatewaysecrets` | `get`, `list`, `create`, `update`, `delete` |
| `gatewaysecrets/use` | `create` |
| `gatewayendpoints` | `get`, `list`, `create`, `update`, `delete` |
| `gatewayendpoints/use` | `create` |
| `gatewaymodeldefinitions` | `get`, `list`, `create`, `update`, `delete` |
| `gatewaymodeldefinitions/use` | `create` |

Reusable example:

- [`examples/team-a-workspace.yaml`](../examples/team-a-workspace.yaml)

## Trusted Proxy Mode

When `MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=subject_access_review`, the MLflow server must also be allowed to create `subjectaccessreviews.authorization.k8s.io`.

That permission is included in:

- [`examples/mlflow-server-rbac.yaml`](../examples/mlflow-server-rbac.yaml)

## Token Creation

To create a service-account token for testing:

```bash
kubectl -n team-a create token mlflow-writer
kubectl -n team-a create token mlflow-experiments-reader
```

Then send the token with the workspace header:

```bash
curl \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-MLFLOW-WORKSPACE: team-a" \
  https://mlflow.example/api/2.0/mlflow/runs/search
```

Use plain HTTP only for local-only testing such as `localhost` or a `kubectl port-forward`.
