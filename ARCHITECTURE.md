# Architecture Overview

This document provides a high-level overview of the codebase's architecture. For detailed plugin
behavior and configuration, see the docs in [`docs/`](docs/index.md).

## System Diagram

```
        HTTP request (bearer token or proxy auth headers)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                         MLflow Server                         │
│                                                               │
│  ┌──────────────────────────┐  ┌────────────────────────────┐ │
│  │   Authorization Plugin   │  │   Workspace Provider       │ │
│  │   (kubernetes-auth)      │  │   (kubernetes)             │ │
│  │                          │  │                            │ │
│  │  Intercepts requests,    │  │  Maps K8s namespaces       │ │
│  │  checks K8s RBAC,        │  │  to MLflow workspaces,     │ │
│  │  filters responses       │  │  watches for changes       │ │
│  └────────────┬─────────────┘  └─────────────┬──────────────┘ │
│               │                              │                │
└───────────────┼──────────────────────────────┼────────────────┘
                │                              │
                ▼                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       Kubernetes API                          │
│                                                               │
│  SubjectAccessReview    Namespaces       MLflowConfig CRD     │
│  (authorization)        (workspaces)     (artifact overrides) │
│                                          Secrets              │
└───────────────────────────────────────────────────────────────┘
```

## Plugin Discovery

Both plugins are registered as [setuptools entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
in `pyproject.toml`. MLflow discovers and loads them automatically at server startup — no
additional configuration is needed beyond installing the package.

## Core Components

### Workspace Provider (`mlflow.workspace_provider` → `kubernetes`)

Exposes Kubernetes namespaces as MLflow workspaces with 1:1 mapping. Watches namespaces in the
background and reads per-namespace configuration from the MLflowConfig CRD.

### Authorization Plugin (`mlflow.app` → `kubernetes-auth`)

Starlette middleware that enforces Kubernetes RBAC on every MLflow request by mapping operations to
virtual resources and issuing SubjectAccessReviews.

### MLflowConfig CRD

A namespace-scoped CRD (`api/mlflowconfig/v1/`, `config/crd/bases/`) that lets namespace owners
override default artifact storage locations.

For detailed configuration, request flow, and RBAC setup, see the [`docs/`](docs/index.md)
directory.

## Security

- Startup validation fails if any MLflow endpoint lacks explicit authorization coverage.
- Collection responses are filtered so callers only see resources they can access.
- Workspace create/update/delete operations are blocked — namespace management belongs to Kubernetes.
- See [`docs/kubernetes-rbac.md`](docs/kubernetes-rbac.md) for RBAC requirements and example manifests.

## Glossary

- **CRD:** Custom Resource Definition — extends the Kubernetes API with new resource types
- **RBAC:** Role-Based Access Control — Kubernetes authorization system
- **SAR:** SubjectAccessReview — Kubernetes API for checking authorization decisions
- **SSAR:** SelfSubjectAccessReview — same as SAR but uses the caller's own token
