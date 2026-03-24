"""Compatibility exports for GraphQL authorization helpers."""

from mlflow_kubernetes_plugins.auth_graphql import (
    GRAPHQL_FIELD_RESOURCE_MAP,
    GRAPHQL_FIELD_VERB_MAP,
    GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    GraphQLQueryInfo,
    _build_graphql_operation_rules,
    determine_graphql_rules,
    extract_graphql_query_info,
    validate_graphql_field_authorization,
)

__all__ = [
    "GRAPHQL_FIELD_RESOURCE_MAP",
    "GRAPHQL_FIELD_VERB_MAP",
    "GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS",
    "K8S_GRAPHQL_OPERATION_RESOURCE_MAP",
    "K8S_GRAPHQL_OPERATION_VERB_MAP",
    "GraphQLQueryInfo",
    "_build_graphql_operation_rules",
    "determine_graphql_rules",
    "extract_graphql_query_info",
    "validate_graphql_field_authorization",
]
