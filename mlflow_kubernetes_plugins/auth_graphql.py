"""GraphQL authorization support for the Kubernetes auth plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, NamedTuple

import graphql
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2

from mlflow_kubernetes_plugins.auth import (
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
)

if TYPE_CHECKING:
    from mlflow_kubernetes_plugins.auth import AuthorizationRule

_logger = logging.getLogger(__name__)


# Mapping from GraphQL operationName values to Kubernetes resources.
# These are the operation names that clients may send in the operationName field.
K8S_GRAPHQL_OPERATION_RESOURCE_MAP = {
    # Experiment / Run surfaces
    "MlflowGetExperimentQuery": RESOURCE_EXPERIMENTS,
    "GetExperiment": RESOURCE_EXPERIMENTS,
    "GetRun": RESOURCE_EXPERIMENTS,
    "MlflowGetRunQuery": RESOURCE_EXPERIMENTS,
    "SearchRuns": RESOURCE_EXPERIMENTS,
    "MlflowSearchRunsQuery": RESOURCE_EXPERIMENTS,
    "GetMetricHistoryBulkInterval": RESOURCE_EXPERIMENTS,
    "MlflowGetMetricHistoryBulkIntervalQuery": RESOURCE_EXPERIMENTS,
    # Model Registry surfaces
    "SearchModelVersions": RESOURCE_REGISTERED_MODELS,
    "MlflowSearchModelVersionsQuery": RESOURCE_REGISTERED_MODELS,
    "GetModelVersion": RESOURCE_REGISTERED_MODELS,
    "MlflowGetModelVersionQuery": RESOURCE_REGISTERED_MODELS,
    "GetRegisteredModel": RESOURCE_REGISTERED_MODELS,
    "MlflowGetRegisteredModelQuery": RESOURCE_REGISTERED_MODELS,
    "SearchRegisteredModels": RESOURCE_REGISTERED_MODELS,
    "MlflowSearchRegisteredModelsQuery": RESOURCE_REGISTERED_MODELS,
}

# Mapping from GraphQL operationName values to authorization verbs.
K8S_GRAPHQL_OPERATION_VERB_MAP: dict[str, str] = {
    # Experiment / Run surfaces
    "MlflowGetExperimentQuery": "get",
    "GetExperiment": "get",
    "GetRun": "get",
    "MlflowGetRunQuery": "get",
    "SearchRuns": "list",
    "MlflowSearchRunsQuery": "list",
    "GetMetricHistoryBulkInterval": "get",
    "MlflowGetMetricHistoryBulkIntervalQuery": "get",
    # Model Registry surfaces
    "SearchModelVersions": "list",
    "MlflowSearchModelVersionsQuery": "list",
    "GetModelVersion": "get",
    "MlflowGetModelVersionQuery": "get",
    "GetRegisteredModel": "get",
    "MlflowGetRegisteredModelQuery": "get",
    "SearchRegisteredModels": "list",
    "MlflowSearchRegisteredModelsQuery": "list",
}

# Mapping from actual GraphQL field names (camelCase as they appear in queries) to resources.
# This is used when operationName is not provided to determine authorization from the query itself.
GRAPHQL_FIELD_RESOURCE_MAP: dict[str, str] = {
    # Experiment / Run fields (QueryType and MutationType)
    "mlflowGetExperiment": RESOURCE_EXPERIMENTS,
    "mlflowGetRun": RESOURCE_EXPERIMENTS,
    "mlflowSearchRuns": RESOURCE_EXPERIMENTS,
    "mlflowGetMetricHistoryBulkInterval": RESOURCE_EXPERIMENTS,
    "mlflowListArtifacts": RESOURCE_EXPERIMENTS,
    "mlflowSearchDatasets": RESOURCE_EXPERIMENTS,
    # Model Registry fields
    "mlflowSearchModelVersions": RESOURCE_REGISTERED_MODELS,
    # Test fields (simple echo for testing)
    "test": RESOURCE_EXPERIMENTS,
    "testMutation": RESOURCE_EXPERIMENTS,
}

# Mapping from GraphQL field names to verbs for authorization
GRAPHQL_FIELD_VERB_MAP: dict[str, str] = {
    # Experiment / Run fields
    "mlflowGetExperiment": "get",
    "mlflowGetRun": "get",
    "mlflowSearchRuns": "list",
    "mlflowGetMetricHistoryBulkInterval": "get",
    "mlflowListArtifacts": "get",
    "mlflowSearchDatasets": "list",
    # Model Registry fields
    "mlflowSearchModelVersions": "list",
    # Test fields
    "test": "get",
    "testMutation": "get",
}

# Nested field names that indicate model registry access when present anywhere in the query.
# These are fields on types like MlflowRunExtension that fetch model registry data.
GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS: frozenset[str] = frozenset(
    {
        "modelVersions",  # MlflowRunExtension.model_versions - fetches model versions for a run
    }
)


def _build_graphql_operation_rules(
    authorization_rule_cls: type["AuthorizationRule"],
    normalize_resource: Callable[[str | None], str | None],
) -> dict[str, "AuthorizationRule"]:
    """Build the GRAPHQL_OPERATION_RULES mapping from operation names to AuthorizationRules.

    Args:
        authorization_rule_cls: The AuthorizationRule class from auth.py.
        normalize_resource: Function to normalize resource names.

    Returns:
        Dictionary mapping operation names to AuthorizationRule instances.
    """
    rules: dict[str, AuthorizationRule] = {}
    resource_operations = set(K8S_GRAPHQL_OPERATION_RESOURCE_MAP)
    verb_operations = set(K8S_GRAPHQL_OPERATION_VERB_MAP)

    missing_verbs = resource_operations - verb_operations
    extra_verbs = verb_operations - resource_operations
    if missing_verbs or extra_verbs:
        details = []
        if missing_verbs:
            details.append(f"missing verbs for {sorted(missing_verbs)}")
        if extra_verbs:
            details.append(f"unexpected verbs for {sorted(extra_verbs)}")
        raise ValueError("GraphQL operation mappings are inconsistent: " + "; ".join(details))

    for operation_name in resource_operations:
        resource = K8S_GRAPHQL_OPERATION_RESOURCE_MAP[operation_name]
        verb = K8S_GRAPHQL_OPERATION_VERB_MAP[operation_name]
        rules[operation_name] = authorization_rule_cls(
            verb, resource=normalize_resource(resource) or RESOURCE_EXPERIMENTS
        )
    return rules


class GraphQLQueryInfo(NamedTuple):
    """Information extracted from a GraphQL query for authorization."""

    root_fields: set[str]
    has_nested_model_registry_access: bool


def extract_graphql_query_info(query_string: str) -> GraphQLQueryInfo:
    """
    Parse a GraphQL query string and extract authorization-relevant information.

    Scans the entire query tree to detect:
    - Root field names (for determining primary resource type)
    - Nested fields that access model registry data (e.g., modelVersions on runs)

    The traversal handles all GraphQL selection types:
    - FieldNode: Regular field selections
    - InlineFragmentNode: Inline fragments (... on Type { fields })
    - FragmentSpreadNode: Named fragment references (... FragmentName)

    Args:
        query_string: The GraphQL query string from the request payload.

    Returns:
        A GraphQLQueryInfo with root fields and whether nested model registry access exists.
    """
    try:
        ast = graphql.parse(query_string)
    except graphql.GraphQLError:
        return GraphQLQueryInfo(root_fields=set(), has_nested_model_registry_access=False)

    # Collect fragment definitions for resolving FragmentSpread references
    fragment_definitions: dict[str, graphql.language.ast.FragmentDefinitionNode] = {}
    for definition in ast.definitions:
        if isinstance(definition, graphql.language.ast.FragmentDefinitionNode):
            fragment_definitions[definition.name.value] = definition

    root_fields: set[str] = set()
    has_nested_model_registry_access = False

    for definition in ast.definitions:
        # Skip fragment definitions - they're processed when referenced
        if isinstance(definition, graphql.language.ast.FragmentDefinitionNode):
            continue

        selection_set = getattr(definition, "selection_set", None)
        if selection_set is None:
            continue

        # Use a stack to traverse the entire query tree
        # Track visited fragments to prevent infinite loops from circular references
        visited_fragments: set[str] = set()
        stack = [(selection_set, True)]  # (selection_set, is_root_level)

        while stack:
            current_selection_set, is_root = stack.pop()
            for selection in current_selection_set.selections:
                if isinstance(selection, graphql.language.ast.FieldNode):
                    field_name = selection.name.value
                    if is_root:
                        root_fields.add(field_name)
                    # Check if this field accesses model registry data
                    if field_name in GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS:
                        has_nested_model_registry_access = True
                    # Continue traversing nested selections
                    if selection.selection_set:
                        stack.append((selection.selection_set, False))

                elif isinstance(selection, graphql.language.ast.InlineFragmentNode):
                    # Inline fragment: ... on Type { fields }
                    # Traverse its selection set, preserving root level status
                    if selection.selection_set:
                        stack.append((selection.selection_set, is_root))

                elif isinstance(selection, graphql.language.ast.FragmentSpreadNode):
                    # Fragment spread: ... FragmentName
                    # Look up the fragment definition and traverse it
                    fragment_name = selection.name.value
                    if fragment_name not in visited_fragments:
                        visited_fragments.add(fragment_name)
                        fragment_def = fragment_definitions.get(fragment_name)
                        if fragment_def and fragment_def.selection_set:
                            stack.append((fragment_def.selection_set, is_root))

    return GraphQLQueryInfo(
        root_fields=root_fields,
        has_nested_model_registry_access=has_nested_model_registry_access,
    )


def determine_graphql_rules(
    query_info: GraphQLQueryInfo, authorization_rule_cls: type["AuthorizationRule"]
) -> "list[AuthorizationRule] | None":
    """
    Determine all authorization rules needed for a GraphQL query.

    Returns a unique list of authorization rules - one for each (resource, verb)
    combination accessed by the query. This ensures that users must have permissions
    for ALL operations accessed by a query.

    Args:
        query_info: Information extracted from the GraphQL query.
        authorization_rule_cls: The AuthorizationRule class from auth.py.

    Returns:
        A unique list of AuthorizationRules for all operations accessed by the query,
        or None if any field is not recognized (missing authorization coverage).
    """
    unknown_fields: list[str] = []
    rules_set: set[tuple[str, str]] = set()  # (verb, resource) pairs

    # Check for nested model registry access (e.g., modelVersions on runs)
    if query_info.has_nested_model_registry_access:
        rules_set.add(("get", RESOURCE_REGISTERED_MODELS))

    for field in query_info.root_fields:
        resource = GRAPHQL_FIELD_RESOURCE_MAP.get(field)
        if resource is None:
            unknown_fields.append(field)
            continue

        verb = GRAPHQL_FIELD_VERB_MAP.get(field, "get")
        rules_set.add((verb, resource))

    if unknown_fields:
        _logger.error(
            "GraphQL query contains fields without authorization coverage: %s. "
            "Add these fields to GRAPHQL_FIELD_RESOURCE_MAP and GRAPHQL_FIELD_VERB_MAP.",
            sorted(unknown_fields),
        )
        return None

    if not rules_set:
        return None

    return [authorization_rule_cls(verb, resource=resource) for verb, resource in rules_set]


def validate_graphql_field_authorization() -> None:
    """Ensure all GraphQL schema fields are covered by authorization rules.

    This validates that every field in the GraphQL Query and Mutation types
    has a corresponding entry in GRAPHQL_FIELD_RESOURCE_MAP and GRAPHQL_FIELD_VERB_MAP.
    """
    # Import the schema here to avoid circular imports at module load time
    from mlflow.server.graphql.graphql_schema_extensions import schema

    missing_resource: list[str] = []
    missing_verb: list[str] = []

    # Get the underlying GraphQL schema
    graphql_schema = schema.graphql_schema

    # Check Query type fields
    if graphql_schema.query_type:
        for field_name in graphql_schema.query_type.fields:
            # Skip introspection fields (start with __)
            if field_name.startswith("__"):
                continue
            if field_name not in GRAPHQL_FIELD_RESOURCE_MAP:
                missing_resource.append(f"Query.{field_name}")
            if field_name not in GRAPHQL_FIELD_VERB_MAP:
                missing_verb.append(f"Query.{field_name}")

    # Check Mutation type fields
    if graphql_schema.mutation_type:
        for field_name in graphql_schema.mutation_type.fields:
            if field_name.startswith("__"):
                continue
            if field_name not in GRAPHQL_FIELD_RESOURCE_MAP:
                missing_resource.append(f"Mutation.{field_name}")
            if field_name not in GRAPHQL_FIELD_VERB_MAP:
                missing_verb.append(f"Mutation.{field_name}")

    errors: list[str] = []
    if missing_resource:
        errors.append(
            f"Missing from GRAPHQL_FIELD_RESOURCE_MAP: {', '.join(sorted(missing_resource))}"
        )
    if missing_verb:
        errors.append(f"Missing from GRAPHQL_FIELD_VERB_MAP: {', '.join(sorted(missing_verb))}")

    if errors:
        raise MlflowException(
            "Kubernetes auth plugin is missing GraphQL field authorization coverage. "
            + "; ".join(errors),
            error_code=databricks_pb2.INTERNAL_ERROR,
        )
