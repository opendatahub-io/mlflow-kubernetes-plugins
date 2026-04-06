"""GraphQL authorization support for the Kubernetes auth plugin."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Callable, NamedTuple

import graphql
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.utils import workspace_context

from mlflow_kubernetes_plugins.auth.collection_filters import (
    COLLECTION_POLICY_GRAPHQL_FILTER,
    filter_graphql_experiment_ids,
    filter_graphql_model_versions_result,
)
from mlflow_kubernetes_plugins.auth.constants import (
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
)
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES,
)

if TYPE_CHECKING:
    from mlflow_kubernetes_plugins.auth.rules import AuthorizationRule

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

GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT = "single_object"
GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER = "request_filter"
GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER = "response_filter"
GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY = "broad_only"

GRAPHQL_FIELD_AUTH_POLICY_MAP: dict[str, str] = {
    "mlflowGetExperiment": GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT,
    "mlflowGetRun": GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT,
    "mlflowSearchRuns": GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER,
    "mlflowGetMetricHistoryBulkInterval": GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT,
    "mlflowListArtifacts": GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT,
    "mlflowSearchDatasets": GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER,
    "mlflowSearchModelVersions": GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER,
    "test": GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY,
    "testMutation": GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY,
}

GRAPHQL_FIELD_RESOURCE_NAME_PARSERS: dict[str, tuple[str, ...]] = {
    "mlflowGetExperiment": (RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,),
    "mlflowGetRun": (RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,),
    "mlflowGetMetricHistoryBulkInterval": (
        RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES,
    ),
    "mlflowListArtifacts": (RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,),
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
    nested_model_registry_root_fields: frozenset[str] = frozenset()
    root_field_inputs: tuple["GraphQLRootField", ...] = ()


class GraphQLRootField(NamedTuple):
    """A root field and the evaluated arguments passed to it."""

    field_name: str
    args: dict[str, object]


def _field_arguments_to_dict(
    field_node: graphql.language.ast.FieldNode, variables: dict[str, object]
) -> dict[str, object]:
    args: dict[str, object] = {}
    for argument in field_node.arguments or ():
        args[argument.name.value] = graphql.utilities.value_from_ast_untyped(
            argument.value, variables
        )
    return args


def _root_field_has_filterable_experiment_ids(root_field: "GraphQLRootField") -> bool:
    input_arg = root_field.args.get("input")
    if not isinstance(input_arg, dict):
        return False
    experiment_ids = input_arg.get("experimentIds") or input_arg.get("experiment_ids")
    return isinstance(experiment_ids, list) and bool(experiment_ids)


def _all_request_filter_occurrences_are_filterable(
    query_info: "GraphQLQueryInfo", field_name: str
) -> bool:
    matching_root_fields = [
        root_field
        for root_field in query_info.root_field_inputs
        if root_field.field_name == field_name
    ]
    return bool(matching_root_fields) and all(
        _root_field_has_filterable_experiment_ids(root_field)
        for root_field in matching_root_fields
    )


def extract_graphql_query_info(
    query_string: str, variables: dict[str, object] | None = None
) -> GraphQLQueryInfo:
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
        variables: Optional GraphQL variables used by the query.

    Returns:
        A GraphQLQueryInfo with root fields and whether nested model registry access exists.
    """
    variables = variables or {}

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
    nested_model_registry_root_fields: set[str] = set()
    root_field_inputs: list[GraphQLRootField] = []

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
        stack = [(selection_set, True, None)]  # (selection_set, is_root_level, root_field_name)

        while stack:
            current_selection_set, is_root, root_field_name = stack.pop()
            for selection in current_selection_set.selections:
                if isinstance(selection, graphql.language.ast.FieldNode):
                    field_name = selection.name.value
                    if is_root:
                        root_fields.add(field_name)
                        root_field_name = field_name
                        root_field_inputs.append(
                            GraphQLRootField(
                                field_name=field_name,
                                args=_field_arguments_to_dict(selection, variables),
                            )
                        )
                    # Check if this field accesses model registry data
                    if field_name in GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS:
                        has_nested_model_registry_access = True
                        if root_field_name is not None:
                            nested_model_registry_root_fields.add(root_field_name)
                    # Continue traversing nested selections
                    if selection.selection_set:
                        stack.append((selection.selection_set, False, root_field_name))

                elif isinstance(selection, graphql.language.ast.InlineFragmentNode):
                    # Inline fragment: ... on Type { fields }
                    # Traverse its selection set, preserving root level status
                    if selection.selection_set:
                        stack.append((selection.selection_set, is_root, root_field_name))

                elif isinstance(selection, graphql.language.ast.FragmentSpreadNode):
                    # Fragment spread: ... FragmentName
                    # Look up the fragment definition and traverse it
                    fragment_name = selection.name.value
                    if fragment_name not in visited_fragments:
                        visited_fragments.add(fragment_name)
                        fragment_def = fragment_definitions.get(fragment_name)
                        if fragment_def and fragment_def.selection_set:
                            stack.append((fragment_def.selection_set, is_root, root_field_name))

    return GraphQLQueryInfo(
        root_fields=root_fields,
        has_nested_model_registry_access=has_nested_model_registry_access,
        nested_model_registry_root_fields=frozenset(nested_model_registry_root_fields),
        root_field_inputs=tuple(root_field_inputs),
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
    rules_map: dict[tuple[str, str, str], tuple[str, ...]] = {}
    collection_policies: dict[tuple[str, str, str], str | None] = {}
    request_filterable_keys: dict[tuple[str, str, str], bool] = {}
    response_filter_keys: set[tuple[str, str, str]] = set()
    # Check for nested model registry access (e.g., modelVersions on runs). Skip the extra
    # broad registered-model get when the only nested access comes from the root
    # mlflowSearchModelVersions response-filter path, which should stay eligible for
    # partial-access filtering.
    if query_info.has_nested_model_registry_access and any(
        not (
            GRAPHQL_FIELD_RESOURCE_MAP.get(field) == RESOURCE_REGISTERED_MODELS
            and GRAPHQL_FIELD_AUTH_POLICY_MAP.get(field) == GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER
        )
        for field in query_info.nested_model_registry_root_fields
    ):
        rules_map[("get", RESOURCE_REGISTERED_MODELS, GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY)] = ()

    for field in sorted(query_info.root_fields):
        resource = GRAPHQL_FIELD_RESOURCE_MAP.get(field)
        if resource is None:
            unknown_fields.append(field)
            continue

        auth_policy = GRAPHQL_FIELD_AUTH_POLICY_MAP.get(field, GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY)
        verb = GRAPHQL_FIELD_VERB_MAP.get(field, "get")
        key = (verb, resource, auth_policy)
        parser_ids = (
            GRAPHQL_FIELD_RESOURCE_NAME_PARSERS.get(field, ())
            if auth_policy == GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT
            else ()
        )
        if existing_parser_ids := rules_map.get(key):
            parser_ids = tuple(dict.fromkeys(existing_parser_ids + parser_ids))
        rules_map[key] = parser_ids
        if auth_policy == GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER:
            request_filterable_keys[key] = request_filterable_keys.get(key, True) and (
                _all_request_filter_occurrences_are_filterable(query_info, field)
            )
        elif auth_policy == GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER:
            response_filter_keys.add(key)

    if unknown_fields:
        _logger.error(
            "GraphQL query contains fields without authorization coverage: %s. "
            "Add these fields to GRAPHQL_FIELD_RESOURCE_MAP and GRAPHQL_FIELD_VERB_MAP.",
            sorted(unknown_fields),
        )
        return None

    if not rules_map:
        return None

    for key, is_filterable in request_filterable_keys.items():
        if is_filterable:
            collection_policies[key] = COLLECTION_POLICY_GRAPHQL_FILTER
    for key in response_filter_keys:
        collection_policies[key] = COLLECTION_POLICY_GRAPHQL_FILTER

    return [
        authorization_rule_cls(
            verb,
            resource=resource,
            resource_name_parsers=rules_map[(verb, resource, policy)],
            collection_policy=collection_policies.get((verb, resource, policy)),
        )
        for verb, resource, policy in sorted(rules_map)
    ]


def validate_graphql_field_authorization() -> None:
    """Ensure all GraphQL schema fields are covered by authorization rules.

    This validates that every field in the GraphQL Query and Mutation types
    has a corresponding entry in GRAPHQL_FIELD_RESOURCE_MAP and GRAPHQL_FIELD_VERB_MAP.
    """
    # Import the schema here to avoid circular imports at module load time
    from mlflow.server.graphql.graphql_schema_extensions import schema

    missing_resource: list[str] = []
    missing_verb: list[str] = []
    missing_policy: list[str] = []
    invalid_single_object_policy: list[str] = []
    invalid_non_single_object_policy: list[str] = []

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
            if field_name not in GRAPHQL_FIELD_AUTH_POLICY_MAP:
                missing_policy.append(f"Query.{field_name}")

    # Check Mutation type fields
    if graphql_schema.mutation_type:
        for field_name in graphql_schema.mutation_type.fields:
            if field_name.startswith("__"):
                continue
            if field_name not in GRAPHQL_FIELD_RESOURCE_MAP:
                missing_resource.append(f"Mutation.{field_name}")
            if field_name not in GRAPHQL_FIELD_VERB_MAP:
                missing_verb.append(f"Mutation.{field_name}")
            if field_name not in GRAPHQL_FIELD_AUTH_POLICY_MAP:
                missing_policy.append(f"Mutation.{field_name}")

    for field_name, parser_ids in GRAPHQL_FIELD_RESOURCE_NAME_PARSERS.items():
        if parser_ids and GRAPHQL_FIELD_AUTH_POLICY_MAP.get(field_name) != GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT:
            invalid_non_single_object_policy.append(field_name)

    for field_name, auth_policy in GRAPHQL_FIELD_AUTH_POLICY_MAP.items():
        if (
            auth_policy == GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT
            and field_name in GRAPHQL_FIELD_RESOURCE_MAP
            and field_name not in GRAPHQL_FIELD_RESOURCE_NAME_PARSERS
        ):
            invalid_single_object_policy.append(field_name)

    errors: list[str] = []
    if missing_resource:
        errors.append(
            f"Missing from GRAPHQL_FIELD_RESOURCE_MAP: {', '.join(sorted(missing_resource))}"
        )
    if missing_verb:
        errors.append(f"Missing from GRAPHQL_FIELD_VERB_MAP: {', '.join(sorted(missing_verb))}")
    if missing_policy:
        errors.append(
            f"Missing from GRAPHQL_FIELD_AUTH_POLICY_MAP: {', '.join(sorted(missing_policy))}"
        )
    if invalid_single_object_policy:
        errors.append(
            "Missing single-object GraphQL resource-name policy: "
            + ", ".join(sorted(invalid_single_object_policy))
        )
    if invalid_non_single_object_policy:
        errors.append(
            "GraphQL resource-name parsers are only valid for single-object fields: "
            + ", ".join(sorted(invalid_non_single_object_policy))
        )

    if errors:
        raise MlflowException(
            "Kubernetes auth plugin is missing GraphQL field authorization coverage. "
            + "; ".join(errors),
            error_code=databricks_pb2.INTERNAL_ERROR,
        )


class KubernetesGraphQLAuthorizationMiddleware:
    """GraphQL field-level filtering for collection queries."""

    def __init__(self, authorizer) -> None:
        self._authorizer = authorizer

    def resolve(self, next, root, info, **args):
        field_name = info.field_name
        policy = GRAPHQL_FIELD_AUTH_POLICY_MAP.get(field_name)
        if policy not in {
            GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER,
            GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER,
        }:
            return next(root, info, **args)

        from mlflow_kubernetes_plugins.auth.core import _AUTHORIZATION_HANDLED

        auth_result = _AUTHORIZATION_HANDLED.get()
        identity = auth_result.identity if auth_result is not None else None
        workspace_name = workspace_context.get_request_workspace()
        if identity is None or not workspace_name:
            _logger.warning(
                "GraphQL collection authorization missing identity or workspace for field %s",
                field_name,
            )
            return None

        if policy == GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER:
            input_obj = args.get("input")
            if isinstance(input_obj, MutableMapping):
                experiment_ids = (
                    input_obj.get("experiment_ids") or input_obj.get("experimentIds") or []
                )
            else:
                experiment_ids = (
                    getattr(input_obj, "experiment_ids", None)
                    or getattr(input_obj, "experimentIds", None)
                    or []
                )
            if experiment_ids:
                readable_ids = filter_graphql_experiment_ids(
                    self._authorizer,
                    identity,
                    workspace_name,
                    [str(experiment_id) for experiment_id in experiment_ids],
                )
                if not readable_ids:
                    return None
                if isinstance(input_obj, MutableMapping):
                    if "experiment_ids" in input_obj:
                        input_obj["experiment_ids"] = readable_ids
                    elif "experimentIds" in input_obj:
                        input_obj["experimentIds"] = readable_ids
                elif hasattr(input_obj, "experiment_ids"):
                    input_obj.experiment_ids = readable_ids
                elif hasattr(input_obj, "experimentIds"):
                    input_obj.experimentIds = readable_ids

        result = next(root, info, **args)
        if result is None:
            return None
        if policy == GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER:
            return filter_graphql_model_versions_result(
                result,
                authorizer=self._authorizer,
                identity=identity,
                workspace_name=workspace_name,
            )
        return result


def get_graphql_authorization_middleware(authorizer):
    return [KubernetesGraphQLAuthorizationMiddleware(authorizer)]


__all__ = [
    "GRAPHQL_FIELD_AUTH_POLICY_BROAD_ONLY",
    "GRAPHQL_FIELD_AUTH_POLICY_MAP",
    "GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER",
    "GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER",
    "GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT",
    "GRAPHQL_FIELD_RESOURCE_MAP",
    "GRAPHQL_FIELD_RESOURCE_NAME_PARSERS",
    "GRAPHQL_FIELD_VERB_MAP",
    "GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS",
    "GraphQLQueryInfo",
    "GraphQLRootField",
    "K8S_GRAPHQL_OPERATION_RESOURCE_MAP",
    "K8S_GRAPHQL_OPERATION_VERB_MAP",
    "KubernetesGraphQLAuthorizationMiddleware",
    "determine_graphql_rules",
    "extract_graphql_query_info",
    "get_graphql_authorization_middleware",
    "validate_graphql_field_authorization",
]
