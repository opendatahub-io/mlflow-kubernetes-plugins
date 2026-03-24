import os

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import CreateRun
from mlflow_kubernetes_plugins.auth import (
    GRAPHQL_OPERATION_RULES,
    AuthorizationRule,
    _compile_authorization_rules,
    _find_authorization_rules,
)
from mlflow_kubernetes_plugins.auth_graphql import (
    GRAPHQL_FIELD_RESOURCE_MAP,
    GRAPHQL_FIELD_VERB_MAP,
    GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
    GraphQLQueryInfo,
    determine_graphql_rules,
    extract_graphql_query_info,
    validate_graphql_field_authorization,
)


@pytest.fixture(autouse=True)
def _compile_rules(monkeypatch):
    """Ensure authorization rules are populated before each test."""
    if os.environ.get("K8S_AUTH_TEST_SKIP_COMPILE") == "1":
        return

    # Limit endpoint discovery to avoid unrelated Flask routes during tests
    def _fake_get_endpoints(resolver):
        return [
            ("/api/2.0/mlflow/runs/create", resolver(CreateRun), ["POST"]),
        ]

    monkeypatch.setattr("mlflow_kubernetes_plugins.auth.get_endpoints", _fake_get_endpoints)
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.mlflow_app.url_map.iter_rules",
        lambda: [],
    )
    _compile_authorization_rules()


def test_graphql_operation_map_matches_constant():
    assert set(K8S_GRAPHQL_OPERATION_RESOURCE_MAP) == set(K8S_GRAPHQL_OPERATION_VERB_MAP)
    for operation_name in K8S_GRAPHQL_OPERATION_RESOURCE_MAP:
        assert operation_name in GRAPHQL_OPERATION_RULES
        assert (
            GRAPHQL_OPERATION_RULES[operation_name].verb
            == K8S_GRAPHQL_OPERATION_VERB_MAP[operation_name]
        )


def test_graphql_unknown_operation_returns_none():
    rules = _find_authorization_rules(
        "/graphql", "POST", graphql_payload={"operationName": "NewGraphQLOperation"}
    )
    assert rules is None


def test_graphql_field_maps_are_consistent():
    assert set(GRAPHQL_FIELD_RESOURCE_MAP) == set(GRAPHQL_FIELD_VERB_MAP)


def test_extract_graphql_query_info_single_query():
    query = '{ mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } } }'
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowGetExperiment"}
    assert info.has_nested_model_registry_access is False


def test_extract_graphql_query_info_multiple_fields():
    query = """
    {
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
        mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } }
    }
    """
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowGetExperiment", "mlflowSearchModelVersions"}
    # Note: modelVersions in the response path is detected, but this doesn't affect
    # authorization since mlflowSearchModelVersions already requires model registry access
    assert info.has_nested_model_registry_access is True


def test_extract_graphql_query_info_named_operation():
    query = """
    query GetModels {
        mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } }
    }
    """
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowSearchModelVersions"}
    # Note: modelVersions in response path is detected, but authorization is still correct
    assert info.has_nested_model_registry_access is True


def test_extract_graphql_query_info_experiment_only_no_nested_models():
    query = """
    {
        mlflowGetExperiment(input: { experimentId: "123" }) {
            experiment { name tags { key value } }
        }
    }
    """
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowGetExperiment"}
    assert info.has_nested_model_registry_access is False


def test_extract_graphql_query_info_invalid_query():
    info = extract_graphql_query_info("not a valid graphql query {{{")
    assert info.root_fields == set()
    assert info.has_nested_model_registry_access is False


def test_extract_graphql_query_info_empty_query():
    info = extract_graphql_query_info("")
    assert info.root_fields == set()
    assert info.has_nested_model_registry_access is False


def test_extract_graphql_query_info_nested_model_versions():
    query = """
    {
        mlflowGetRun(input: { runId: "abc" }) {
            run {
                info { runId }
                modelVersions { name version }
            }
        }
    }
    """
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowGetRun"}
    assert info.has_nested_model_registry_access is True


def test_extract_graphql_query_info_deeply_nested_model_versions():
    query = """
    {
        mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs {
                info { runId }
                data { params { key value } }
                modelVersions { name version status }
            }
        }
    }
    """
    info = extract_graphql_query_info(query)
    assert info.root_fields == {"mlflowSearchRuns"}
    assert info.has_nested_model_registry_access is True


def test_determine_graphql_rules_experiments_only():
    info = GraphQLQueryInfo(
        root_fields={"mlflowGetExperiment", "mlflowGetRun"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource == RESOURCE_EXPERIMENTS
    assert rules[0].verb == "get"


def test_determine_graphql_rules_models_only():
    info = GraphQLQueryInfo(
        root_fields={"mlflowSearchModelVersions"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource == RESOURCE_REGISTERED_MODELS
    assert rules[0].verb == "list"


def test_determine_graphql_rules_mixed_query_returns_both():
    info = GraphQLQueryInfo(
        root_fields={"mlflowGetExperiment", "mlflowSearchModelVersions"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    # Should return TWO rules - one for experiments, one for models
    assert len(rules) == 2
    resources = {r.resource for r in rules}
    assert RESOURCE_EXPERIMENTS in resources
    assert RESOURCE_REGISTERED_MODELS in resources
    # Check verbs
    exp_rule = next(r for r in rules if r.resource == RESOURCE_EXPERIMENTS)
    model_rule = next(r for r in rules if r.resource == RESOURCE_REGISTERED_MODELS)
    assert exp_rule.verb == "get"
    assert model_rule.verb == "list"


def test_determine_graphql_rules_multiple_verbs_same_resource():
    info = GraphQLQueryInfo(
        root_fields={"mlflowGetExperiment", "mlflowSearchRuns"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    # Should return TWO rules - one for get, one for list (both experiments)
    assert len(rules) == 2
    verbs = {r.verb for r in rules}
    assert "get" in verbs
    assert "list" in verbs
    for rule in rules:
        assert rule.resource == RESOURCE_EXPERIMENTS


def test_determine_graphql_rules_unknown_fields_returns_none():
    info = GraphQLQueryInfo(
        root_fields={"unknownField", "anotherUnknownField"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is None


def test_determine_graphql_rules_test_field():
    info = GraphQLQueryInfo(
        root_fields={"test"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource == RESOURCE_EXPERIMENTS
    assert rules[0].verb == "get"


def test_determine_graphql_rules_nested_model_registry_access():
    info = GraphQLQueryInfo(
        root_fields={"mlflowGetRun"},
        has_nested_model_registry_access=True,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    # Should return TWO rules - experiments for the root field, models for nested access
    assert len(rules) == 2
    resources = {r.resource for r in rules}
    assert RESOURCE_EXPERIMENTS in resources
    assert RESOURCE_REGISTERED_MODELS in resources


def test_determine_graphql_rules_nested_model_access_with_list_verb():
    info = GraphQLQueryInfo(
        root_fields={"mlflowSearchRuns"},
        has_nested_model_registry_access=True,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    # Should return TWO rules - list for experiments, get for nested model access
    assert len(rules) == 2
    exp_rule = next(r for r in rules if r.resource == RESOURCE_EXPERIMENTS)
    model_rule = next(r for r in rules if r.resource == RESOURCE_REGISTERED_MODELS)
    assert exp_rule.verb == "list"
    assert model_rule.verb == "get"


def test_graphql_query_parsing_without_operation_name_experiments():
    query = '{ mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } } }'
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource == RESOURCE_EXPERIMENTS
    assert rules[0].verb == "get"


def test_graphql_query_parsing_without_operation_name_models():
    # This query requests modelVersions in the response which triggers nested detection too
    query = '{ mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } } }'
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    # Returns 2 rules: list (from root field) + get (from nested modelVersions)
    assert len(rules) == 2
    verbs = {r.verb for r in rules}
    assert "list" in verbs
    assert "get" in verbs
    for rule in rules:
        assert rule.resource == RESOURCE_REGISTERED_MODELS


def test_graphql_query_parsing_mixed_query():
    query = """
    {
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
        mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } }
    }
    """
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    # Mixed queries return 3 rules:
    # - experiments:get (from mlflowGetExperiment)
    # - registeredmodels:list (from mlflowSearchModelVersions)
    # - registeredmodels:get (from nested modelVersions field)
    assert rules is not None
    assert len(rules) == 3
    resources = {r.resource for r in rules}
    assert RESOURCE_EXPERIMENTS in resources
    assert RESOURCE_REGISTERED_MODELS in resources


def test_graphql_operation_name_does_not_bypass_query_parsing():
    """Test that operationName cannot be used to bypass query parsing (security fix).

    SECURITY: A malicious client could send operationName="GetRun" but include
    model registry fields in the query, attempting to bypass authorization checks.
    We must always parse the actual query to determine required permissions.
    """
    # Query contains model registry fields, but operationName claims it's an experiment query
    query = '{ mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } } }'
    payload = {"query": query, "operationName": "MlflowGetExperimentQuery"}
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload=payload)
    # Query parsing should be used, not operationName - requires registry permissions
    assert rules is not None
    assert len(rules) == 2  # model registry root + nested modelVersions
    resources = {rule.resource for rule in rules}
    assert RESOURCE_REGISTERED_MODELS in resources


def test_graphql_query_with_nested_model_versions_requires_both():
    # Query is for runs (experiment resource) but includes nested modelVersions
    query = """
    {
        mlflowGetRun(input: { runId: "abc" }) {
            run {
                info { runId }
                modelVersions { name version }
            }
        }
    }
    """
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    # Should require BOTH experiments and model registry access
    assert rules is not None
    assert len(rules) == 2
    resources = {r.resource for r in rules}
    assert RESOURCE_EXPERIMENTS in resources
    assert RESOURCE_REGISTERED_MODELS in resources


def test_graphql_search_runs_with_nested_model_versions():
    query = """
    {
        mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs {
                info { runId }
                modelVersions { name version }
            }
        }
    }
    """
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    # Should require BOTH experiments (list) and model registry (get for nested) access
    assert rules is not None
    assert len(rules) == 2
    exp_rule = next(r for r in rules if r.resource == RESOURCE_EXPERIMENTS)
    model_rule = next(r for r in rules if r.resource == RESOURCE_REGISTERED_MODELS)
    assert exp_rule.verb == "list"
    assert model_rule.verb == "get"


def test_graphql_nested_model_registry_fields_constant():
    assert "modelVersions" in GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS


def test_graphql_inline_fragment_hidden_model_registry_access():
    """Test that model registry fields inside inline fragments are detected (security fix).

    SECURITY: A malicious client could hide modelVersions inside an inline fragment
    to bypass authorization checks. The AST traversal must descend into inline fragments.
    """
    query_info = extract_graphql_query_info("""
    {
        mlflowGetRun(input: { runId: "123" }) {
            run {
                info { runId }
                ... on MlflowRun {
                    modelVersions { name version }
                }
            }
        }
    }
    """)
    assert query_info.root_fields == {"mlflowGetRun"}
    assert query_info.has_nested_model_registry_access is True


def test_graphql_fragment_spread_hidden_model_registry_access():
    """Test that model registry fields inside named fragments are detected (security fix).

    SECURITY: A malicious client could hide modelVersions inside a named fragment
    to bypass authorization checks. The AST traversal must resolve fragment spreads.
    """
    query_info = extract_graphql_query_info("""
    fragment RunWithModels on MlflowRun {
        info { runId }
        modelVersions { name version }
    }

    {
        mlflowGetRun(input: { runId: "123" }) {
            run {
                ...RunWithModels
            }
        }
    }
    """)
    assert query_info.root_fields == {"mlflowGetRun"}
    assert query_info.has_nested_model_registry_access is True


def test_graphql_deeply_nested_fragment_model_registry_access():
    query_info = extract_graphql_query_info("""
    fragment ModelInfo on MlflowRun {
        modelVersions { name }
    }

    fragment RunInfo on MlflowRun {
        info { runId }
        ...ModelInfo
    }

    {
        mlflowGetRun(input: { runId: "123" }) {
            run {
                ...RunInfo
            }
        }
    }
    """)
    assert query_info.root_fields == {"mlflowGetRun"}
    assert query_info.has_nested_model_registry_access is True


def test_graphql_fragment_without_model_registry_no_false_positive():
    query_info = extract_graphql_query_info("""
    fragment RunBasicInfo on MlflowRun {
        info { runId experimentId }
        data { metrics { key value } }
    }

    {
        mlflowGetRun(input: { runId: "123" }) {
            run {
                ...RunBasicInfo
            }
        }
    }
    """)
    assert query_info.root_fields == {"mlflowGetRun"}
    assert query_info.has_nested_model_registry_access is False


def test_validate_graphql_field_authorization_passes():
    validate_graphql_field_authorization()


def test_validate_graphql_field_authorization_detects_missing_resource(monkeypatch):
    original_map = GRAPHQL_FIELD_RESOURCE_MAP.copy()
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth_graphql.GRAPHQL_FIELD_RESOURCE_MAP",
        {k: v for k, v in original_map.items() if k != "test"},
    )
    with pytest.raises(MlflowException, match="Missing from GRAPHQL_FIELD_RESOURCE_MAP"):
        validate_graphql_field_authorization()


def test_validate_graphql_field_authorization_detects_missing_verb(monkeypatch):
    original_map = GRAPHQL_FIELD_VERB_MAP.copy()
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth_graphql.GRAPHQL_FIELD_VERB_MAP",
        {k: v for k, v in original_map.items() if k != "mlflowGetExperiment"},
    )
    with pytest.raises(MlflowException, match="Missing from GRAPHQL_FIELD_VERB_MAP"):
        validate_graphql_field_authorization()


def test_graphql_empty_query_returns_none():
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": ""})
    assert rules is None


def test_graphql_unparsable_query_returns_none():
    rules = _find_authorization_rules(
        "/graphql", "POST", graphql_payload={"query": "not valid graphql {{{"}
    )
    assert rules is None


def test_graphql_no_query_or_operation_name_returns_none():
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={})
    assert rules is None


def test_graphql_query_with_unknown_fields_returns_none():
    query = "{ unknownField { data } }"
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is None
