from types import SimpleNamespace
from unittest.mock import Mock

import mlflow_kubernetes_plugins.auth.resource_names as resource_names_mod
import pytest
from flask import Flask
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import CreateRun
from mlflow.utils import workspace_context
from mlflow_kubernetes_plugins.auth.authorizer import KubernetesAuthConfig
from mlflow_kubernetes_plugins.auth.collection_filters import COLLECTION_POLICY_GRAPHQL_FILTER
from mlflow_kubernetes_plugins.auth.compiler import _find_authorization_rules
from mlflow_kubernetes_plugins.auth.core import (
    _AUTHORIZATION_HANDLED,
    _RequestIdentity,
)
from mlflow_kubernetes_plugins.auth.graphql import (
    GRAPHQL_FIELD_AUTH_POLICY_MAP,
    GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER,
    GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER,
    GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT,
    GRAPHQL_FIELD_RESOURCE_MAP,
    GRAPHQL_FIELD_VERB_MAP,
    GRAPHQL_NESTED_MODEL_REGISTRY_FIELDS,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
    GraphQLQueryInfo,
    GraphQLRootField,
    KubernetesGraphQLAuthorizationMiddleware,
    determine_graphql_rules,
    extract_graphql_query_info,
    validate_graphql_field_authorization,
)
from mlflow_kubernetes_plugins.auth.request_context import AuthorizationRequest
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES,
)
from mlflow_kubernetes_plugins.auth.rules import GRAPHQL_OPERATION_RULES, AuthorizationRule

from conftest import _authorize_request


@pytest.fixture(autouse=True)
def _compile_rules(compile_auth_rules):
    """Ensure authorization rules are populated before each test."""
    compile_auth_rules(
        [
            ("/api/2.0/mlflow/runs/create", CreateRun, ["POST"]),
        ]
    )


def _invalidate_experiment_lookup_cache(experiment_id: str) -> None:
    resource_names_mod._experiment_name_cache.invalidate(experiment_id)


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
    assert set(GRAPHQL_FIELD_RESOURCE_MAP) == set(GRAPHQL_FIELD_AUTH_POLICY_MAP)
    assert (
        GRAPHQL_FIELD_AUTH_POLICY_MAP["mlflowGetExperiment"]
        == GRAPHQL_FIELD_AUTH_POLICY_SINGLE_OBJECT
    )
    assert (
        GRAPHQL_FIELD_AUTH_POLICY_MAP["mlflowSearchRuns"]
        == GRAPHQL_FIELD_AUTH_POLICY_REQUEST_FILTER
    )
    assert (
        GRAPHQL_FIELD_AUTH_POLICY_MAP["mlflowSearchModelVersions"]
        == GRAPHQL_FIELD_AUTH_POLICY_RESPONSE_FILTER
    )


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


def test_extract_graphql_query_info_resolves_variable_inputs():
    query = """
    query GetRun($data: MlflowGetRunInput!) {
        mlflowGetRun(input: $data) {
            run { info { runId } }
        }
    }
    """
    info = extract_graphql_query_info(query, {"data": {"runId": "run-123"}})
    assert info.root_field_inputs == (
        GraphQLRootField(
            field_name="mlflowGetRun",
            args={"input": {"runId": "run-123"}},
        ),
    )


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


def test_determine_graphql_rules_single_root_field_adds_name_parser():
    info = extract_graphql_query_info(
        '{ mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } } }'
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,
    )


def test_determine_graphql_rules_multiple_single_object_fields_preserve_all_name_parsers():
    info = extract_graphql_query_info("""
    {
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
        mlflowGetRun(input: { runId: "run-123" }) { run { info { runId } } }
    }
    """)
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,
        RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,
    )


def test_determine_graphql_rules_search_runs_uses_collection_policy():
    info = extract_graphql_query_info(
        '{ mlflowSearchRuns(input: { experimentIds: ["1"] }) { runs { info { runId } } } }'
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].collection_policy == COLLECTION_POLICY_GRAPHQL_FILTER
    assert rules[0].resource_name_parsers == ()


def test_determine_graphql_rules_search_runs_without_experiment_ids_stays_broad_only():
    info = extract_graphql_query_info(
        '{ mlflowSearchRuns(input: { filter: "" }) { runs { info { runId } } } }'
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].collection_policy is None


def test_determine_graphql_rules_search_runs_with_multiple_scoped_aliases_uses_collection_policy():
    info = extract_graphql_query_info("""
    {
        first: mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs { info { runId } }
        }
        second: mlflowSearchRuns(input: { experimentIds: ["2"] }) {
            runs { info { runId } }
        }
    }
    """)
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].collection_policy == COLLECTION_POLICY_GRAPHQL_FILTER


def test_determine_graphql_rules_alias_mixture_without_experiment_ids_stays_broad_only():
    info = extract_graphql_query_info("""
    {
        scoped: mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs { info { runId } }
        }
        unscoped: mlflowSearchRuns(input: { filter: "" }) {
            runs { info { runId } }
        }
    }
    """)
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].collection_policy is None


def test_determine_graphql_rules_shared_experiment_list_rule_requires_all_fields_filterable():
    info = extract_graphql_query_info("""
    {
        runs: mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs { info { runId } }
        }
        datasets: mlflowSearchDatasets(input: { filter: "" }) {
            datasets { dataset { name } }
        }
    }
    """)
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].collection_policy is None


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


def test_determine_graphql_rules_keeps_broad_and_single_object_policies_separate():
    info = GraphQLQueryInfo(
        root_fields={"mlflowGetExperiment", "test"},
        has_nested_model_registry_access=False,
    )
    rules = determine_graphql_rules(info, AuthorizationRule)
    assert rules is not None
    assert len(rules) == 2
    broad_rule = next(
        rule for rule in rules if rule.resource == RESOURCE_EXPERIMENTS and rule.resource_name_parsers == ()
    )
    scoped_rule = next(
        rule
        for rule in rules
        if rule.resource == RESOURCE_EXPERIMENTS
        and rule.resource_name_parsers == (RESOURCE_NAME_PARSER_GRAPHQL_EXPERIMENT_ID_TO_NAME,)
    )
    assert broad_rule.verb == "get"
    assert broad_rule.collection_policy is None
    assert scoped_rule.verb == "get"
    assert scoped_rule.collection_policy is None


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
        nested_model_registry_root_fields=frozenset({"mlflowGetRun"}),
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
        nested_model_registry_root_fields=frozenset({"mlflowSearchRuns"}),
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


def test_graphql_query_parsing_run_query_includes_name_parser():
    query = '{ mlflowGetRun(input: { runId: "run-123" }) { run { info { runId } } } }'
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,
    )


def test_graphql_query_parsing_list_artifacts_includes_name_parser():
    query = '{ mlflowListArtifacts(input: { runId: "run-123" }) { files { path } } }'
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GRAPHQL_RUN_ID_TO_EXPERIMENT_NAME,
    )


def test_graphql_query_parsing_metric_history_bulk_interval_includes_name_parser():
    query = """
    {
        mlflowGetMetricHistoryBulkInterval(input: { runIds: ["run-1"], metricKey: "loss" }) {
            metrics { key value }
        }
    }
    """
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource_name_parsers == (
        RESOURCE_NAME_PARSER_GRAPHQL_RUN_IDS_TO_EXPERIMENT_NAMES,
    )


def test_graphql_query_parsing_without_operation_name_models():
    # This query requests modelVersions in the response which triggers nested detection too
    query = '{ mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } } }'
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    assert rules is not None
    assert len(rules) == 1
    assert rules[0].resource == RESOURCE_REGISTERED_MODELS
    assert rules[0].verb == "list"
    assert rules[0].collection_policy == COLLECTION_POLICY_GRAPHQL_FILTER


def test_graphql_query_parsing_mixed_query():
    query = """
    {
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
        mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } }
    }
    """
    rules = _find_authorization_rules("/graphql", "POST", graphql_payload={"query": query})
    # Mixed queries return 2 rules:
    # - experiments:get (from mlflowGetExperiment)
    # - registeredmodels:list (from mlflowSearchModelVersions, filtered at response time)
    assert rules is not None
    assert len(rules) == 2
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
    assert len(rules) == 1
    resources = {rule.resource for rule in rules}
    assert RESOURCE_REGISTERED_MODELS in resources


def test_authorize_request_graphql_search_model_versions_defers_to_response_filter(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    query = '{ mlflowSearchModelVersions(input: { filter: "" }) { modelVersions { name } } }'
    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query},
            graphql_payload={"query": query},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert len(result.rules) == 1
    assert result.rules[0].resource == RESOURCE_REGISTERED_MODELS
    assert result.rules[0].verb == "list"
    assert result.rules[0].collection_policy == COLLECTION_POLICY_GRAPHQL_FILTER
    assert authorizer.is_allowed.call_count == 1


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


def test_authorize_request_graphql_retries_with_experiment_name(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_experiment=lambda experiment_id: SimpleNamespace(name="exp-a")),
    )

    query = '{ mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } } }'
    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query},
            graphql_payload={"query": query},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.kwargs == {"resource_name": "exp-a"}


def test_authorize_request_graphql_resolves_variables_for_run_queries(monkeypatch):
    _invalidate_experiment_lookup_cache("456")
    resource_names_mod._run_experiment_name_cache.invalidate("run-123")
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True]
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="456")),
            get_experiment=lambda experiment_id: SimpleNamespace(name="exp-from-run"),
        ),
    )

    query = """
    query GetRun($data: MlflowGetRunInput!) {
        mlflowGetRun(input: $data) {
            run { info { runId } }
        }
    }
    """
    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query, "variables": {"data": {"runId": "run-123"}}},
            graphql_payload={"query": query, "variables": {"data": {"runId": "run-123"}}},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    second_call = authorizer.is_allowed.call_args_list[1]
    assert second_call.kwargs == {"resource_name": "exp-from-run"}


def test_authorize_request_graphql_mixed_queries_retry_each_single_object(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True, True]
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_experiment=lambda experiment_id: SimpleNamespace(
                name={"123": "exp-a", "exp-456": "exp-from-run"}[experiment_id]
            ),
            get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="exp-456")),
        ),
    )

    query = """
    {
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
        mlflowGetRun(input: { runId: "run-123" }) { run { info { runId } } }
    }
    """
    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query},
            graphql_payload={"query": query},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert authorizer.is_allowed.call_count == 3
    assert authorizer.is_allowed.call_args_list[0].kwargs == {}
    assert authorizer.is_allowed.call_args_list[1].kwargs == {"resource_name": "exp-a"}
    assert authorizer.is_allowed.call_args_list[2].kwargs == {"resource_name": "exp-from-run"}


def test_authorize_request_graphql_mixed_broad_and_single_object_denies_without_broad_access(
    monkeypatch,
):
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(get_experiment=lambda experiment_id: SimpleNamespace(name="exp-a")),
    )

    query = """
    {
        test(input: "hello")
        mlflowGetExperiment(input: { experimentId: "123" }) { experiment { name } }
    }
    """
    with pytest.raises(MlflowException, match="Permission denied"):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer valid-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/graphql",
                method="POST",
                workspace="team-a",
                json_body={"query": query},
                graphql_payload={"query": query},
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_count == 1


def test_authorize_request_graphql_aliased_run_queries_retry_each_alias(monkeypatch):
    _invalidate_experiment_lookup_cache("exp-1")
    _invalidate_experiment_lookup_cache("exp-2")
    resource_names_mod._run_experiment_name_cache.invalidate("run-1")
    resource_names_mod._run_experiment_name_cache.invalidate("run-2")
    authorizer = Mock()
    authorizer.is_allowed.side_effect = [False, True, True]
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.resource_names._get_tracking_store",
        lambda: SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(
                info=SimpleNamespace(experiment_id={"run-1": "exp-1", "run-2": "exp-2"}[run_id])
            ),
            get_experiment=lambda experiment_id: SimpleNamespace(
                name={"exp-1": "exp-a", "exp-2": "exp-b"}[experiment_id]
            ),
        ),
    )

    query = """
    {
        first: mlflowGetRun(input: { runId: "run-1" }) { run { info { runId } } }
        second: mlflowGetRun(input: { runId: "run-2" }) { run { info { runId } } }
    }
    """
    _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query},
            graphql_payload={"query": query},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert authorizer.is_allowed.call_count == 3
    assert authorizer.is_allowed.call_args_list[1].kwargs == {"resource_name": "exp-a"}
    assert authorizer.is_allowed.call_args_list[2].kwargs == {"resource_name": "exp-b"}


def test_authorize_request_graphql_search_runs_defers_to_collection_filter(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    query = """
    {
        mlflowSearchRuns(input: { experimentIds: ["1", "2"] }) {
            runs { info { runId } }
        }
    }
    """
    result = _authorize_request(
        AuthorizationRequest(
            authorization_header="Bearer valid-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/graphql",
            method="POST",
            workspace="team-a",
            json_body={"query": query},
            graphql_payload={"query": query},
        ),
        authorizer=authorizer,
        config_values=KubernetesAuthConfig(),
    )

    assert result.rules[0].collection_policy == COLLECTION_POLICY_GRAPHQL_FILTER
    assert authorizer.is_allowed.call_count == 1


def test_authorize_request_graphql_search_runs_without_experiment_ids_denies(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    query = """
    {
        mlflowSearchRuns(input: { filter: "" }) {
            runs { info { runId } }
        }
    }
    """
    with pytest.raises(MlflowException, match="Permission denied"):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer valid-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/graphql",
                method="POST",
                workspace="team-a",
                json_body={"query": query},
                graphql_payload={"query": query},
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_count == 1


def test_authorize_request_graphql_alias_mixture_without_experiment_ids_denies(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = False
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.core._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    query = """
    {
        scoped: mlflowSearchRuns(input: { experimentIds: ["1"] }) {
            runs { info { runId } }
        }
        unscoped: mlflowSearchRuns(input: { filter: "" }) {
            runs { info { runId } }
        }
    }
    """
    with pytest.raises(MlflowException, match="Permission denied"):
        _authorize_request(
            AuthorizationRequest(
                authorization_header="Bearer valid-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/graphql",
                method="POST",
                workspace="team-a",
                json_body={"query": query},
                graphql_payload={"query": query},
            ),
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
        )

    assert authorizer.is_allowed.call_count == 1


def test_kubernetes_graphql_middleware_filters_search_runs_input(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {"1": "exp-a", "2": "exp-b"}[experiment_id],
    )
    middleware = KubernetesGraphQLAuthorizationMiddleware(authorizer)

    with app.test_request_context("/graphql"):
        token = _AUTHORIZATION_HANDLED.set(SimpleNamespace(identity=_RequestIdentity(token="token")))
        workspace_context.set_server_request_workspace("team-a")
        try:
            input_obj = SimpleNamespace(experiment_ids=["1", "2"])
            info = SimpleNamespace(field_name="mlflowSearchRuns")
            result = middleware.resolve(
                lambda _root, _info, **kwargs: list(kwargs["input"].experiment_ids),
                None,
                info,
                input=input_obj,
            )
        finally:
            _AUTHORIZATION_HANDLED.reset(token)
            workspace_context.clear_server_request_workspace()

    assert result == ["1"]


def test_kubernetes_graphql_middleware_filters_search_datasets_input(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {"1": "exp-a", "2": "exp-b"}[experiment_id],
    )
    middleware = KubernetesGraphQLAuthorizationMiddleware(authorizer)

    with app.test_request_context("/graphql"):
        token = _AUTHORIZATION_HANDLED.set(SimpleNamespace(identity=_RequestIdentity(token="token")))
        workspace_context.set_server_request_workspace("team-a")
        try:
            input_obj = SimpleNamespace(experiment_ids=["1", "2"])
            info = SimpleNamespace(field_name="mlflowSearchDatasets")
            result = middleware.resolve(
                lambda _root, _info, **kwargs: list(kwargs["input"].experiment_ids),
                None,
                info,
                input=input_obj,
            )
        finally:
            _AUTHORIZATION_HANDLED.reset(token)
            workspace_context.clear_server_request_workspace()

    assert result == ["1"]


def test_kubernetes_graphql_middleware_filters_search_runs_dict_input(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "exp-a"
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.collection_filters._resolve_experiment_name_from_experiment_id",
        lambda experiment_id: {"1": "exp-a", "2": "exp-b"}[experiment_id],
    )
    middleware = KubernetesGraphQLAuthorizationMiddleware(authorizer)

    with app.test_request_context("/graphql"):
        token = _AUTHORIZATION_HANDLED.set(SimpleNamespace(identity=_RequestIdentity(token="token")))
        workspace_context.set_server_request_workspace("team-a")
        try:
            input_obj = {"experimentIds": ["1", "2"]}
            info = SimpleNamespace(field_name="mlflowSearchRuns")
            result = middleware.resolve(
                lambda _root, _info, **kwargs: list(kwargs["input"]["experimentIds"]),
                None,
                info,
                input=input_obj,
            )
        finally:
            _AUTHORIZATION_HANDLED.reset(token)
            workspace_context.clear_server_request_workspace()

    assert result == ["1"]


def test_kubernetes_graphql_middleware_filters_model_versions_response():
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.is_allowed.side_effect = lambda *args, **kwargs: kwargs.get("resource_name") == "model-a"
    middleware = KubernetesGraphQLAuthorizationMiddleware(authorizer)
    result_obj = SimpleNamespace(
        model_versions=[SimpleNamespace(name="model-a"), SimpleNamespace(name="model-b")]
    )

    with app.test_request_context("/graphql"):
        token = _AUTHORIZATION_HANDLED.set(SimpleNamespace(identity=_RequestIdentity(token="token")))
        workspace_context.set_server_request_workspace("team-a")
        try:
            info = SimpleNamespace(field_name="mlflowSearchModelVersions")
            filtered = middleware.resolve(
                lambda _root, _info, **_kwargs: result_obj,
                None,
                info,
                input=SimpleNamespace(filter=""),
            )
        finally:
            _AUTHORIZATION_HANDLED.reset(token)
            workspace_context.clear_server_request_workspace()

    assert [model.name for model in filtered.model_versions] == ["model-a"]


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
        "mlflow_kubernetes_plugins.auth.graphql.GRAPHQL_FIELD_RESOURCE_MAP",
        {k: v for k, v in original_map.items() if k != "test"},
    )
    with pytest.raises(MlflowException, match="Missing from GRAPHQL_FIELD_RESOURCE_MAP"):
        validate_graphql_field_authorization()


def test_validate_graphql_field_authorization_detects_missing_verb(monkeypatch):
    original_map = GRAPHQL_FIELD_VERB_MAP.copy()
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.graphql.GRAPHQL_FIELD_VERB_MAP",
        {k: v for k, v in original_map.items() if k != "mlflowGetExperiment"},
    )
    with pytest.raises(MlflowException, match="Missing from GRAPHQL_FIELD_VERB_MAP"):
        validate_graphql_field_authorization()


def test_validate_graphql_field_authorization_detects_missing_policy(monkeypatch):
    original_map = GRAPHQL_FIELD_AUTH_POLICY_MAP.copy()
    monkeypatch.setattr(
        "mlflow_kubernetes_plugins.auth.graphql.GRAPHQL_FIELD_AUTH_POLICY_MAP",
        {k: v for k, v in original_map.items() if k != "mlflowSearchRuns"},
    )
    with pytest.raises(MlflowException, match="Missing from GRAPHQL_FIELD_AUTH_POLICY_MAP"):
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
