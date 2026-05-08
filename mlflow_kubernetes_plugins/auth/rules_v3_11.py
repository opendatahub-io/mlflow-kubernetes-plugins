"""MLflow 3.11 authorization deltas layered on top of the baseline tables."""

from __future__ import annotations

from mlflow_kubernetes_plugins.auth._compat import (
    BatchGetTraceInfos,
    CreateGatewayBudgetPolicy,
    CreateIssue,
    DeleteGatewayBudgetPolicy,
    GetGatewayBudgetPolicy,
    GetIssue,
    GetPresignedDownloadUrl,
    ListGatewayBudgetPolicies,
    ListGatewayBudgetWindows,
    SearchIssues,
    UpdateGatewayBudgetPolicy,
    UpdateIssue,
)
from mlflow_kubernetes_plugins.auth.collection_filters import (
    COLLECTION_POLICY_BROAD_ONLY,
    COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY,
    COLLECTION_POLICY_RESPONSE_TRACES,
)
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME,
    RESOURCE_NAME_PARSER_OPTIONAL_GATEWAY_ENDPOINT_NAME,
    RESOURCE_NAME_PARSER_OPTIONAL_GATEWAY_SECRET_ID_TO_NAME,
    RESOURCE_NAME_PARSER_OPTIONAL_TRACE_IDS_TO_EXPERIMENT_NAMES,
)
from mlflow_kubernetes_plugins.auth.rules import (
    AuthorizationRule,
    _experiments_rule,
    _gateway_budgets_rule,
    _gateway_endpoints_use_rule,
    _gateway_secrets_use_rule,
)

_GATEWAY_BUDGET_WINDOWS_DENIED_MESSAGE = (
    "Gateway budget window listing is disabled because the upstream MLflow endpoint "
    "returns tracker state without workspace filtering."
)


def apply_v3_11_deltas(
    *,
    request_authorization_rules: dict[type, AuthorizationRule | tuple[AuthorizationRule, ...]],
    path_authorization_rules: dict[
        tuple[str, str], AuthorizationRule | tuple[AuthorizationRule, ...]
    ],
) -> None:
    request_authorization_rules.update(
        {
            BatchGetTraceInfos: _experiments_rule(
                "list",
                collection_policy=COLLECTION_POLICY_RESPONSE_TRACES,
            ),
            GetPresignedDownloadUrl: _experiments_rule(
                "get",
                resource_name_parsers=(RESOURCE_NAME_PARSER_ARTIFACT_EXPERIMENT_ID_TO_NAME,),
            ),
            CreateIssue: _experiments_rule(
                "update",
                resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
            ),
            GetIssue: _experiments_rule(
                "get",
                resource_name_parsers=(RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME,),
            ),
            UpdateIssue: _experiments_rule(
                "update",
                resource_name_parsers=(RESOURCE_NAME_PARSER_ISSUE_ID_TO_EXPERIMENT_NAME,),
            ),
            SearchIssues: _experiments_rule(
                "list",
                collection_policy=COLLECTION_POLICY_REQUEST_EXPERIMENT_ID_BODY,
            ),
            # Budget policies intentionally remain workspace-scoped in this plugin.
            # MLflow exposes only opaque budget_policy_id values, not a declarative unique name that
            # could be pre-provisioned through GitOps-friendly RBAC resourceNames, and the extra
            # granularity would add little operational value for budgets.
            CreateGatewayBudgetPolicy: _gateway_budgets_rule("create"),
            GetGatewayBudgetPolicy: _gateway_budgets_rule("get"),
            UpdateGatewayBudgetPolicy: _gateway_budgets_rule("update"),
            DeleteGatewayBudgetPolicy: _gateway_budgets_rule("delete"),
            ListGatewayBudgetPolicies: _gateway_budgets_rule(
                "list",
                collection_policy=COLLECTION_POLICY_BROAD_ONLY,
            ),
            # Fail closed here until upstream makes the tracker-backed endpoint workspace-aware.
            ListGatewayBudgetWindows: _gateway_budgets_rule(
                "list",
                deny=True,
                deny_message=_GATEWAY_BUDGET_WINDOWS_DENIED_MESSAGE,
            ),
        }
    )
    issue_invoke_rules = (
        _experiments_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_EXPERIMENT_ID_TO_NAME,),
        ),
        _experiments_rule(
            "get",
            resource_name_parsers=(RESOURCE_NAME_PARSER_OPTIONAL_TRACE_IDS_TO_EXPERIMENT_NAMES,),
            allow_if_resource_reference_missing=True,
        ),
        _gateway_secrets_use_rule(
            resource_name_parsers=(RESOURCE_NAME_PARSER_OPTIONAL_GATEWAY_SECRET_ID_TO_NAME,),
            allow_if_resource_reference_missing=True,
        ),
        _gateway_endpoints_use_rule(
            resource_name_parsers=(RESOURCE_NAME_PARSER_OPTIONAL_GATEWAY_ENDPOINT_NAME,),
            allow_if_resource_reference_missing=True,
        ),
    )
    path_authorization_rules.update(
        {
            ("/ajax-api/3.0/mlflow/issues/invoke", "POST"): issue_invoke_rules,
            ("/ajax-api/3.0/mlflow/issues/invoke/", "POST"): issue_invoke_rules,
        }
    )
