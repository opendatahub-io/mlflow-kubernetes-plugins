"""MLflow 3.12 authorization deltas layered on top of the earlier tables."""

from __future__ import annotations

from mlflow_kubernetes_plugins.auth._compat import (
    AddGuardrailToEndpoint,
    CreateGatewayGuardrail,
    CreatePresignedUploadUrl,
    DeleteGatewayGuardrail,
    GetGatewayGuardrail,
    ListEndpointGuardrailConfigs,
    ListGatewayGuardrails,
    RemoveGuardrailFromEndpoint,
    UpdateEndpointGuardrailConfig,
)
from mlflow_kubernetes_plugins.auth.resource_names import (
    RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_OPTIONAL_ACTION_ENDPOINT_ID_TO_NAME,
    RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,
)
from mlflow_kubernetes_plugins.auth.rules import (
    AuthorizationRule,
    _experiments_rule,
    _gateway_endpoints_rule,
    _gateway_endpoints_use_rule,
    _gateway_guardrails_rule,
)


def apply_v3_12_deltas(
    *,
    request_authorization_rules: dict[type, AuthorizationRule | tuple[AuthorizationRule, ...]],
    path_authorization_rules: dict[
        tuple[str, str], AuthorizationRule | tuple[AuthorizationRule, ...]
    ],
) -> None:
    del path_authorization_rules

    create_guardrail_rules = (
        _gateway_guardrails_rule("create"),
        _gateway_endpoints_use_rule(
            resource_name_parsers=(RESOURCE_NAME_PARSER_OPTIONAL_ACTION_ENDPOINT_ID_TO_NAME,),
            allow_if_resource_reference_missing=True,
        ),
    )
    update_endpoint_guardrail_rules = (
        _gateway_endpoints_rule(
            "update",
            resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
            # These operations mutate the endpoint/guardrail association, not the endpoint's
            # model-definition dependencies, so they should not inherit endpoint dependency checks.
            skip_gateway_dependency_permissions=True,
        ),
        _gateway_guardrails_rule("update"),
    )
    list_endpoint_guardrail_rules = (
        _gateway_endpoints_rule(
            "get",
            resource_name_parsers=(RESOURCE_NAME_PARSER_GATEWAY_ENDPOINT_ID_TO_NAME,),
        ),
        _gateway_guardrails_rule("list"),
    )

    request_authorization_rules.update(
        {
            CreatePresignedUploadUrl: _experiments_rule(
                "update",
                resource_name_parsers=(RESOURCE_NAME_PARSER_RUN_ID_TO_EXPERIMENT_NAME,),
            ),
            CreateGatewayGuardrail: create_guardrail_rules,
            GetGatewayGuardrail: _gateway_guardrails_rule("get"),
            DeleteGatewayGuardrail: _gateway_guardrails_rule("delete"),
            ListGatewayGuardrails: _gateway_guardrails_rule("list"),
            AddGuardrailToEndpoint: update_endpoint_guardrail_rules,
            RemoveGuardrailFromEndpoint: update_endpoint_guardrail_rules,
            ListEndpointGuardrailConfigs: list_endpoint_guardrail_rules,
            UpdateEndpointGuardrailConfig: update_endpoint_guardrail_rules,
        }
    )
