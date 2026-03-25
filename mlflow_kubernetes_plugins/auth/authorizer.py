"""Kubernetes authorization config, cache, and access-review client logic."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, NamedTuple

from kubernetes import client, config
from kubernetes.client import AuthorizationV1Api
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2

from mlflow_kubernetes_plugins.auth.constants import (
    AUTHORIZATION_MODE_ENV,
    CACHE_TTL_ENV,
    DEFAULT_AUTH_GROUP,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_REMOTE_GROUPS_HEADER,
    DEFAULT_REMOTE_GROUPS_SEPARATOR,
    DEFAULT_REMOTE_USER_HEADER,
    DEFAULT_USERNAME_CLAIM,
    REMOTE_GROUPS_HEADER_ENV,
    REMOTE_GROUPS_SEPARATOR_ENV,
    REMOTE_USER_HEADER_ENV,
    USERNAME_CLAIM_ENV,
    WORKSPACE_PERMISSION_RESOURCE_PRIORITY,
)

if TYPE_CHECKING:
    from mlflow_kubernetes_plugins.auth.core import _RequestIdentity

_logger = logging.getLogger(__name__)


class AuthorizationMode(str, Enum):
    SELF_SUBJECT_ACCESS_REVIEW = "self_subject_access_review"
    SUBJECT_ACCESS_REVIEW = "subject_access_review"


class _ReadWriteLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        with self._condition:
            self._writer = False
            self._condition.notify_all()


class _CacheEntry(NamedTuple):
    allowed: bool
    expires_at: float


_AuthorizationCacheKey = tuple[str, str, str, str | None, str]


class _AuthorizationCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[_AuthorizationCacheKey, _CacheEntry] = {}
        self._lock = _ReadWriteLock()

    def get(self, key: _AuthorizationCacheKey) -> bool | None:
        observed_expiration: float | None = None

        self._lock.acquire_read()
        try:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at > time.time():
                return entry.allowed
            observed_expiration = entry.expires_at
        finally:
            self._lock.release_read()

        self._lock.acquire_write()
        try:
            current = self._entries.get(key)
            if current is not None:
                if observed_expiration is None or current.expires_at <= observed_expiration:
                    self._entries.pop(key, None)
        finally:
            self._lock.release_write()
        return None

    def set(self, key: _AuthorizationCacheKey, allowed: bool) -> None:
        self._lock.acquire_write()
        try:
            self._entries[key] = _CacheEntry(
                allowed=allowed, expires_at=time.time() + self._ttl_seconds
            )
        finally:
            self._lock.release_write()


def _load_kubernetes_configuration() -> client.Configuration:
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    try:
        return client.Configuration.get_default_copy()
    except AttributeError:  # pragma: no cover - fallback for older client versions
        return client.Configuration()


def _create_api_client_for_subject_access_reviews() -> client.ApiClient:
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException as exc:  # pragma: no cover - depends on env
            raise MlflowException(
                "Failed to load Kubernetes configuration for authorization plugin",
                error_code=databricks_pb2.INVALID_STATE,
            ) from exc
    return client.ApiClient()


class KubernetesAuthorizer:
    def __init__(
        self,
        config_values: "KubernetesAuthConfig",
        group: str = DEFAULT_AUTH_GROUP,
    ) -> None:
        self._group = group
        self._cache = _AuthorizationCache(config_values.cache_ttl_seconds)
        self._mode = config_values.authorization_mode
        self._base_configuration: client.Configuration | None = None
        self._sar_api_client: client.ApiClient | None = None
        self._user_header_label = (
            f"Header '{config_values.user_header}'"
            if config_values.user_header
            else "Remote user header"
        )

        if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
            self._base_configuration = _load_kubernetes_configuration()
        else:
            self._sar_api_client = _create_api_client_for_subject_access_reviews()

    def _build_api_client_with_token(self, token: str) -> client.ApiClient:
        if self._base_configuration is None:
            raise MlflowException(
                "SelfSubjectAccessReview mode is not initialized with base configuration.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        base = self._base_configuration
        configuration = client.Configuration()
        # Make a copy of the Kubernetes client without credential information
        configuration.host = base.host
        configuration.ssl_ca_cert = base.ssl_ca_cert
        configuration.verify_ssl = base.verify_ssl
        configuration.proxy = base.proxy
        configuration.no_proxy = base.no_proxy
        configuration.proxy_headers = base.proxy_headers
        configuration.safe_chars_for_path_param = base.safe_chars_for_path_param
        configuration.connection_pool_maxsize = base.connection_pool_maxsize
        configuration.assert_hostname = getattr(base, "assert_hostname", None)
        configuration.retries = getattr(base, "retries", None)
        configuration.cert_file = None
        configuration.key_file = None
        configuration.username = None
        configuration.password = None
        configuration.refresh_api_key_hook = None
        configuration.api_key = {"authorization": token}
        configuration.api_key_prefix = {"authorization": "Bearer"}
        return client.ApiClient(configuration)

    def _submit_self_subject_access_review(
        self,
        token: str,
        resource: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        body = client.V1SelfSubjectAccessReview(
            spec=client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                    subresource=subresource,
                )
            )
        )

        api_client = self._build_api_client_with_token(token)
        try:
            authorization_api = AuthorizationV1Api(api_client)
            response = authorization_api.create_self_subject_access_review(body)  # type: ignore[call-arg]
        finally:
            api_client.close()

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SelfSubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def _submit_subject_access_review(
        self,
        user: str,
        groups: tuple[str, ...],
        resource: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        body = client.V1SubjectAccessReview(
            spec=client.V1SubjectAccessReviewSpec(
                user=user,
                groups=list(groups) if groups else None,
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                    subresource=subresource,
                ),
            )
        )

        if self._sar_api_client is None:
            raise MlflowException(
                "SubjectAccessReview mode requires a Kubernetes client.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )

        authorization_api = AuthorizationV1Api(self._sar_api_client)
        response = authorization_api.create_subject_access_review(body)  # type: ignore[call-arg]

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def is_allowed(
        self,
        identity: "_RequestIdentity",
        resource_type: str,
        verb: str,
        namespace: str,
        subresource: str | None = None,
    ) -> bool:
        resource = resource_type.replace("_", "")
        identity_hash = identity.subject_hash(
            self._mode, missing_user_label=self._user_header_label
        )
        cache_key = (identity_hash, namespace, resource, subresource, verb)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                allowed = self._submit_self_subject_access_review(
                    identity.token or "", resource, verb, namespace, subresource
                )
            else:
                allowed = self._submit_subject_access_review(
                    identity.user or "",
                    identity.groups,
                    resource,
                    verb,
                    namespace,
                    subresource,
                )
        except ApiException as exc:  # pragma: no cover - depends on live cluster
            if exc.status == 401:
                raise MlflowException(
                    "Authentication with the Kubernetes API failed. The provided token may be "
                    "invalid or expired.",
                    error_code=databricks_pb2.UNAUTHENTICATED,
                ) from exc
            if exc.status == 403:
                if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SelfSubjectAccessReview. Grant the required authorization."
                    )
                else:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SubjectAccessReview. Grant 'create' on subjectaccessreviews."
                    )
                raise MlflowException(
                    message,
                    error_code=databricks_pb2.PERMISSION_DENIED,
                ) from exc
            raise MlflowException(
                f"Failed to perform Kubernetes authorization check: {exc}",
                error_code=databricks_pb2.INTERNAL_ERROR,
            ) from exc

        self._cache.set(cache_key, allowed)
        _logger.debug(
            "Access review evaluated subject_hash=%s resource=%s subresource=%s namespace=%s "
            + "verb=%s allowed=%s",
            identity_hash,
            resource,
            subresource,
            namespace,
            verb,
            allowed,
        )
        return allowed

    def accessible_workspaces(self, identity: "_RequestIdentity", names: Iterable[str]) -> set[str]:
        accessible: set[str] = set()
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for workspace_name in names:
            if self.can_access_workspace(identity, workspace_name, verb="list"):
                accessible.add(workspace_name)
            else:
                _logger.debug(
                    "Workspace %s excluded for subject_hash=%s; no list permission detected",
                    workspace_name,
                    subject_hash,
                )
        return accessible

    def can_access_workspace(
        self, identity: "_RequestIdentity", workspace_name: str, verb: str = "get"
    ) -> bool:
        """Check if the identity can access the workspace via any priority resource."""
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for resource in WORKSPACE_PERMISSION_RESOURCE_PRIORITY:
            if self.is_allowed(identity, resource, verb, workspace_name):
                _logger.debug(
                    "Workspace %s accessible for subject_hash=%s via resource=%s verb=%s",
                    workspace_name,
                    subject_hash,
                    resource,
                    verb,
                )
                return True
        return False


@dataclass(frozen=True)
class KubernetesAuthConfig:
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    username_claim: str = DEFAULT_USERNAME_CLAIM
    authorization_mode: AuthorizationMode = AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW
    user_header: str = DEFAULT_REMOTE_USER_HEADER
    groups_header: str = DEFAULT_REMOTE_GROUPS_HEADER
    groups_separator: str = DEFAULT_REMOTE_GROUPS_SEPARATOR

    @classmethod
    def from_env(cls) -> "KubernetesAuthConfig":
        ttl_env = os.environ.get(CACHE_TTL_ENV)
        username_claim = os.environ.get(USERNAME_CLAIM_ENV, DEFAULT_USERNAME_CLAIM)
        mode_env = os.environ.get(
            AUTHORIZATION_MODE_ENV, AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW.value
        )
        user_header = os.environ.get(REMOTE_USER_HEADER_ENV, DEFAULT_REMOTE_USER_HEADER)
        groups_header = os.environ.get(REMOTE_GROUPS_HEADER_ENV, DEFAULT_REMOTE_GROUPS_HEADER)
        groups_separator = os.environ.get(
            REMOTE_GROUPS_SEPARATOR_ENV, DEFAULT_REMOTE_GROUPS_SEPARATOR
        )

        cache_ttl_seconds = DEFAULT_CACHE_TTL_SECONDS
        if ttl_env:
            try:
                cache_ttl_seconds = float(ttl_env)
                if cache_ttl_seconds <= 0:
                    raise ValueError
            except ValueError as exc:
                raise MlflowException(
                    f"Environment variable {CACHE_TTL_ENV} must be a positive number if set",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                ) from exc

        try:
            authorization_mode = AuthorizationMode(mode_env.strip().lower())
        except ValueError as exc:
            valid_modes = ", ".join(mode.value for mode in AuthorizationMode)
            raise MlflowException(
                f"Environment variable {AUTHORIZATION_MODE_ENV} must be one of: {valid_modes}",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            ) from exc

        user_header = user_header.strip()
        groups_header = groups_header.strip()
        if not user_header:
            raise MlflowException(
                f"Environment variable {REMOTE_USER_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        if not groups_header:
            raise MlflowException(
                f"Environment variable {REMOTE_GROUPS_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        return cls(
            cache_ttl_seconds=cache_ttl_seconds,
            username_claim=username_claim,
            authorization_mode=authorization_mode,
            user_header=user_header,
            groups_header=groups_header,
            groups_separator=groups_separator or DEFAULT_REMOTE_GROUPS_SEPARATOR,
        )


__all__ = [
    "AuthorizationMode",
    "KubernetesAuthConfig",
    "KubernetesAuthorizer",
    "_AuthorizationCache",
    "_CacheEntry",
    "_create_api_client_for_subject_access_reviews",
    "_load_kubernetes_configuration",
]
