from __future__ import annotations

import atexit
import base64
import binascii
import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase

from kubernetes import watch
from kubernetes.client import CoreV1Api, CustomObjectsApi
from kubernetes.client.exceptions import ApiException
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION_ANNOTATION = "mlflow.kubeflow.org/workspace-description"

# MLflowConfig CRD constants
MLFLOW_CONFIG_GROUP = "mlflow.kubeflow.org"
MLFLOW_CONFIG_VERSION = "v1"
MLFLOW_CONFIG_PLURAL = "mlflowconfigs"

ARTIFACT_CONNECTION_SECRET_NAME = "mlflow-artifact-connection"
_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
_LIVE_CACHES: "weakref.WeakSet[object]" = weakref.WeakSet()
_LIVE_CACHES_LOCK = threading.Lock()


def _register_cache_for_shutdown(cache: object) -> None:
    with _LIVE_CACHES_LOCK:
        _LIVE_CACHES.add(cache)


def _stop_live_caches() -> None:
    with _LIVE_CACHES_LOCK:
        live_caches = list(_LIVE_CACHES)
    for cache in live_caches:
        try:
            stop = getattr(cache, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            _logger.debug("Failed to stop cache during interpreter shutdown", exc_info=True)


def _wait_for_stop(stop_event: threading.Event, timeout: float) -> bool:
    return stop_event.wait(timeout)


atexit.register(_stop_live_caches)


@dataclass(frozen=True, slots=True)
class NamespaceInfo:
    name: str
    description: str | None


@dataclass(frozen=True, slots=True)
class MlflowConfigInfo:
    """Stores MLflowConfig CRD data for a namespace."""

    namespace: str
    artifact_root_path: str | None
    artifact_root_secret: str | None


class NamespaceCache:
    """Caches namespace name/description pairs using a watch loop."""

    def __init__(
        self,
        api: CoreV1Api,
        label_selector: str | None,
        namespace_exclude_globs: tuple[str, ...],
    ):
        self._api = api
        self._label_selector = label_selector
        self._namespace_exclude_globs = namespace_exclude_globs
        self._watch_factory = watch.Watch
        self._lock = threading.RLock()
        self._watcher: watch.Watch | None = None
        self._namespaces: dict[str, NamespaceInfo] = {}
        self._resource_version: str | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._refresh_full()

        self._thread = threading.Thread(
            target=self._run,
            name="mlflow-k8s-namespace-watch",
            daemon=True,
        )
        self._thread.start()
        _register_cache_for_shutdown(self)

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            watcher = self._watcher
        if watcher is not None:
            watcher.stop()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def list_namespaces(self) -> list[NamespaceInfo]:
        self._wait_until_ready()
        with self._lock:
            return list(self._namespaces.values())

    def get_namespace(self, name: str) -> NamespaceInfo | None:
        self._wait_until_ready()
        with self._lock:
            return self._namespaces.get(name)

    def _wait_until_ready(self) -> None:
        # Namespace listings depend on a background watch loop that typically initializes within
        # a second. Provide a generous timeout so transient API hiccups (e.g., API server restarts)
        # do not immediately surface as user-facing errors.
        if self._ready_event.wait(timeout=30):
            return
        raise MlflowException(
            "Timed out while waiting for the Kubernetes namespace cache to initialize. "
            "Double-check the MLflow server can reach the Kubernetes API.",
            INTERNAL_ERROR,
        )

    def _refresh_full(self) -> None:
        try:
            response = self._api.list_namespace(label_selector=self._label_selector, watch=False)
        except ApiException as exc:  # pragma: no cover - depends on live cluster
            raise MlflowException(
                f"Failed to list Kubernetes namespaces: {exc}",
                INVALID_PARAMETER_VALUE,
            ) from exc

        items = getattr(response, "items", []) or []
        metadata = getattr(response, "metadata", None)
        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            infos: dict[str, NamespaceInfo] = {}
            for ns in items:
                info = self._extract_info(ns)
                if info is not None:
                    infos[info.name] = info

            self._namespaces = infos
            self._resource_version = resource_version

        self._ready_event.set()

    def _run(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            if self._resource_version is None:
                try:
                    self._refresh_full()
                except MlflowException:
                    _logger.warning("Failed to refresh namespaces; retrying shortly", exc_info=True)
                    if _wait_for_stop(self._stop_event, 2):
                        break
                    continue

            watcher = self._watch_factory()
            with self._lock:
                self._watcher = watcher
            try:
                if self._stop_event.is_set():
                    break
                for event in watcher.stream(
                    self._api.list_namespace,
                    label_selector=self._label_selector,
                    resource_version=self._resource_version,
                    timeout_seconds=300,
                ):
                    if self._stop_event.is_set():
                        watcher.stop()
                        break
                    self._handle_event(event)
            except ApiException as exc:
                if exc.status == 410:
                    _logger.debug("Namespace watch resource version expired; resyncing.")
                    self._resource_version = None
                else:
                    _logger.warning("Namespace watch error: %s", exc)
                    if _wait_for_stop(self._stop_event, 2):
                        break
            except Exception:
                _logger.exception("Unexpected error in namespace watch loop")
                if _wait_for_stop(self._stop_event, 2):
                    break
            else:
                if _wait_for_stop(self._stop_event, 1):
                    break
            finally:
                with self._lock:
                    if self._watcher is watcher:
                        self._watcher = None

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        obj = event.get("object")
        if not obj:
            return

        metadata = getattr(obj, "metadata", None)
        name = getattr(metadata, "name", None)
        if not name:
            return

        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            if event_type in {"ADDED", "MODIFIED"}:
                info = self._extract_info(obj)
                if info is None:
                    self._namespaces.pop(name, None)
                else:
                    self._namespaces[name] = info
                self._resource_version = resource_version
            elif event_type == "DELETED":
                self._namespaces.pop(name, None)
                self._resource_version = resource_version
            elif event_type == "BOOKMARK":
                self._resource_version = resource_version or self._resource_version
            elif event_type == "ERROR":
                self._resource_version = None
                return
            else:
                # Unknown event type; trigger a resync.
                self._resource_version = None
                return

        self._ready_event.set()

    def _extract_info(self, namespace: object) -> NamespaceInfo | None:
        metadata = getattr(namespace, "metadata", None)
        name = getattr(metadata, "name", None)
        if not name:
            return None

        if self._is_excluded(name):
            return None

        annotations = getattr(metadata, "annotations", None) or {}
        description = annotations.get(DEFAULT_DESCRIPTION_ANNOTATION)

        return NamespaceInfo(name=name, description=description)

    def _is_excluded(self, name: str) -> bool:
        return any(fnmatchcase(name, pattern) for pattern in self._namespace_exclude_globs)


class MlflowConfigCache:
    """Caches MLflowConfig objects using a watch loop."""

    def __init__(
        self,
        api: CustomObjectsApi,
        ensure_artifact_connection_secret_cache: Callable[[], "SecretCache"] | None = None,
    ):
        self._api = api
        self._ensure_artifact_connection_secret_cache = ensure_artifact_connection_secret_cache
        self._watch_factory = watch.Watch
        self._lock = threading.RLock()
        self._watcher: watch.Watch | None = None
        self._configs: dict[str, MlflowConfigInfo] = {}
        self._resource_version: str | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._retry_event = threading.Event()
        self._crd_available = True
        self._crd_missing_logged = False

        self._refresh_full()

        self._thread = threading.Thread(
            target=self._run,
            name="mlflow-k8s-config-watch",
            daemon=True,
        )
        self._thread.start()
        _register_cache_for_shutdown(self)

    def stop(self) -> None:
        self._stop_event.set()
        self._retry_event.set()
        with self._lock:
            watcher = self._watcher
        if watcher is not None:
            watcher.stop()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def get_config(self, namespace: str) -> MlflowConfigInfo | None:
        """Get MLflowConfig for a namespace, or None if not found."""
        self._wait_until_ready()
        with self._lock:
            if self._crd_available:
                return self._configs.get(namespace)
        # CRD was unavailable — attempt a blocking reload outside the lock.
        if not self._try_reload():
            return None
        with self._lock:
            return self._configs.get(namespace)

    def _wait_until_ready(self) -> None:
        if self._ready_event.wait(timeout=30):
            return
        raise MlflowException(
            "Timed out waiting for MLflowConfig cache to initialize.",
            INTERNAL_ERROR,
        )

    def _try_reload(self) -> bool:
        """Attempt an immediate reload when the CRD was previously unavailable."""
        _logger.debug("MLflowConfig CRD unavailable; attempting immediate reload")
        self._refresh_full()
        # Wake the background thread so it transitions from the retry-sleep
        # into the active watch loop if the CRD is now available.
        self._retry_event.set()
        return self._crd_available

    def _refresh_full(self) -> None:
        try:
            response = self._api.list_cluster_custom_object(
                group=MLFLOW_CONFIG_GROUP,
                version=MLFLOW_CONFIG_VERSION,
                plural=MLFLOW_CONFIG_PLURAL,
            )
        except ApiException as exc:
            if exc.status == 404:
                # CRD not installed - this is okay, MLflowConfig is optional
                if not self._crd_missing_logged:
                    _logger.info(
                        "MLflowConfig CRD not installed. Artifact root overrides disabled. "
                        "Install the CRD to enable per-namespace artifact storage configuration."
                    )
                with self._lock:
                    self._crd_missing_logged = True
                    self._crd_available = False
                    self._configs.clear()
                    self._resource_version = None
            elif exc.status == 403:
                _logger.warning(
                    "Permission denied listing MLflowConfig CRDs. "
                    "Ensure RBAC allows 'list' and 'watch' on mlflowconfigs.mlflow.kubeflow.org"
                )
                with self._lock:
                    self._crd_available = False
                    self._configs.clear()
                    self._resource_version = None
            else:
                _logger.warning(f"Failed to list MLflowConfig CRDs: {exc}")
            self._ready_event.set()
            return
        except Exception:
            _logger.warning("Unexpected error listing MLflowConfig CRDs", exc_info=True)
            self._ready_event.set()
            return

        items = response.get("items", [])
        metadata = response.get("metadata", {})
        resource_version = metadata.get("resourceVersion")
        should_ensure_secret_cache = False

        with self._lock:
            self._configs = {}
            for item in items:
                if info := self._extract_info(item):
                    self._configs[info.namespace] = info
                    should_ensure_secret_cache = (
                        should_ensure_secret_cache or self._uses_artifact_connection_secret(info)
                    )
            self._resource_version = resource_version
            self._crd_available = True
            self._crd_missing_logged = False

        if should_ensure_secret_cache:
            self._ensure_secret_cache()
        self._ready_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Wait for a request-triggered retry or the 5-minute timeout.
            if not self._crd_available:
                self._retry_event.wait(timeout=300)
                self._retry_event.clear()
                if self._stop_event.is_set():
                    break
                # _try_reload may have already refreshed; only reset if still unavailable.
                with self._lock:
                    if not self._crd_available:
                        self._crd_available = True
                        self._resource_version = None
                continue

            if self._resource_version is None:
                try:
                    self._refresh_full()
                except Exception:
                    _logger.warning("Failed to refresh MLflowConfig; retrying", exc_info=True)
                    if _wait_for_stop(self._stop_event, 2):
                        break
                    continue

            # Skip watch if CRD became unavailable during refresh
            if not self._crd_available:
                continue

            watcher = self._watch_factory()
            with self._lock:
                self._watcher = watcher
            try:
                if self._stop_event.is_set():
                    break
                for event in watcher.stream(
                    self._api.list_cluster_custom_object,
                    group=MLFLOW_CONFIG_GROUP,
                    version=MLFLOW_CONFIG_VERSION,
                    plural=MLFLOW_CONFIG_PLURAL,
                    resource_version=self._resource_version,
                    timeout_seconds=300,
                ):
                    if self._stop_event.is_set():
                        watcher.stop()
                        break
                    self._handle_event(event)
            except ApiException as exc:
                if exc.status == 410:
                    _logger.debug("MLflowConfig watch expired; resyncing")
                    self._resource_version = None
                elif exc.status in (404, 403):
                    # CRD removed or permissions revoked
                    with self._lock:
                        self._crd_available = False
                        self._configs.clear()
                        self._resource_version = None
                else:
                    _logger.warning("MLflowConfig watch error: %s", exc)
                    if _wait_for_stop(self._stop_event, 2):
                        break
            except Exception:
                _logger.exception("Unexpected error in MLflowConfig watch loop")
                if _wait_for_stop(self._stop_event, 2):
                    break
            else:
                if _wait_for_stop(self._stop_event, 1):
                    break
            finally:
                with self._lock:
                    if self._watcher is watcher:
                        self._watcher = None

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        obj = event.get("object", {})
        metadata = obj.get("metadata", {})
        namespace = metadata.get("namespace")
        resource_version = metadata.get("resourceVersion")
        should_ensure_secret_cache = False

        if not namespace:
            return

        with self._lock:
            if event_type in {"ADDED", "MODIFIED"}:
                if info := self._extract_info(obj):
                    self._configs[namespace] = info
                    should_ensure_secret_cache = self._uses_artifact_connection_secret(info)
                self._resource_version = resource_version
            elif event_type == "DELETED":
                self._configs.pop(namespace, None)
                self._resource_version = resource_version
            elif event_type == "BOOKMARK":
                self._resource_version = resource_version or self._resource_version
            elif event_type == "ERROR":
                self._resource_version = None
            else:
                # Unknown event type; trigger a resync.
                self._resource_version = None

        if should_ensure_secret_cache:
            self._ensure_secret_cache()

    def _extract_info(self, obj: dict[str, object]) -> MlflowConfigInfo | None:
        metadata = obj.get("metadata", {})
        namespace = metadata.get("namespace")
        if not namespace:
            return None

        spec = obj.get("spec") or {}
        return MlflowConfigInfo(
            namespace=namespace,
            artifact_root_path=spec.get("artifactRootPath"),
            artifact_root_secret=spec.get("artifactRootSecret"),
        )

    @staticmethod
    def _uses_artifact_connection_secret(info: MlflowConfigInfo) -> bool:
        return info.artifact_root_secret == ARTIFACT_CONNECTION_SECRET_NAME

    def _ensure_secret_cache(self) -> None:
        """Ask the owner to ensure the shared SecretCache is running."""
        if self._ensure_artifact_connection_secret_cache is not None:
            self._ensure_artifact_connection_secret_cache()


@dataclass(frozen=True, slots=True)
class SecretInfo:
    namespace: str
    bucket_uri: str | None


class SecretCache:
    """Caches the shared artifact-connection Secret across namespaces."""

    def __init__(self, api: CoreV1Api):
        self._api = api
        self._watch_factory = watch.Watch
        self._lock = threading.RLock()
        self._watcher: watch.Watch | None = None
        self._secrets: dict[str, SecretInfo] = {}
        self._resource_version: str | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._retry_event = threading.Event()
        self._available = True

        self._refresh_full()

        self._thread = threading.Thread(
            target=self._run,
            name="mlflow-k8s-secret-watch",
            daemon=True,
        )
        self._thread.start()
        _register_cache_for_shutdown(self)

    def stop(self) -> None:
        self._stop_event.set()
        self._retry_event.set()
        with self._lock:
            watcher = self._watcher
        if watcher is not None:
            watcher.stop()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def get_secret(self, namespace: str) -> SecretInfo | None:
        self._wait_until_ready()
        with self._lock:
            if not self._available:
                return None
            return self._secrets.get(namespace)

    def _wait_until_ready(self) -> None:
        if self._ready_event.wait(timeout=30):
            return
        raise MlflowException(
            "Timed out waiting for the Kubernetes secret cache to initialize. "
            "Double-check the MLflow server can reach the Kubernetes API.",
            INTERNAL_ERROR,
        )

    def _refresh_full(self) -> None:
        try:
            response = self._api.list_secret_for_all_namespaces(
                field_selector=f"metadata.name={ARTIFACT_CONNECTION_SECRET_NAME}",
                watch=False,
            )
        except ApiException as exc:
            if exc.status == 403:
                _logger.warning(
                    "Permission denied listing secrets. Ensure RBAC allows 'list' and 'watch' "
                    "on secrets with resourceNames ['%s']. Per-namespace artifact root "
                    "overrides will be unavailable until this is resolved.",
                    ARTIFACT_CONNECTION_SECRET_NAME,
                )
                with self._lock:
                    self._available = False
                    self._secrets.clear()
                    self._resource_version = None
                self._ready_event.set()
                return
            raise MlflowException(
                f"Failed to list Kubernetes secrets: {exc}",
                INTERNAL_ERROR,
            ) from exc

        items = getattr(response, "items", []) or []
        metadata = getattr(response, "metadata", None)
        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            self._secrets = {}
            for secret in items:
                info = self._extract_info(secret)
                if info is not None:
                    self._secrets[info.namespace] = info
            self._resource_version = resource_version
            self._available = True

        self._ready_event.set()

    def _run(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            if not self._available:
                self._retry_event.wait(timeout=300)
                self._retry_event.clear()
                if self._stop_event.is_set():
                    break
                with self._lock:
                    if not self._available:
                        self._available = True
                        self._resource_version = None
                continue

            if self._resource_version is None:
                try:
                    self._refresh_full()
                except Exception:
                    _logger.warning("Failed to refresh secrets; retrying shortly", exc_info=True)
                    if _wait_for_stop(self._stop_event, 2):
                        break
                    continue

            if not self._available:
                continue

            watcher = self._watch_factory()
            with self._lock:
                self._watcher = watcher
            try:
                if self._stop_event.is_set():
                    break
                for event in watcher.stream(
                    self._api.list_secret_for_all_namespaces,
                    field_selector=f"metadata.name={ARTIFACT_CONNECTION_SECRET_NAME}",
                    resource_version=self._resource_version,
                    timeout_seconds=300,
                ):
                    if self._stop_event.is_set():
                        watcher.stop()
                        break
                    self._handle_event(event)
            except ApiException as exc:
                if exc.status == 410:
                    _logger.debug("Secret watch resource version expired; resyncing.")
                    self._resource_version = None
                elif exc.status == 403:
                    with self._lock:
                        self._available = False
                        self._secrets.clear()
                        self._resource_version = None
                else:
                    _logger.warning("Secret watch error: %s", exc)
                    if _wait_for_stop(self._stop_event, 2):
                        break
            except Exception:
                _logger.exception("Unexpected error in secret watch loop")
                if _wait_for_stop(self._stop_event, 2):
                    break
            else:
                if _wait_for_stop(self._stop_event, 1):
                    break
            finally:
                with self._lock:
                    if self._watcher is watcher:
                        self._watcher = None

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        obj = event.get("object")
        if not obj:
            return

        metadata = getattr(obj, "metadata", None)
        namespace = getattr(metadata, "namespace", None)
        if not namespace:
            return

        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            if event_type in {"ADDED", "MODIFIED"}:
                info = self._extract_info(obj)
                if info is None:
                    self._secrets.pop(namespace, None)
                else:
                    self._secrets[namespace] = info
                self._resource_version = resource_version
            elif event_type == "DELETED":
                self._secrets.pop(namespace, None)
                self._resource_version = resource_version
            elif event_type == "BOOKMARK":
                self._resource_version = resource_version or self._resource_version
            elif event_type == "ERROR":
                self._resource_version = None
                return
            else:
                self._resource_version = None
                return

        self._ready_event.set()

    def _extract_info(self, secret: object) -> SecretInfo | None:
        metadata = getattr(secret, "metadata", None)
        namespace = getattr(metadata, "namespace", None)
        if not namespace:
            return None

        data = getattr(secret, "data", None) or {}
        bucket_uri = self._decode_bucket_uri(data, namespace)
        return SecretInfo(namespace=namespace, bucket_uri=bucket_uri)

    @staticmethod
    def _decode_bucket_uri(data: dict[str, str], namespace: str) -> str | None:
        raw = data.get("AWS_S3_BUCKET")
        if not raw:
            return None
        try:
            bucket = base64.b64decode(raw).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            _logger.warning(
                "Invalid base64 data for AWS_S3_BUCKET in secret '%s' (namespace '%s')",
                ARTIFACT_CONNECTION_SECRET_NAME,
                namespace,
            )
            return None
        return f"s3://{bucket}"


__all__ = [
    "ARTIFACT_CONNECTION_SECRET_NAME",
    "DEFAULT_DESCRIPTION_ANNOTATION",
    "MLFLOW_CONFIG_GROUP",
    "MLFLOW_CONFIG_PLURAL",
    "MLFLOW_CONFIG_VERSION",
    "MlflowConfigCache",
    "MlflowConfigInfo",
    "NamespaceCache",
    "NamespaceInfo",
    "SecretCache",
    "SecretInfo",
]
