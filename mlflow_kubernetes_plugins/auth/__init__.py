"""Compatibility package for the MLflow Kubernetes auth plugin."""

from __future__ import annotations

from pathlib import Path

_LEGACY_MODULE_PATH = Path(__file__).resolve().parent.parent / "auth.py"

# During the package split, execute the legacy module inside this package namespace so
# imports, monkeypatching, and module-level caches still behave as if
# `mlflow_kubernetes_plugins.auth` were a single module.
exec(compile(_LEGACY_MODULE_PATH.read_text(encoding="utf-8"), str(_LEGACY_MODULE_PATH), "exec"))
