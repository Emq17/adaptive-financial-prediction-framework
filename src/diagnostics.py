"""Opt-in diagnostics for comparing local and deployed prediction runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "streamlit",
    "shap",
    "plotly",
)


def _git_identifier(project_root: Path) -> str:
    """Return the current commit and dirty state when Git is available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def build_environment_diagnostics(
    data_path: Path,
    market_data: pd.DataFrame,
) -> dict[str, Any]:
    """Collect environment and dataset identity information."""
    resolved_path = data_path.resolve()
    package_versions = {
        package: importlib.metadata.version(package)
        for package in PACKAGE_NAMES
    }

    return {
        "git_commit": _git_identifier(resolved_path.parents[2]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "dataset_path": str(resolved_path),
        "dataset_sha256": hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest(),
        "dataset_row_count": len(market_data),
        "dataset_first_date": market_data["date"].min(),
        "dataset_last_date": market_data["date"].max(),
    }


def _json_safe(value: Any) -> Any:
    """Convert diagnostic values to JSON-compatible structures."""
    if isinstance(value, pd.DataFrame):
        return [
            {
                str(key): _json_safe(item)
                for key, item in record.items()
            }
            for record in value.to_dict(orient="records")
        ]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def emit_debug_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Write diagnostics to application logs when debug mode is enabled."""
    print(
        "AFPF_DEBUG_DIAGNOSTICS="
        + json.dumps(_json_safe(diagnostics), sort_keys=True),
        flush=True,
    )


def render_debug_diagnostics(result: Any) -> None:
    """Display diagnostics in an expanded Streamlit section."""
    import streamlit as st

    if result.diagnostics is None:
        return

    with st.expander("Debug Diagnostics", expanded=False):
        st.warning(
            "Diagnostic mode is enabled. Disable AFPF_DEBUG for the "
            "public application."
        )
        st.json(_json_safe(result.diagnostics), expanded=False)
