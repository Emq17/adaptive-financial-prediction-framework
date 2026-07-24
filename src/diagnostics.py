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
    "joblib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "streamlit",
    "shap",
    "plotly",
    "threadpoolctl",
)


def _run_git(project_root: Path, *arguments: str) -> str:
    """Run a Git inspection command and return its standard output."""
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_diagnostics(project_root: Path) -> dict[str, Any]:
    """Return the commit and each component of the working-tree state."""
    try:
        commit = _run_git(project_root, "rev-parse", "HEAD")
        porcelain = _run_git(
            project_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        status_lines = porcelain.splitlines() if porcelain else []
        tracked_modifications = [
            line for line in status_lines if not line.startswith("??")
        ]
        untracked_files = [
            line[3:] for line in status_lines if line.startswith("??")
        ]

        return {
            "git_head_commit": commit,
            "git_is_dirty": bool(status_lines),
            "git_tracked_file_modifications": tracked_modifications,
            "git_untracked_files": untracked_files,
            "git_status_porcelain": status_lines,
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "git_head_commit": "unavailable",
            "git_is_dirty": None,
            "git_tracked_file_modifications": [],
            "git_untracked_files": [],
            "git_status_porcelain": [],
        }


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

    diagnostics = {
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
    diagnostics.update(_git_diagnostics(resolved_path.parents[2]))
    return diagnostics


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
    """Display comparison-ready diagnostics when debug mode is enabled."""
    import streamlit as st

    if result.diagnostics is None:
        return

    diagnostics = result.diagnostics

    st.divider()
    st.subheader("Debug Diagnostics")
    st.warning(
        "Diagnostic mode is enabled. Disable AFPF_DEBUG for the public "
        "application."
    )

    st.markdown("#### Environment and dataset")
    overview_keys = [
        "git_head_commit",
        "git_is_dirty",
        "python_version",
        "platform",
        "dataset_path",
        "dataset_sha256",
        "dataset_row_count",
        "dataset_first_date",
        "dataset_last_date",
    ]
    st.dataframe(
        pd.DataFrame(
            {
                "Field": overview_keys,
                "Value": [
                    str(_json_safe(diagnostics.get(key)))
                    for key in overview_keys
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Git working-tree state")
    git_rows = [
        {
            "Category": "Tracked-file modification",
            "Entry": entry,
        }
        for entry in diagnostics["git_tracked_file_modifications"]
    ]
    git_rows.extend(
        {
            "Category": "Untracked file",
            "Entry": entry,
        }
        for entry in diagnostics["git_untracked_files"]
    )
    if not git_rows:
        git_rows.append(
            {
                "Category": "Clean",
                "Entry": "No tracked modifications or untracked files",
            }
        )
    st.dataframe(pd.DataFrame(git_rows), hide_index=True, width="stretch")
    st.code(
        "\n".join(diagnostics["git_status_porcelain"])
        or "(clean working tree)",
        language="text",
    )

    st.markdown("#### Package versions")
    st.dataframe(
        pd.DataFrame(
            [
                {"Package": package, "Version": version}
                for package, version in diagnostics[
                    "package_versions"
                ].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Random-state values")
    st.dataframe(
        pd.DataFrame(
            [
                {"Setting": setting, "Value": value}
                for setting, value in diagnostics[
                    "random_state_values"
                ].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Feature columns and matrix")
    st.dataframe(
        pd.DataFrame(
            {
                "Position": range(len(diagnostics["feature_columns"])),
                "Feature": diagnostics["feature_columns"],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rows": diagnostics["feature_matrix_shape"][0],
                    "Columns": diagnostics["feature_matrix_shape"][1],
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Target class distribution")
    st.dataframe(
        pd.DataFrame(
            [
                {"Class": target_class, "Count": count}
                for target_class, count in diagnostics[
                    "target_class_distribution"
                ].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Analysis-window scores before ranking")
    st.dataframe(
        diagnostics["analysis_window_scores_before_ranking"],
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Analysis-window ranking")
    st.dataframe(
        diagnostics["analysis_window_ranking"],
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Final model output")
    classes = diagnostics["model_classes"]
    probabilities = diagnostics["raw_predict_proba"]
    mapping = diagnostics["final_class_confidence_mapping"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Class index": index,
                    "Model class": model_class,
                    "Raw probability": probabilities[index],
                    "Mapped confidence": mapping[str(model_class)],
                }
                for index, model_class in enumerate(classes)
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Selected analysis window": diagnostics[
                        "selected_analysis_window"
                    ],
                    "Predicted class": diagnostics["predicted_class"],
                    "Predicted label": diagnostics["predicted_label"],
                    "Final confidence": diagnostics[
                        "displayed_confidence"
                    ],
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
