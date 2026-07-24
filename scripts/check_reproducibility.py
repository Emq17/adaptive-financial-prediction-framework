"""Verify identical recommended predictions in separate Python processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_prediction() -> dict[str, object]:
    """Run one prediction and return its reproducibility signature."""
    from src.constants import DATA_PATH
    from src.framework import FinancialPredictionFramework

    result = FinancialPredictionFramework(DATA_PATH).run(
        prediction_date="2026-06-17"
    )
    ranking = result.recommendation.rankings[
        [
            "rank",
            "lookback",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
        ]
    ]

    return {
        "selected_analysis_window": result.selected_lookback,
        "predicted_class": result.prediction.predicted_class,
        "confidence": result.prediction.probability,
        "analysis_window_ranking": ranking.to_dict(orient="records"),
    }


def run_worker() -> dict[str, object]:
    """Execute the worker in a clean Python process."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    """Run the worker or compare two clean-process results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()

    if arguments.worker:
        print(json.dumps(run_prediction(), sort_keys=True))
        return

    first = run_worker()
    second = run_worker()

    if first != second:
        raise SystemExit(
            "Reproducibility check failed:\n"
            f"first={json.dumps(first, indent=2)}\n"
            f"second={json.dumps(second, indent=2)}"
        )

    print("Cross-process reproducibility check passed.")
    print(json.dumps(first, indent=2))


if __name__ == "__main__":
    main()
