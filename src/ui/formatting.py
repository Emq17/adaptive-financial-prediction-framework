"""Formatting helpers for dashboard tables."""

from typing import Any

import pandas as pd


def prepare_prediction_log(
    result: Any,
) -> pd.DataFrame:
    """Create a readable historical prediction log."""
    predictions = result.selected_evaluation.predictions.copy()

    predictions["Prediction Date"] = pd.to_datetime(
        predictions["prediction_date"]
    ).dt.strftime("%Y-%m-%d")

    predictions["Lookback"] = (
        f"{result.selected_lookback} Days"
    )

    predictions["Prediction"] = predictions["predicted"].map(
        {
            0: "Bearish",
            1: "Bullish",
        }
    )

    predictions["Actual"] = predictions["actual"].map(
        {
            0: "Bearish",
            1: "Bullish",
        }
    )

    predictions["Correct"] = predictions["correct"].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    predictions["Estimated Confidence"] = predictions[
        "probability"
    ].map(lambda value: f"{value:.1%}")

    return predictions[
        [
            "Prediction Date",
            "Lookback",
            "Prediction",
            "Actual",
            "Correct",
            "Estimated Confidence",
        ]
    ]


def prepare_rankings_table(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Create the simplified lookback ranking table."""
    display_table = rankings.copy()

    display_table["Winner"] = display_table["rank"].map(
        lambda rank: "⭐" if rank == 1 else ""
    )

    display_table["Prediction Lookback"] = (
        display_table["lookback"].astype(str) + " Days"
    )

    display_table["Lookback Accuracy"] = display_table[
        "accuracy"
    ].map(lambda value: f"{value:.1%}")

    display_table["Average Confidence"] = display_table[
        "average_confidence"
    ].map(lambda value: f"{value:.1%}")

    display_table["Out-of-Sample Predictions"] = display_table[
        "out_of_sample_predictions"
    ]

    display_table["Rank"] = display_table["rank"]

    return display_table[
        [
            "Winner",
            "Prediction Lookback",
            "Lookback Accuracy",
            "Average Confidence",
            "Out-of-Sample Predictions",
            "Rank",
        ]
    ]


def highlight_winner(row: pd.Series) -> list[str]:
    """Highlight the first-ranked lookback."""
    if row["Rank"] == 1:
        return [
            "background-color: rgba(250, 204, 21, 0.18); "
            "font-weight: 700;"
        ] * len(row)

    return [""] * len(row)


def prepare_advanced_metrics(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Create the technical model metrics table."""
    advanced_table = rankings[
        [
            "lookback",
            "precision",
            "recall",
            "f1_score",
        ]
    ].copy()

    advanced_table.columns = [
        "Lookback",
        "Bullish Prediction Precision",
        "Bullish Detection Rate",
        "Balanced F1 Score",
    ]

    advanced_table["Lookback"] = (
        advanced_table["Lookback"].astype(str) + " Days"
    )

    for column in [
        "Bullish Prediction Precision",
        "Bullish Detection Rate",
        "Balanced F1 Score",
    ]:
        advanced_table[column] = advanced_table[column].map(
            lambda value: f"{value:.1%}"
        )

    return advanced_table