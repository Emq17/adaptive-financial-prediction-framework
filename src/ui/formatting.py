"""Formatting helpers for dashboard tables."""

import re
from typing import Any

import pandas as pd


def format_feature_label(feature_name: str) -> str:
    """Return a readable presentation label without renaming model inputs."""
    exact_labels = {
        "daily_return": "Daily Return",
        "price_range": "Daily Price Range",
        "body_size": "Candle Body Size",
        "volume_change": "Volume Change",
        "rolling_volatility": "Rolling Volatility",
    }

    if feature_name in exact_labels:
        return exact_labels[feature_name]

    lag_match = re.fullmatch(r"return_lag_(\d+)", feature_name)

    if lag_match:
        return f"{lag_match.group(1)}-Day Lagged Return"

    rolling_mean_match = re.fullmatch(
        r"rolling_mean_(\d+)",
        feature_name,
    )

    if rolling_mean_match:
        return f"{rolling_mean_match.group(1)}-Day Rolling Mean"

    rolling_std_match = re.fullmatch(
        r"rolling_std_(\d+)",
        feature_name,
    )

    if rolling_std_match:
        return f"{rolling_std_match.group(1)}-Day Rolling Volatility"

    momentum_match = re.fullmatch(r"momentum_(\d+)", feature_name)

    if momentum_match:
        return f"{momentum_match.group(1)}-Day Momentum"

    return feature_name.replace("_", " ").title()


def prepare_prediction_log(
    result: Any,
) -> pd.DataFrame:
    """Create a readable historical prediction log."""
    predictions = result.selected_evaluation.predictions.copy()

    predictions["Prediction Date"] = pd.to_datetime(
        predictions["prediction_date"]
    ).dt.strftime("%Y-%m-%d")

    predictions["Analysis Window"] = (
        f"{result.selected_lookback}-Day Analysis Window"
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
            "Analysis Window",
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

    display_table["Selected"] = display_table["rank"].map(
        lambda rank: "★" if rank == 1 else ""
    )

    display_table["Analysis Window"] = (
        display_table["lookback"].astype(str) + " Days"
    )

    display_table["Accuracy"] = display_table[
        "accuracy"
    ].map(lambda value: f"{value:.1%}")

    display_table["Average Confidence"] = display_table[
        "average_confidence"
    ].map(lambda value: f"{value:.1%}")

    display_table["Rank"] = display_table["rank"]

    return display_table[
        [
            "Selected",
            "Analysis Window",
            "Accuracy",
            "Average Confidence",
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
