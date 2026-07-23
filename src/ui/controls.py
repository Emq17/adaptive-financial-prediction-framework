"""Sidebar controls for the Streamlit application."""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import streamlit as st

from src.constants import (
    MAX_CUSTOM_LOOKBACK,
    MIN_CUSTOM_LOOKBACK,
    MINIMUM_TRAINING_ROWS,
    ROLLING_EVALUATION_WINDOW,
)


@dataclass
class PredictionSettings:
    """User-selected dashboard settings."""

    prediction_date: pd.Timestamp
    custom_lookback: int | None
    run_prediction: bool


def create_prediction_date_options(
    market_data: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Return valid historical dates plus the next available date."""
    dates = (
        pd.to_datetime(market_data["date"])
        .dt.normalize()
        .sort_values()
        .reset_index(drop=True)
    )

    minimum_required_history = (
        MINIMUM_TRAINING_ROWS
        + ROLLING_EVALUATION_WINDOW
        + MAX_CUSTOM_LOOKBACK
    )

    historical_dates = dates.iloc[minimum_required_history:].tolist()
    next_date = dates.iloc[-1] + timedelta(days=1)

    return [next_date] + list(reversed(historical_dates))


def format_prediction_date(
    date_value: pd.Timestamp,
    latest_available_date: pd.Timestamp,
) -> str:
    """Format a prediction date for the dropdown."""
    date_text = date_value.strftime("%B %d, %Y")

    if date_value > latest_available_date:
        return f"{date_text} — Next Available Day"

    return f"{date_text} — Historical"


def render_sidebar(
    market_data: pd.DataFrame,
) -> PredictionSettings:
    """Render prediction controls and return selected settings."""
    latest_available_date = pd.Timestamp(
        market_data["date"].max()
    ).normalize()

    prediction_date_options = create_prediction_date_options(
        market_data
    )

    with st.sidebar:
        st.header("Prediction Settings")

        st.selectbox(
            "Financial Asset",
            options=["Bitcoin"],
            disabled=True,
            help="Version 1 currently supports Bitcoin only.",
        )

        st.selectbox(
            "Timeframe",
            options=["Daily"],
            disabled=True,
            help=(
                "The current dataset contains daily candles. "
                "Additional timeframes are planned for future versions."
            ),
        )

        prediction_date = st.selectbox(
            "Prediction Date",
            options=prediction_date_options,
            format_func=lambda date_value: format_prediction_date(
                date_value,
                latest_available_date,
            ),
            help=(
                "Historical dates simulate a prediction using only "
                "information available before that date."
            ),
        )

        selection_mode = st.radio(
            "Lookback Selection",
            options=["Recommended", "Custom"],
            help=(
                "Recommended mode compares multiple lookbacks. "
                "Custom mode evaluates the lookback you choose."
            ),
        )

        custom_lookback = None

        if selection_mode == "Custom":
            custom_lookback = int(
                st.number_input(
                    "Custom Lookback (Days)",
                    min_value=MIN_CUSTOM_LOOKBACK,
                    max_value=MAX_CUSTOM_LOOKBACK,
                    value=20,
                    step=1,
                    help=(
                        "The number of historical days used when creating "
                        "lagged model features."
                    ),
                )
            )

        run_prediction = st.button(
            "Run Prediction",
            type="primary",
            width="stretch",
        )

    return PredictionSettings(
        prediction_date=prediction_date,
        custom_lookback=custom_lookback,
        run_prediction=run_prediction,
    )