"""Methodology and limitation components."""

from typing import Any

import streamlit as st


def render_explanation(result: Any) -> None:
    """Explain the prediction workflow in concise, plain language."""
    prediction_count = len(result.selected_evaluation.predictions)
    dataset_cutoff = result.latest_available_date.strftime("%B %d, %Y")

    st.subheader("Prediction Process")
    st.write(
        "The framework creates market indicators from completed daily "
        "Bitcoin data, trains a Random Forest classifier on eligible "
        "historical observations, and predicts whether the next trading day "
        "is more likely to close bullish or bearish."
    )
    st.caption(
        "Bullish indicates a close above the previous day's close. Bearish "
        "indicates a close at or below the previous day's close."
    )

    with st.expander("Prediction cutoff details"):
        st.write(
            "This demonstration uses a static dataset containing completed "
            f"Bitcoin data through **{dataset_cutoff}**; it does not use "
            "live market data. For this prediction, the model used completed "
            "candles through "
            f"**{result.latest_market_date.strftime('%B %d, %Y')}**. "
            "Later observations were excluded so the displayed prediction "
            "and historical evaluation remain reproducible."
        )

        if result.prediction_date > result.latest_available_date:
            st.caption(
                f"{result.prediction_date.strftime('%B %d, %Y')} is the next "
                "date after the final completed input candle, not a live "
                "current-date prediction."
            )

    st.divider()
    st.subheader("Adaptive Analysis Window")
    st.write(
        "The analysis window is the number of previous trading days the "
        "model reviews before making a prediction. Recommended mode compares "
        "the available window lengths and selects the one with the strongest "
        "recent walk-forward performance."
    )

    if result.recommendation is not None:
        available_windows = sorted(
            int(value)
            for value in result.recommendation.rankings["lookback"]
        )
        window_list = (
            ", ".join(str(value) for value in available_windows[:-1])
            + f", and {available_windows[-1]}"
        )
        st.caption(
            f"For this prediction, the {result.selected_lookback}-day "
            "analysis window achieved the strongest recent walk-forward "
            "performance among the available analysis windows "
            f"({window_list} days)."
        )
    else:
        st.caption(
            f"This result uses the manually selected "
            f"{result.selected_lookback}-day analysis window."
        )

    st.divider()
    st.subheader("Walk-Forward Validation")
    st.write(
        "Walk-forward validation tests the model on sequential historical "
        "days it had not yet seen. For each prediction, the model trains only "
        "on earlier observations and then predicts the next unseen outcome."
    )
    st.caption(
        f"The selected analysis window was evaluated with "
        f"{prediction_count} recent out-of-sample predictions."
    )

    st.divider()
    st.subheader("Confidence")
    st.write(
        "Estimated confidence is the proportion of Random Forest trees that "
        "voted for the displayed class. It measures agreement within the "
        "model, not the probability that the market prediction will be "
        "correct."
    )

    st.divider()
    st.subheader("Limitations")
    st.write(
        "Historical performance does not guarantee future results. Confidence "
        "measures tree agreement, and SHAP explains fitted-model behavior; "
        "neither establishes causation. This dashboard is educational and "
        "does not provide financial advice."
    )
