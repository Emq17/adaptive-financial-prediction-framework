"""Methodology and definition components."""

from typing import Any

import streamlit as st


def render_explanation(result: Any) -> None:
    """Explain the prediction workflow in concise, plain language."""
    prediction_count = len(result.selected_evaluation.predictions)

    st.subheader("Prediction Process")
    st.write(
        "The framework creates market features from completed daily Bitcoin "
        "candles, trains a Random Forest on eligible historical rows, and "
        "predicts whether the selected day's close will be above the prior "
        "close."
    )

    with st.expander("Prediction cutoff details"):
        st.write(
            "The final model used completed candles through "
            f"**{result.latest_market_date.strftime('%B %d, %Y')}**. "
            "Observations after the selected prediction cutoff were excluded "
            "from feature creation, evaluation, and training."
        )

    st.divider()
    st.subheader("Adaptive Analysis Window")

    if result.recommendation is not None:
        st.write(
            f"Recommended mode compared "
            f"{len(result.recommendation.rankings)} available analysis "
            f"windows. The {result.selected_lookback}-day analysis window "
            "ranked highest using recent out-of-sample performance."
        )
    else:
        st.write(
            f"Custom mode used the selected {result.selected_lookback}-day "
            "analysis window and evaluated it with the same walk-forward "
            "method."
        )

    st.divider()
    st.subheader("Walk-Forward Validation")
    st.write(
        f"The model made {prediction_count} sequential out-of-sample "
        "predictions. For each one, it trained only on earlier observations "
        "and then predicted the next unseen outcome."
    )

    st.divider()
    st.subheader("Confidence")
    st.write(
        "Estimated confidence is the proportion of Random Forest trees that "
        "voted for the displayed class. It measures agreement within the "
        "forest, not the probability that the market prediction is correct."
    )

    st.divider()
    st.subheader("Definitions")

    with st.expander("Open definitions"):
        st.markdown(
            """
            **Analysis Window**
            The number of previous trading days the model reviews before
            making a prediction.

            **Analysis Window Accuracy**
            How often the model correctly predicted market direction when
            using a specific analysis window.

            **Out-of-Sample Prediction**
            A prediction made using historical data the model did not learn
            from during training, providing a more realistic measure of
            performance.

            **SHAP Value**
            Shows how each market indicator influenced this specific
            prediction, indicating whether it pushed the model toward a
            bullish or bearish outlook.

            **Adaptive Selection**
            Automatically chooses the analysis window that has performed best
            on recent historical predictions.
            """
        )

    st.divider()
    st.subheader("Limitations")
    st.write(
        "Historical performance does not guarantee future results. Confidence "
        "measures tree agreement, and SHAP explains fitted-model behavior; "
        "neither establishes causation. This dashboard is educational and "
        "does not provide financial advice."
    )
