"""Main results tab components."""

from typing import Any

import streamlit as st

from src.ui.formatting import prepare_prediction_log
from src.visualization import create_prediction_drivers_chart


def render_results(result: Any) -> None:
    """Render the prediction, local explanation, and recent performance."""
    st.subheader("Prediction Summary")

    selection_mode = (
        "Recommended"
        if result.recommendation is not None
        else "Custom"
    )

    with st.container(border=True):
        primary_metrics = st.columns(3)
        primary_metrics[0].metric(
            "Predicted Direction",
            result.prediction.label,
            help=(
                "Bullish expects a higher close; Bearish expects a lower "
                "or unchanged close."
            ),
        )
        primary_metrics[1].metric(
            "Current Prediction Confidence",
            f"{result.prediction.probability:.1%}",
            help=(
                "The proportion of Random Forest trees supporting the "
                "predicted direction."
            ),
        )
        primary_metrics[2].metric(
            "Prediction Date",
            result.prediction_date.strftime("%Y-%m-%d"),
        )

        selection_metrics = st.columns(2)
        selection_metrics[0].metric("Mode", selection_mode)
        selection_metrics[1].metric(
            "Selected Analysis Window",
            f"{result.selected_lookback}-Day Analysis Window",
        )

    st.divider()
    st.subheader("Current Prediction Drivers")
    st.caption(
        "Which current inputs moved the model toward Bullish or Bearish?"
    )

    if result.local_explanation is not None:
        st.plotly_chart(
            create_prediction_drivers_chart(
                result.local_explanation
            ),
            width="stretch",
            key="results_current_prediction_drivers",
        )
        st.caption(
            "SHAP values estimate how each current input moved this specific "
            "prediction away from the model's usual prediction level. "
            "Positive chart values push toward bullish; negative chart "
            "values push toward bearish. These values explain the fitted "
            "model's behavior and do not establish causation or guarantee "
            "future market movement."
        )
    else:
        st.warning(
            "Current prediction drivers are unavailable. "
            f"{result.local_explanation_error or 'Validation failed.'}"
        )

    st.divider()
    st.subheader("Recent Walk-Forward Performance")

    evaluation = result.selected_evaluation
    predictions = evaluation.predictions

    centered_metrics = st.columns([1, 3, 1], gap="small")

    with centered_metrics[1]:
        performance_columns = st.columns(2, gap="small")

        with performance_columns[0]:
            with st.container(border=True):
                st.metric(
                    "Accuracy",
                    f"{predictions['correct'].mean():.1%}",
                )

        with performance_columns[1]:
            with st.container(border=True):
                st.metric(
                    "Predictions Evaluated",
                    len(predictions),
                )

    st.caption(
        "These metrics summarize recent out-of-sample evaluation results for "
        "the selected analysis window. They do not indicate whether the "
        "current prediction will be correct."
    )

    st.divider()
    st.subheader("Additional Details")

    with st.expander("Historical Prediction Log"):
        st.dataframe(
            prepare_prediction_log(result),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Complete Generated Explanation"):
        st.write(result.explanation)

    with st.expander("Prediction Verification"):
        if result.actual_class is None:
            st.write(
                "The actual outcome is not yet present in the dataset."
            )
        else:
            actual_label = (
                "Bullish" if result.actual_class == 1 else "Bearish"
            )
            result_label = (
                "Correct" if result.was_correct else "Incorrect"
            )
            st.write(
                f"Actual outcome: **{actual_label}**  \n"
                f"Prediction: **{result.prediction.label}**  \n"
                f"Result: **{result_label}**"
            )
