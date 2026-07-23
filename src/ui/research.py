"""Research tab components."""

from typing import Any

import streamlit as st

from src.visualization import (
    create_confusion_matrix,
    create_probability_timeline,
)


def render_research(result: Any) -> None:
    """Render model diagnostics and monitoring visuals."""
    st.subheader("Model Research and Diagnostics")

    st.caption(
        "These diagnostics examine the selected lookback's recent "
        "out-of-sample behavior."
    )

    st.plotly_chart(
        create_probability_timeline(
            result.selected_evaluation
        ),
        width="stretch",
    )

    st.plotly_chart(
        create_confusion_matrix(
            result.selected_evaluation
        ),
        width="stretch",
    )

    predictions = result.selected_evaluation.predictions

    overall_accuracy = predictions["correct"].mean()
    average_confidence = predictions["probability"].mean()

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "Recent Accuracy",
        f"{overall_accuracy:.1%}",
    )

    metric_columns[1].metric(
        "Average Confidence",
        f"{average_confidence:.1%}",
    )

    metric_columns[2].metric(
        "Predictions Analyzed",
        len(predictions),
    )

    st.info(
        "Future research can expand this page with accuracy by month, "
        "market volatility, day of week, market regime, and major events."
    )