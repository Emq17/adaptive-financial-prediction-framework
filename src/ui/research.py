"""Research and model-diagnostic components."""

from typing import Any

import streamlit as st

from src.ui.formatting import (
    highlight_winner,
    prepare_rankings_table,
)
from src.visualization import (
    create_confusion_matrix,
    create_probability_timeline,
    create_rankings_chart,
)


def render_research(result: Any) -> None:
    """Render validation results and deeper model diagnostics."""
    evaluation = result.selected_evaluation
    predictions = evaluation.predictions

    st.subheader("Analysis Window Performance")
    st.caption("Which analysis window performed best?")

    if result.recommendation is not None:
        st.plotly_chart(
            create_rankings_chart(
                result.recommendation.rankings,
                result.recommendation.recommended_lookback,
            ),
            width="stretch",
            key="research_lookback_accuracy",
        )
        st.caption(
            "The highlighted bar is the analysis window selected from the "
            "available options based on recent walk-forward accuracy."
        )
    else:
        st.info(
            "Analysis window comparison is available in Recommended mode. "
            "Custom mode evaluates only the selected analysis window."
        )

    st.divider()
    st.subheader("Historical Prediction Timeline")
    st.caption(
        "How did the model's predictions and outcomes change over time?"
    )
    st.plotly_chart(
        create_probability_timeline(evaluation),
        width="stretch",
        key="research_walk_forward_timeline",
    )

    st.divider()
    st.subheader("Performance Statistics")
    st.caption("What do the recent out-of-sample results summarize?")

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Recent Accuracy",
        f"{predictions['correct'].mean():.1%}",
    )
    metric_columns[1].metric(
        "Average Confidence",
        f"{predictions['probability'].mean():.1%}",
    )
    metric_columns[2].metric(
        "Predictions Analyzed",
        len(predictions),
    )

    st.divider()
    st.subheader("Confusion Matrix")
    st.caption(
        "The confusion matrix summarizes correct and incorrect bullish and "
        "bearish predictions."
    )
    st.plotly_chart(
        create_confusion_matrix(evaluation),
        width="stretch",
        key="research_confusion_matrix",
    )

    st.divider()
    st.subheader("Analysis Window Comparison")
    st.caption("How did the available analysis windows compare?")

    if result.recommendation is not None:
        with st.expander("Open analysis window comparison"):
            rankings = prepare_rankings_table(
                result.recommendation.rankings
            )
            st.dataframe(
                rankings.style.apply(highlight_winner, axis=1),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info(
            "Analysis window comparison details require Recommended mode."
        )
