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

    st.subheader("Analysis Window Performance")
    st.caption(
        "Which analysis window performed best during recent walk-forward "
        "evaluation?"
    )

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
            "The highlighted bar is the selected analysis window. Selection "
            "is based primarily on recent out-of-sample accuracy, with "
            "existing internal tie-breakers applied when needed."
        )

        with st.expander("View exact analysis window values"):
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
            "Analysis window comparison is available in Recommended mode. "
            "Custom mode evaluates only the selected analysis window."
        )

    st.divider()
    st.subheader("Historical Prediction Timeline")
    st.caption(
        "How did confidence and prediction correctness vary across the "
        "recent walk-forward evaluation period?"
    )
    st.plotly_chart(
        create_probability_timeline(evaluation),
        width="stretch",
        key="research_walk_forward_timeline",
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
