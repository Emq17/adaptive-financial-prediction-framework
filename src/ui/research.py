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
            "The highlighted bar is the selected analysis window. Analysis "
            "windows are ranked primarily by walk-forward accuracy, with "
            "internal tie-breaking rules applied when needed."
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
        "Each point represents one historical prediction. The marker "
        "indicates whether the prediction was correct, while the vertical "
        "position shows the model's estimated confidence."
    )
    st.plotly_chart(
        create_probability_timeline(evaluation),
        width="stretch",
        key="research_walk_forward_timeline",
    )

    st.divider()
    st.subheader("Confusion Matrix")
    st.plotly_chart(
        create_confusion_matrix(evaluation),
        width="stretch",
        key="research_confusion_matrix",
    )
    st.caption(
        "Cells labeled 'Correct' show where the predicted market direction "
        "matched the actual outcome. Cells labeled 'Incorrect' show where "
        "the prediction differed from the actual outcome."
    )
