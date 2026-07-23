"""Main results tab components."""

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.formatting import (
    highlight_winner,
    prepare_prediction_log,
    prepare_rankings_table,
)
from src.visualization import (
    create_price_chart,
    create_rankings_chart,
)


def render_results(
    result: Any,
    market_data: pd.DataFrame,
) -> None:
    """Render the primary prediction results."""
    st.subheader("Prediction")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Direction",
        result.prediction.label,
        help=(
            "Bullish means the model expects the selected day's close "
            "to finish above the prior daily close."
        ),
    )

    metric_columns[1].metric(
        "Estimated Confidence",
        f"{result.prediction.probability:.1%}",
        help=(
            "The proportion of Random Forest trees supporting "
            "the predicted direction."
        ),
    )

    metric_columns[2].metric(
        "Prediction Date",
        result.prediction_date.strftime("%Y-%m-%d"),
    )

    metric_columns[3].metric(
        "Lookback",
        f"{result.selected_lookback} Days",
    )

    st.divider()
    st.subheader("Model Performance")

    evaluation = result.selected_evaluation

    performance_columns = st.columns(4)

    performance_columns[0].metric(
        "Lookback Accuracy",
        f"{evaluation.accuracy:.1%}",
    )

    performance_columns[1].metric(
        "Average Confidence",
        f"{evaluation.predictions['probability'].mean():.1%}",
    )

    performance_columns[2].metric(
        "Evaluated Predictions",
        len(evaluation.predictions),
    )

    performance_columns[3].metric(
        "Evaluation Method",
        "Walk-Forward",
    )

    if result.recommendation is not None:
        st.divider()
        st.subheader("Adaptive Lookback Recommendation")

        rankings = prepare_rankings_table(
            result.recommendation.rankings
        )

        st.dataframe(
            rankings.style.apply(
                highlight_winner,
                axis=1,
            ),
            width="stretch",
            hide_index=True,
        )

        st.plotly_chart(
            create_rankings_chart(
                result.recommendation.rankings
            ),
            width="stretch",
        )

    st.divider()
    st.subheader("Historical Prediction Log")

    st.dataframe(
        prepare_prediction_log(result),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Bitcoin Price History")

    st.plotly_chart(
        create_price_chart(market_data),
        width="stretch",
    )