"""Main results tab components."""

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.formatting import (
    highlight_winner,
    prepare_prediction_log,
    prepare_rankings_table,
)
from src.visualization import create_rankings_chart


def center_table(
    dataframe: pd.DataFrame,
):
    """Center-align table headers and values."""
    return (
        dataframe.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center")],
                }
            ]
        )
    )


def render_results(
    result: Any,
    market_data: pd.DataFrame,
) -> None:
    """Render the primary prediction results."""
    del market_data

    st.subheader("Prediction")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Direction",
        result.prediction.label,
        help=(
            "Bullish means the model expects the selected day's close "
            "to finish above the previous daily close."
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
        help="The date whose daily closing direction is being predicted.",
    )

    metric_columns[3].metric(
        "Model Lookback",
        f"{result.selected_lookback} Days",
        help=(
            "The historical feature window applied to the final "
            "Random Forest model."
        ),
    )

    if result.recommendation is not None:
        st.success(
            f"**Selected Lookback: {result.selected_lookback} Days**  \n"
            f"The adaptive evaluation ranked this lookback first. It was "
            f"then applied to the final Random Forest model trained through "
            f"**{result.latest_market_date.strftime('%B %d, %Y')}** to "
            f"generate the prediction for "
            f"**{result.prediction_date.strftime('%B %d, %Y')}**."
        )
    else:
        st.info(
            f"**Custom Lookback: {result.selected_lookback} Days**  \n"
            f"This lookback was applied to the final Random Forest model "
            f"trained through "
            f"**{result.latest_market_date.strftime('%B %d, %Y')}** to "
            f"generate the prediction for "
            f"**{result.prediction_date.strftime('%B %d, %Y')}**."
        )

    st.divider()
    st.subheader("Model Lookback Performance")

    evaluation = result.selected_evaluation

    average_confidence = evaluation.predictions[
        "probability"
    ].mean()

    performance_columns = st.columns(4)

    performance_columns[0].metric(
        "Lookback Accuracy",
        f"{evaluation.accuracy:.1%}",
        help=(
            "The percentage of recent out-of-sample predictions "
            "that were correct for this lookback."
        ),
    )

    performance_columns[1].metric(
        "Average Confidence",
        f"{average_confidence:.1%}",
        help=(
            "Average model agreement across the evaluated "
            "out-of-sample predictions."
        ),
    )

    performance_columns[2].metric(
        "Evaluated Predictions",
        len(evaluation.predictions),
        help=(
            "The number of sequential out-of-sample predictions "
            "used to evaluate the lookback."
        ),
    )

    performance_columns[3].metric(
        "Evaluation Method",
        "Walk-Forward",
        help=(
            "Each model was trained using only observations "
            "available before its prediction date."
        ),
    )

    if result.recommendation is not None:
        st.divider()
        st.subheader("Adaptive Lookback Recommendation")

        rankings = prepare_rankings_table(
            result.recommendation.rankings
        )

        styled_rankings = (
            rankings.style
            .apply(
                highlight_winner,
                axis=1,
            )
            .set_properties(**{"text-align": "center"})
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [("text-align", "center")],
                    }
                ]
            )
        )

        st.dataframe(
            styled_rankings,
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

    prediction_log = prepare_prediction_log(result)

    st.dataframe(
        center_table(prediction_log),
        width="stretch",
        hide_index=True,
    )