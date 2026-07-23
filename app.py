"""Streamlit interface for the Adaptive Financial Prediction Framework."""

from pathlib import Path

import streamlit as st

from src.constants import (
    APP_TITLE,
    CANDIDATE_LOOKBACKS,
    MAX_CUSTOM_LOOKBACK,
    MIN_CUSTOM_LOOKBACK,
)
from src.framework import FinancialPredictionFramework
from src.visualization import (
    create_evaluation_chart,
    create_price_chart,
    create_rankings_chart,
)

DATA_PATH = Path("data/raw/btc_1d_data_2018_to_2025.csv")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="₿",
    layout="wide",
)

st.title(APP_TITLE)

st.caption(
    "Daily Bitcoin direction prediction using Random Forest models, "
    "walk-forward validation, and adaptive lookback selection."
)

with st.sidebar:
    st.header("Prediction Settings")

    st.selectbox(
        "Financial Asset",
        options=["Bitcoin"],
        disabled=True,
    )

    st.selectbox(
        "Timeframe",
        options=["Daily"],
        disabled=True,
    )

    selection_mode = st.radio(
        "Lookback Selection",
        options=["Recommended", "Custom"],
    )

    custom_lookback = None

    if selection_mode == "Custom":
        custom_lookback = st.number_input(
            "Custom Lookback (Days)",
            min_value=MIN_CUSTOM_LOOKBACK,
            max_value=MAX_CUSTOM_LOOKBACK,
            value=20,
            step=1,
        )

    run_prediction = st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    )

st.info(
    "The recommended mode evaluates "
    f"{', '.join(map(str, CANDIDATE_LOOKBACKS))}-day lookbacks "
    "using the latest 30 walk-forward predictions."
)

try:
    framework = FinancialPredictionFramework(DATA_PATH)
except Exception as error:
    st.error(f"Unable to load the market dataset: {error}")
    st.stop()

st.plotly_chart(
    create_price_chart(framework.market_data),
    use_container_width=True,
)

if run_prediction:
    try:
        with st.spinner(
            "Evaluating models and generating the next-day prediction..."
        ):
            result = framework.run(
                custom_lookback=(
                    int(custom_lookback)
                    if custom_lookback is not None
                    else None
                )
            )

        st.success("Prediction completed successfully.")

        metric_one, metric_two, metric_three, metric_four = st.columns(4)

        metric_one.metric(
            "Prediction Date",
            result.prediction_date.strftime("%Y-%m-%d"),
        )

        metric_two.metric(
            "Direction",
            result.prediction.label,
        )

        metric_three.metric(
            "Estimated Probability",
            f"{result.prediction.probability:.1%}",
        )

        metric_four.metric(
            "Selected Lookback",
            f"{result.selected_lookback} days",
        )

        st.subheader("Prediction Summary")

        st.write(
            f"The model predicts a **{result.prediction.label.lower()}** "
            f"Bitcoin daily close for "
            f"**{result.prediction_date.strftime('%B %d, %Y')}**, "
            f"with an estimated probability of "
            f"**{result.prediction.probability:.1%}**."
        )

        if result.recommendation is not None:
            st.subheader("Adaptive Lookback Recommendation")

            rankings = result.recommendation.rankings.copy()
            rankings["accuracy"] = rankings["accuracy"].map(
                lambda value: f"{value:.1%}"
            )
            rankings["precision"] = rankings["precision"].map(
                lambda value: f"{value:.1%}"
            )
            rankings["recall"] = rankings["recall"].map(
                lambda value: f"{value:.1%}"
            )
            rankings["f1_score"] = rankings["f1_score"].map(
                lambda value: f"{value:.1%}"
            )

            st.dataframe(
                rankings,
                use_container_width=True,
                hide_index=True,
            )

            st.plotly_chart(
                create_rankings_chart(
                    result.recommendation.rankings
                ),
                use_container_width=True,
            )

            selected_evaluation = (
                result.recommendation.evaluations[
                    result.selected_lookback
                ]
            )

            st.plotly_chart(
                create_evaluation_chart(selected_evaluation),
                use_container_width=True,
            )

        st.caption(
            "This application is an educational decision-support tool and "
            "does not provide financial advice."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")