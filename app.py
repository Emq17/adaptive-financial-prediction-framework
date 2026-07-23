"""Streamlit entry point for the financial prediction dashboard."""

from pathlib import Path

import streamlit as st

from src.constants import APP_TITLE
from src.framework import FinancialPredictionFramework
from src.ui.controls import render_sidebar
from src.ui.explanation import render_explanation
from src.ui.research import render_research
from src.ui.results import render_results

DATA_PATH = Path("data/raw/btc_1d_data_2018_to_2025.csv")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="₿",
    layout="wide",
)


@st.cache_resource
def load_framework() -> FinancialPredictionFramework:
    """Load the application framework once."""
    return FinancialPredictionFramework(DATA_PATH)


st.title(APP_TITLE)

st.caption(
    "Daily Bitcoin decision-support system using Random Forest models, "
    "walk-forward validation, and adaptive lookback selection."
)


try:
    framework = load_framework()
except Exception as error:
    st.error(f"Unable to load the market dataset: {error}")
    st.stop()


settings = render_sidebar(framework.market_data)


if not settings.run_prediction:
    st.info(
        "Choose a prediction date and lookback mode, then select "
        "**Run Prediction**."
    )

    st.stop()


try:
    spinner_text = (
        "Evaluating candidate models and generating prediction..."
        if settings.custom_lookback is None
        else "Evaluating the custom model and generating prediction..."
    )

    with st.spinner(spinner_text):
        result = framework.run(
            prediction_date=settings.prediction_date,
            custom_lookback=settings.custom_lookback,
        )

    st.success("Prediction completed successfully.")

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()


results_tab, explanation_tab, research_tab = st.tabs(
    [
        "Results",
        "Explanation",
        "Research",
    ]
)


with results_tab:
    render_results(
        result=result,
        market_data=framework.market_data,
    )


with explanation_tab:
    render_explanation(result)


with research_tab:
    render_research(result)


st.caption(
    "Educational decision-support application. "
    "This framework does not provide financial advice."
)