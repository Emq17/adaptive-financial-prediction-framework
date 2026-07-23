"""Streamlit entry point for the financial prediction dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import APP_TITLE
from src.framework import (
    FinancialPredictionFramework,
    FrameworkResult,
)
from src.ui.controls import render_sidebar
from src.ui.explanation import render_explanation
from src.ui.research import render_research
from src.ui.results import render_results
from src.visualization import create_price_chart

DATA_PATH = Path("data/raw/btc_1d_data_2018_to_2025.csv")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="₿",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }

        h1,
        h2,
        h3,
        h4,
        p,
        .stCaption,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            text-align: center;
        }

        [data-testid="stMetric"] {
            text-align: center;
        }

        [data-testid="stAlert"] {
            text-align: center;
        }

        [data-testid="stSidebar"] {
            text-align: center;
        }

        [data-testid="stSidebar"] button {
            width: 100%;
        }

        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 2rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_framework() -> FinancialPredictionFramework:
    """Load the dataset and framework once."""
    return FinancialPredictionFramework(DATA_PATH)


@st.cache_data(show_spinner=False)
def run_cached_prediction(
    prediction_date_iso: str,
    custom_lookback: int | None,
) -> FrameworkResult:
    """
    Run and cache a prediction for a specific date and lookback selection.

    Repeating the same request returns the saved result instead of retraining
    every walk-forward model.
    """
    prediction_framework = FinancialPredictionFramework(DATA_PATH)

    return prediction_framework.run(
        prediction_date=pd.Timestamp(prediction_date_iso),
        custom_lookback=custom_lookback,
    )


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


# The complete price chart remains visible at the top before and after
# predictions are generated.
st.plotly_chart(
    create_price_chart(framework.market_data),
    width="stretch",
)


if not settings.run_prediction:
    st.info(
        "Choose a prediction date and lookback mode, then select "
        "**Run Prediction**."
    )

    st.stop()


try:
    spinner_text = (
        "Evaluating candidate lookbacks and training the final model..."
        if settings.custom_lookback is None
        else "Evaluating the custom lookback and training the final model..."
    )

    with st.spinner(spinner_text):
        result = run_cached_prediction(
            prediction_date_iso=(
                settings.prediction_date.isoformat()
            ),
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