"""Streamlit entry point for the financial prediction dashboard."""

from pathlib import Path

import streamlit as st

from src.constants import APP_TITLE
from src.framework import FinancialPredictionFramework
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


@st.cache_resource
def load_framework() -> FinancialPredictionFramework:
    """Load the application framework once."""
    return FinancialPredictionFramework(DATA_PATH)


st.title(APP_TITLE)

st.caption(
    "Daily Bitcoin decision-support system using Random Forest models, "
    "walk-forward validation, and adaptive analysis window selection."
)


try:
    framework = load_framework()
except Exception as error:
    st.error(f"Unable to load the market dataset: {error}")
    st.stop()

st.subheader("Bitcoin Price History")
st.plotly_chart(
    create_price_chart(framework.market_data),
    width="stretch",
    key="persistent_bitcoin_price_history",
)

settings = render_sidebar(framework.market_data)

result = st.session_state.get("prediction_result")

if settings.run_prediction:
    try:
        spinner_text = (
            "Evaluating candidate models, generating the prediction, and "
            "explaining the current prediction..."
            if settings.custom_lookback is None
            else "Evaluating the custom model, generating the prediction, "
            "and explaining the current prediction..."
        )

        with st.spinner(spinner_text):
            result = framework.run(
                prediction_date=settings.prediction_date,
                custom_lookback=settings.custom_lookback,
            )

        st.session_state["prediction_result"] = result
        st.success("Prediction completed successfully.")

    except Exception as error:
        st.session_state.pop("prediction_result", None)
        st.error(f"Prediction failed: {error}")
        st.stop()

if result is None:
    st.info(
        "Choose a prediction date and analysis window mode, then select "
        "**Run Prediction**."
    )
    st.stop()

st.subheader("Explore This Prediction")
selected_section = st.segmented_control(
    "Prediction sections",
    options=[
        "Results",
        "How It Works",
        "Model Research",
    ],
    default="Results",
    required=True,
    key="main_navigation",
    label_visibility="collapsed",
    width="stretch",
)

section_captions = {
    "Results": "Prediction, confidence, and recent performance",
    "How It Works": "Methodology and definitions",
    "Model Research": "Validation and deeper diagnostics",
}
st.caption(section_captions[selected_section])
st.divider()

if selected_section == "Results":
    render_results(result)
elif selected_section == "How It Works":
    render_explanation(result)
else:
    render_research(result)


st.caption(
    "Educational decision-support application. "
    "This framework does not provide financial advice."
)
