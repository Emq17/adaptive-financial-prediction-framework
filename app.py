"""Streamlit entry point for the financial prediction dashboard."""

import streamlit as st

from src.constants import APP_TITLE, DATA_PATH, DEBUG
from src.diagnostics import render_debug_diagnostics
from src.framework import FinancialPredictionFramework
from src.ui.controls import render_sidebar
from src.ui.explanation import render_explanation
from src.ui.research import render_research
from src.ui.results import render_results
from src.visualization import create_price_chart

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="₿",
    layout="wide",
)


def load_framework() -> FinancialPredictionFramework:
    """Load the framework from the current tracked dataset."""
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

request_signature = (
    settings.prediction_date.isoformat(),
    settings.custom_lookback,
)
stored_signature = st.session_state.get("prediction_request_signature")
result = (
    st.session_state.get("prediction_result")
    if stored_signature == request_signature
    else None
)

if settings.run_prediction:
    try:
        spinner_text = (
            "Evaluating analysis windows, generating the prediction, and "
            "explaining the current result..."
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
        st.session_state["prediction_request_signature"] = request_signature
        st.success("Prediction completed successfully.")

    except Exception as error:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_request_signature", None)
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

if DEBUG:
    render_debug_diagnostics(result)


st.caption(
    "Educational decision-support application. "
    "This framework does not provide financial advice."
)
