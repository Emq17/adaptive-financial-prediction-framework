# Adaptive Financial Prediction Framework

A WGU C964 Computer Science Capstone project that predicts the direction of the next daily Bitcoin candle using a Random Forest classifier. The application compares recent model performance across multiple analysis windows, supports a custom analysis window, and presents leakage-aware walk-forward evaluation in an interactive Streamlit dashboard.

## Features

- Date-aware historical and next-day prediction
- Recommended or custom analysis window
- Expanding-window, out-of-sample walk-forward evaluation
- Candlestick market-history chart
- Plain-language methodology and prediction explanations
- Prediction log and supporting performance statistics

### Prediction-focused visualizations

The dashboard includes focused visualizations for model interpretation,
recommendation, and recent predictive performance:

- Analysis Window Performance
- Historical Prediction Timeline
- Confusion Matrix
- Simplified Analysis Window Comparison
- Current Prediction Drivers using SHAP: shows how current feature values
  moved the displayed prediction away from the model's usual prediction level

The local SHAP panel explains only the current displayed prediction; it does
not establish causation or guarantee future movement.

## Project structure

```text
app.py                         Streamlit entry point
data/raw/                      Source Bitcoin OHLCV dataset
src/data_loader.py             Dataset loading and validation
src/feature_engineering.py     Feature and target creation
src/models/                    Random Forest training and prediction
src/evaluation/                Walk-forward evaluation and recommendation
src/framework.py               End-to-end orchestration
src/ui/controls.py             Sidebar controls and input validation
src/ui/results.py              Prediction results and performance output
src/ui/explanation.py          Methodology and plain-language definitions
src/ui/research.py             Model comparison and validation analysis
src/visualization.py           Plotly chart construction

## Run locally

Create and activate a Python virtual environment, install the pinned dependencies, and start Streamlit:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py

Use the sidebar to choose a prediction date and Recommended or Custom mode,
then select Run Prediction. The completed result remains available while
switching among Results, How It Works, and Model Research.

## Performance note

Generating a prediction may take several seconds because the application performs walk-forward evaluation before displaying the recommendation and supporting visualizations.

## Methodology

The framework creates price-return, candle, volume, rolling, momentum, and lagged-return features from completed daily candles. Recommended mode evaluates the configured candidate lookbacks using recent expanding-window walk-forward predictions. Each test observation remains outside its model's training data, and prediction-date cutoffs prevent future observations from entering evaluation or final training.

The selected Random Forest is trained on all eligible labeled observations before predicting the latest unlabeled feature row. Estimated confidence is the proportion of trees voting for the predicted class; it represents model agreement, not certainty.

## Important notice

This application is an educational decision-support project. It does not provide financial advice, and historical model performance does not guarantee future results.

For the final submission archive, include only the project source code and required data or assets. Exclude .git/, .venv/, Python cache directories, .pytest_cache/, and operating-system metadata files.
