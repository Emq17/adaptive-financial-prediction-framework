# Adaptive Financial Prediction Framework

An interactive machine learning application that predicts whether the next
daily Bitcoin close is more likely to be bullish or bearish. It combines a
Random Forest classifier, adaptive analysis-window selection, walk-forward
evaluation, and SHAP explanations in a Streamlit dashboard.

The project is designed to make model behavior and recent historical
performance easy to inspect without presenting predictions as financial
advice.

## Live Demo

Explore the deployed application at
[quantmatrix.streamlit.app](https://quantmatrix.streamlit.app).

## Key Features

- Predicts bullish or bearish daily Bitcoin direction
- Supports recommended and custom analysis windows
- Selects an analysis window using recent walk-forward performance
- Evaluates predictions on sequential historical observations
- Explains the displayed prediction with local SHAP values
- Compares analysis-window accuracy with exact supporting values
- Includes a historical prediction timeline and confusion matrix
- Displays Bitcoin price history and a detailed prediction log
- Provides an interactive Streamlit interface

## How It Works

The framework creates market indicators from historical Bitcoin prices and
trading activity. It trains a Random Forest classifier to predict whether the
next close will finish above the previous close (bullish) or at or below it
(bearish).

Recommended mode compares the available analysis windows using recent
walk-forward evaluation. For each historical test day, the model learns only
from earlier observations before predicting the next unseen outcome. This
provides a more realistic view of recent performance while avoiding future
data in training.

After the final prediction, SHAP values show which current market indicators
pushed the fitted model toward a bullish or bearish result. These explanations
describe model behavior; they do not establish causation or guarantee future
performance.

## Dataset

The bundled dataset contains completed daily Bitcoin market observations
through June 16, 2026. It is a static historical dataset rather than a live
market feed, which keeps predictions and evaluation results reproducible.

## Dashboard

The interface is organized into three sections:

- **Results** — prediction, confidence, SHAP drivers, and recent walk-forward
  performance
- **How It Works** — methodology, analysis-window selection, confidence, and
  limitations
- **Model Research** — analysis-window comparison, historical prediction
  timeline, and confusion matrix

Generating a prediction may take several seconds because the application
performs walk-forward evaluation before displaying the result.

## Project Layout

- `app.py` — Streamlit entry point
- `data/raw/` — bundled Bitcoin market dataset
- `src/data_loader.py` — dataset loading and validation
- `src/feature_engineering.py` — market-indicator and target creation
- `src/models/` — Random Forest training and prediction
- `src/evaluation/` — walk-forward evaluation and window recommendation
- `src/framework.py` — prediction workflow orchestration
- `src/ui/` — Streamlit controls and section renderers
- `src/visualization.py` — Plotly chart construction

## Prerequisites

- Python 3.11 or a compatible Python 3 release
- `pip`

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Use the sidebar to choose a prediction date and Recommended or Custom mode,
then select **Run Prediction**. The completed result remains available while
switching among the dashboard sections.

## Disclaimer

This application is intended for educational and research purposes only. It
does not provide financial or investment advice.
