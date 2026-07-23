"""Plotly visualizations for framework results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.evaluation.evaluator import EvaluationResult


def create_price_chart(
    market_data: pd.DataFrame,
) -> go.Figure:
    """Create a clean candlestick chart using the complete dataset."""
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=market_data["date"],
                open=market_data["open"],
                high=market_data["high"],
                low=market_data["low"],
                close=market_data["close"],
                name="BTC/USDT",
            )
        ]
    )

    figure.update_layout(
        title={
            "text": "Bitcoin Historical Price — Entire Dataset",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=560,
        margin=dict(l=20, r=20, t=70, b=30),
        showlegend=False,
    )

    return figure


def create_rankings_chart(
    rankings: pd.DataFrame,
) -> go.Figure:
    """Compare candidate lookbacks by walk-forward accuracy."""
    chart_data = rankings.copy()

    chart_data["Lookback"] = (
        chart_data["lookback"].astype(str) + " Days"
    )

    chart_data["Lookback Accuracy"] = (
        chart_data["accuracy"] * 100
    )

    figure = px.bar(
        chart_data,
        x="Lookback",
        y="Lookback Accuracy",
        text="Lookback Accuracy",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    maximum_accuracy = chart_data["Lookback Accuracy"].max()

    figure.update_layout(
        title={
            "text": "Candidate Lookback Performance",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=430,
        yaxis_title="Lookback Accuracy (%)",
        xaxis_title="Prediction Lookback",
        yaxis_range=[0, max(100, maximum_accuracy + 10)],
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=30),
    )

    return figure


def create_probability_timeline(
    evaluation: EvaluationResult,
) -> go.Figure:
    """Display confidence and correctness across evaluated predictions."""
    chart_data = evaluation.predictions.copy()

    chart_data["confidence_percent"] = (
        chart_data["probability"] * 100
    )

    chart_data["result"] = chart_data["correct"].map(
        {
            True: "Correct",
            False: "Incorrect",
        }
    )

    chart_data["prediction_label"] = chart_data["predicted"].map(
        {
            0: "Bearish",
            1: "Bullish",
        }
    )

    chart_data["actual_label"] = chart_data["actual"].map(
        {
            0: "Bearish",
            1: "Bullish",
        }
    )

    figure = px.scatter(
        chart_data,
        x="prediction_date",
        y="confidence_percent",
        color="result",
        symbol="result",
        custom_data=[
            "prediction_label",
            "actual_label",
            "result",
        ],
        labels={
            "prediction_date": "Prediction Date",
            "confidence_percent": "Estimated Confidence (%)",
            "result": "Result",
        },
    )

    figure.update_traces(
        marker={"size": 12},
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Confidence: %{y:.1f}%<br>"
            "Prediction: %{customdata[0]}<br>"
            "Actual: %{customdata[1]}<br>"
            "Result: %{customdata[2]}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        title={
            "text": (
                f"Confidence Across the Last "
                f"{len(chart_data)} Out-of-Sample Predictions"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=430,
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=70, b=30),
        legend_title="Prediction Result",
    )

    return figure


def create_confusion_matrix(
    evaluation: EvaluationResult,
) -> go.Figure:
    """Create a confusion matrix for bullish and bearish predictions."""
    matrix = confusion_matrix(
        evaluation.predictions["actual"],
        evaluation.predictions["predicted"],
        labels=[0, 1],
    )

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Bearish", "Predicted Bullish"],
            y=["Actual Bearish", "Actual Bullish"],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y}<br>"
                "%{x}<br>"
                "Predictions: %{z}"
                "<extra></extra>"
            ),
            colorscale="Blues",
            showscale=False,
        )
    )

    figure.update_layout(
        title={
            "text": "Prediction Confusion Matrix",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=390,
        xaxis_title="Model Prediction",
        yaxis_title="Actual Outcome",
        margin=dict(l=20, r=20, t=70, b=30),
    )

    return figure