"""Plotly visualizations for framework results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.evaluation.evaluator import EvaluationResult


def create_price_chart(
    market_data: pd.DataFrame,
    display_days: int = 180,
) -> go.Figure:
    """Create a recent Bitcoin candlestick chart."""
    chart_data = market_data.tail(display_days)

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=chart_data["date"],
                open=chart_data["open"],
                high=chart_data["high"],
                low=chart_data["low"],
                close=chart_data["close"],
                name="BTC/USD",
            )
        ]
    )

    figure.update_layout(
        title=f"Bitcoin Price — Last {display_days} Days",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=500,
    )

    return figure


def create_rankings_chart(
    rankings: pd.DataFrame,
) -> go.Figure:
    """Compare candidate lookbacks by walk-forward accuracy."""
    chart_data = rankings.copy()
    chart_data["lookback_label"] = (
        chart_data["lookback"].astype(str) + " days"
    )
    chart_data["accuracy_percent"] = chart_data["accuracy"] * 100

    figure = px.bar(
        chart_data,
        x="lookback_label",
        y="accuracy_percent",
        text="accuracy_percent",
        labels={
            "lookback_label": "Lookback Window",
            "accuracy_percent": "Accuracy (%)",
        },
        title="Candidate Lookback Performance",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    figure.update_layout(
        template="plotly_white",
        height=430,
        yaxis_range=[
            0,
            max(100, chart_data["accuracy_percent"].max() + 10),
        ],
        showlegend=False,
    )

    return figure


def create_evaluation_chart(
    evaluation: EvaluationResult,
) -> go.Figure:
    """Display actual and predicted direction for each evaluation date."""
    chart_data = evaluation.predictions.copy()

    chart_data["actual_label"] = chart_data["actual"].map(
        {0: "Bearish", 1: "Bullish"}
    )
    chart_data["predicted_label"] = chart_data["predicted"].map(
        {0: "Bearish", 1: "Bullish"}
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["actual"],
            mode="lines+markers",
            name="Actual",
            customdata=chart_data["actual_label"],
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Actual: %{customdata}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["predicted"],
            mode="lines+markers",
            name="Predicted",
            customdata=chart_data["predicted_label"],
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Predicted: %{customdata}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=(
            f"Last {len(chart_data)} Walk-Forward Predictions "
            f"— {evaluation.lookback}-Day Lookback"
        ),
        xaxis_title="Date",
        yaxis={
            "title": "Direction",
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["Bearish", "Bullish"],
        },
        template="plotly_white",
        height=430,
        legend_title="Series",
    )

    return figure