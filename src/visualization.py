"""Plotly visualizations for framework results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.evaluation.evaluator import EvaluationResult
from src.models.random_forest import LocalPredictionExplanation
from src.ui.formatting import format_feature_label


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
    recommended_lookback: int,
) -> go.Figure:
    """Compare candidate lookbacks by walk-forward accuracy."""
    chart_data = rankings.copy()

    chart_data["Analysis Window"] = (
        chart_data["lookback"].astype(str)
        + "-Day Analysis Window"
    )

    chart_data["Analysis Window Accuracy"] = (
        chart_data["accuracy"] * 100
    )
    selected_mask = chart_data["lookback"] == recommended_lookback
    bar_colors = [
        "#FACC15" if is_selected else "#4C78A8"
        for is_selected in selected_mask
    ]
    selection_labels = [
        (
            "Selected Analysis Window"
            if is_selected
            else "Other Analysis Window"
        )
        for is_selected in selected_mask
    ]

    figure = go.Figure(
        data=go.Bar(
            x=chart_data["Analysis Window"],
            y=chart_data["Analysis Window Accuracy"],
            text=chart_data["Analysis Window Accuracy"],
            marker_color=bar_colors,
            customdata=selection_labels,
            hovertemplate=(
                "Analysis window: %{x}<br>"
                "Recent accuracy: %{y:.1f}%<br>"
                "Status: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    maximum_accuracy = chart_data["Analysis Window Accuracy"].max()

    figure.update_layout(
        template="plotly_dark",
        height=430,
        yaxis_title="Accuracy (%)",
        xaxis_title="Analysis Window",
        yaxis_range=[0, max(100, maximum_accuracy + 10)],
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=False,
    )

    return figure


def create_prediction_drivers_chart(
    explanation: LocalPredictionExplanation,
) -> go.Figure:
    """Show the strongest local drivers of the current prediction."""
    chart_data = explanation.contributions.copy()
    chart_data["absolute_contribution"] = chart_data[
        "directional_contribution"
    ].abs()
    bullish_drivers = chart_data[
        chart_data["directional_contribution"] >= 0
    ].nlargest(5, "absolute_contribution")
    bearish_drivers = chart_data[
        chart_data["directional_contribution"] < 0
    ].nlargest(5, "absolute_contribution")
    selected_drivers = pd.concat(
        [bullish_drivers, bearish_drivers]
    ).drop_duplicates(subset=["feature"])

    if len(selected_drivers) < min(10, len(chart_data)):
        remaining_drivers = chart_data[
            ~chart_data["feature"].isin(selected_drivers["feature"])
        ].nlargest(
            10 - len(selected_drivers),
            "absolute_contribution",
        )
        selected_drivers = pd.concat(
            [selected_drivers, remaining_drivers]
        )

    chart_data = selected_drivers.sort_values(
        "absolute_contribution",
        ascending=False,
    )
    chart_data["feature_label"] = chart_data["feature"].map(
        format_feature_label
    )
    chart_data["contribution_points"] = (
        chart_data["directional_contribution"] * 100
    )
    chart_data["shap_points"] = chart_data["shap_value"] * 100
    chart_data["direction"] = chart_data[
        "contribution_points"
    ].map(
        lambda value: (
            "Toward Bullish" if value >= 0 else "Toward Bearish"
        )
    )
    chart_data["display_value"] = chart_data["feature_value"].map(
        lambda value: f"{value:,.6g}"
    )

    direction_colors = {
        "Toward Bullish": px.colors.qualitative.Plotly[2],
        "Toward Bearish": px.colors.qualitative.Plotly[1],
    }

    figure = px.bar(
        chart_data,
        x="contribution_points",
        y="feature_label",
        orientation="h",
        color="direction",
        color_discrete_map=direction_colors,
        custom_data=[
            "feature",
            "display_value",
            "shap_points",
            "direction",
        ],
        labels={
            "contribution_points": "Influence on Prediction",
            "feature_label": "Input Feature",
            "direction": "Direction",
        },
    )

    figure.update_traces(
        hovertemplate=(
            "Feature: %{y}<br>"
            "Raw name: %{customdata[0]}<br>"
            "Current value: %{customdata[1]}<br>"
            "Influence value: %{customdata[2]:.3f}<br>"
            "Direction: %{customdata[3]}"
            "<extra></extra>"
        )
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_color="rgba(255,255,255,0.7)",
    )
    figure.update_layout(
        title={
            "text": "What Influenced This Prediction?",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=max(470, 38 * len(chart_data) + 160),
        xaxis_title="Influence on Prediction",
        yaxis_title="Input Feature",
        yaxis={"autorange": "reversed"},
        legend_title="Contribution Direction",
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

    average_confidence = chart_data["confidence_percent"].mean()
    figure.add_hline(
        y=average_confidence,
        line_dash="dash",
        line_width=1,
        line_color="rgba(160,174,192,0.8)",
    )
    figure.add_annotation(
        x=0.99,
        xref="paper",
        y=average_confidence,
        yref="y",
        text=f"Average Confidence: {average_confidence:.1f}%",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=10,
        font={
            "color": "rgba(255,255,255,0.95)",
            "size": 12,
        },
        bgcolor="rgba(17,24,39,0.82)",
        bordercolor="rgba(160,174,192,0.45)",
        borderpad=4,
    )

    figure.update_layout(
        template="plotly_dark",
        height=320,
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Prediction Result",
    )

    return figure


def create_confusion_matrix(
    evaluation: EvaluationResult,
) -> go.Figure:
    """Create a confusion matrix for bullish and bearish predictions."""
    class_values = [0, 1]
    class_labels = {
        0: "Bearish",
        1: "Bullish",
    }
    matrix = confusion_matrix(
        evaluation.predictions["actual"],
        evaluation.predictions["predicted"],
        labels=class_values,
    )
    annotations = []

    for actual_index, actual_class in enumerate(class_values):
        annotation_row = []

        for predicted_index, predicted_class in enumerate(class_values):
            is_correct = actual_class == predicted_class
            status = "Correct ✓" if is_correct else "Incorrect"
            count = int(matrix[actual_index, predicted_index])
            annotation_row.append(f"{status}<br>{count}")

        annotations.append(annotation_row)

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[
                f"Predicted {class_labels[value]}"
                for value in class_values
            ],
            y=[
                f"Actual {class_labels[value]}"
                for value in class_values
            ],
            text=annotations,
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
        template="plotly_dark",
        height=390,
        xaxis_title="Predicted Direction",
        yaxis_title="Actual Direction",
        margin=dict(l=20, r=20, t=30, b=30),
    )

    return figure
