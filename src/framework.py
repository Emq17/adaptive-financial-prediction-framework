"""Orchestration layer for the financial prediction framework."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.constants import (
    MAX_CUSTOM_LOOKBACK,
    MIN_CUSTOM_LOOKBACK,
    TARGET_COLUMN,
)
from src.data_loader import load_market_data
from src.evaluation.recommender import (
    RecommendationResult,
    recommend_lookback,
)
from src.feature_engineering import create_features
from src.models.random_forest import (
    PredictionResult,
    predict_latest,
    train_random_forest,
)


@dataclass
class FrameworkResult:
    """Complete output from one framework execution."""

    selected_lookback: int
    latest_market_date: pd.Timestamp
    prediction_date: pd.Timestamp
    prediction: PredictionResult
    recommendation: RecommendationResult | None


class FinancialPredictionFramework:
    """Coordinate data loading, model selection, training, and prediction."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.market_data = load_market_data(self.data_path)

    def run(
        self,
        custom_lookback: int | None = None,
    ) -> FrameworkResult:
        """
        Run the complete prediction workflow.

        When no custom lookback is supplied, candidate lookbacks are
        evaluated and the strongest recent performer is selected.
        """
        recommendation = None

        if custom_lookback is None:
            recommendation = recommend_lookback(self.market_data)
            selected_lookback = recommendation.recommended_lookback
        else:
            self._validate_custom_lookback(custom_lookback)
            selected_lookback = custom_lookback

        featured_data = create_features(
            dataframe=self.market_data,
            lookback=selected_lookback,
        )

        training_data = featured_data.dropna(
            subset=[TARGET_COLUMN]
        )

        prediction_row = featured_data[
            featured_data[TARGET_COLUMN].isna()
        ]

        if prediction_row.empty:
            raise ValueError(
                "No unlabeled latest row was available for prediction."
            )

        model, feature_columns = train_random_forest(training_data)

        prediction = predict_latest(
            model=model,
            dataframe=prediction_row.iloc[[-1]],
            feature_columns=feature_columns,
        )

        latest_market_date = pd.Timestamp(
            prediction_row.iloc[-1]["date"]
        )

        prediction_date = latest_market_date + timedelta(days=1)

        return FrameworkResult(
            selected_lookback=selected_lookback,
            latest_market_date=latest_market_date,
            prediction_date=prediction_date,
            prediction=prediction,
            recommendation=recommendation,
        )

    @staticmethod
    def _validate_custom_lookback(lookback: int) -> None:
        """Validate a user-selected lookback window."""
        if not MIN_CUSTOM_LOOKBACK <= lookback <= MAX_CUSTOM_LOOKBACK:
            raise ValueError(
                "Custom lookback must be between "
                f"{MIN_CUSTOM_LOOKBACK} and {MAX_CUSTOM_LOOKBACK} days."
            )