"""Orchestration layer for the financial prediction framework."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.constants import (
    DEBUG,
    MAX_CUSTOM_LOOKBACK,
    MIN_CUSTOM_LOOKBACK,
    ROLLING_EVALUATION_WINDOW,
    TARGET_COLUMN,
)
from src.diagnostics import (
    build_environment_diagnostics,
    emit_debug_diagnostics,
)
from src.data_loader import load_market_data
from src.evaluation.evaluator import (
    EvaluationResult,
    walk_forward_evaluate,
)
from src.evaluation.recommender import (
    RecommendationResult,
    recommend_lookback,
)
from src.feature_engineering import create_features
from src.models.random_forest import (
    LocalPredictionExplanation,
    PredictionResult,
    explain_current_prediction,
    predict_latest,
    train_random_forest,
)


@dataclass
class FrameworkResult:
    """Complete output from one framework execution."""

    selected_lookback: int
    latest_available_date: pd.Timestamp
    latest_market_date: pd.Timestamp
    prediction_date: pd.Timestamp
    prediction: PredictionResult
    selected_evaluation: EvaluationResult
    recommendation: RecommendationResult | None
    actual_class: int | None
    was_correct: bool | None
    explanation: str
    local_explanation: LocalPredictionExplanation | None
    local_explanation_error: str | None
    diagnostics: dict[str, object] | None


class FinancialPredictionFramework:
    """Coordinate loading, evaluation, selection, training, and prediction."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path).resolve()
        self.market_data = load_market_data(self.data_path)

    def run(
        self,
        prediction_date: pd.Timestamp | str | None = None,
        custom_lookback: int | None = None,
    ) -> FrameworkResult:
        """Run a date-aware prediction without using future information."""
        latest_available_date = pd.Timestamp(
            self.market_data["date"].max()
        ).normalize()

        selected_prediction_date = (
            pd.Timestamp(prediction_date).normalize()
            if prediction_date is not None
            else latest_available_date + timedelta(days=1)
        )

        self._validate_prediction_date(
            prediction_date=selected_prediction_date,
            latest_available_date=latest_available_date,
        )

        latest_market_date = (
            selected_prediction_date - timedelta(days=1)
        )

        available_data = (
            self.market_data[
                self.market_data["date"] <= latest_market_date
            ]
            .copy()
            .reset_index(drop=True)
        )

        if available_data.empty:
            raise ValueError(
                "No market data was available before the prediction date."
            )

        actual_latest_date = pd.Timestamp(
            available_data["date"].max()
        ).normalize()

        if actual_latest_date != latest_market_date:
            raise ValueError(
                "The selected prediction date does not have a completed "
                "daily candle immediately before it."
            )

        recommendation = None

        if custom_lookback is None:
            recommendation = recommend_lookback(
                market_data=available_data,
                prediction_date=selected_prediction_date,
            )

            selected_lookback = (
                recommendation.recommended_lookback
            )

            selected_evaluation = recommendation.evaluations[
                selected_lookback
            ]
        else:
            self._validate_custom_lookback(custom_lookback)
            selected_lookback = custom_lookback

            custom_features = create_features(
                dataframe=available_data,
                lookback=selected_lookback,
            )

            selected_evaluation = walk_forward_evaluate(
                dataframe=custom_features,
                lookback=selected_lookback,
                evaluation_window=ROLLING_EVALUATION_WINDOW,
                prediction_cutoff=selected_prediction_date,
            )

        featured_data = create_features(
            dataframe=available_data,
            lookback=selected_lookback,
        )

        training_data = featured_data.dropna(
            subset=[TARGET_COLUMN]
        )

        prediction_rows = featured_data[
            featured_data[TARGET_COLUMN].isna()
        ]

        if prediction_rows.empty:
            raise ValueError(
                "No unlabeled latest row was available for prediction."
            )

        prediction_row = prediction_rows.iloc[[-1]]

        model, feature_columns = train_random_forest(training_data)

        prediction = predict_latest(
            model=model,
            dataframe=prediction_row,
            feature_columns=feature_columns,
        )

        diagnostics = None

        if DEBUG:
            model_row = prediction_row.iloc[[-1]][feature_columns]
            raw_probabilities = model.predict_proba(model_row)[0]
            analysis_window_scores = None

            if recommendation is not None:
                score_records = []

                for lookback, evaluation in (
                    recommendation.evaluations.items()
                ):
                    evaluation_predictions = evaluation.predictions
                    actual_sequence = (
                        evaluation_predictions["actual"]
                        .astype(int)
                        .tolist()
                    )
                    predicted_sequence = (
                        evaluation_predictions["predicted"]
                        .astype(int)
                        .tolist()
                    )
                    confusion_counts = [
                        [
                            sum(
                                actual == actual_class
                                and predicted == predicted_class
                                for actual, predicted in zip(
                                    actual_sequence,
                                    predicted_sequence,
                                )
                            )
                            for predicted_class in [0, 1]
                        ]
                        for actual_class in [0, 1]
                    ]
                    score_records.append(
                        {
                            "window": lookback,
                            "accuracy": evaluation.accuracy,
                            "F1": evaluation.f1_score,
                            "precision": evaluation.precision,
                            "recall": evaluation.recall,
                            "number of predictions": len(
                                evaluation_predictions
                            ),
                            "number correct": int(
                                evaluation_predictions["correct"].sum()
                            ),
                            "confusion matrix (actual rows 0/1, "
                            "predicted columns 0/1)": str(
                                confusion_counts
                            ),
                            "prediction sequence": " ".join(
                                map(str, predicted_sequence)
                            ),
                            "actual target sequence": " ".join(
                                map(str, actual_sequence)
                            ),
                        }
                    )

                analysis_window_scores = pd.DataFrame(score_records)

            diagnostics = build_environment_diagnostics(
                data_path=self.data_path,
                market_data=self.market_data,
            )
            diagnostics.update(
                {
                    "final_five_feature_rows": featured_data.tail(5),
                    "feature_columns": feature_columns,
                    "feature_matrix_shape": list(
                        training_data[feature_columns].shape
                    ),
                    "target_class_distribution": {
                        str(key): int(value)
                        for key, value in (
                            training_data[TARGET_COLUMN]
                            .astype(int)
                            .value_counts()
                            .sort_index()
                            .items()
                        )
                    },
                    "random_state_values": {
                        "project_random_state": model.random_state,
                        "random_forest_random_state": model.random_state,
                        "random_forest_n_jobs": model.n_jobs,
                    },
                    "analysis_window_scores_before_ranking": (
                        analysis_window_scores
                    ),
                    "analysis_window_ranking": (
                        recommendation.rankings
                        if recommendation is not None
                        else None
                    ),
                    "selected_analysis_window": selected_lookback,
                    "model_classes": [
                        int(value) for value in model.classes_
                    ],
                    "raw_predict_proba": [
                        float(value) for value in raw_probabilities
                    ],
                    "final_class_confidence_mapping": {
                        str(int(model.classes_[index])): float(value)
                        for index, value in enumerate(raw_probabilities)
                    },
                    "predicted_class": prediction.predicted_class,
                    "predicted_label": prediction.label,
                    "displayed_confidence": prediction.probability,
                }
            )
            emit_debug_diagnostics(diagnostics)

        local_explanation = None
        local_explanation_error = None

        try:
            local_explanation = explain_current_prediction(
                model=model,
                prediction_row=prediction_row,
                feature_columns=feature_columns,
                prediction=prediction,
            )
        except Exception as error:
            local_explanation_error = str(error)

        actual_class = self._get_actual_class(
            prediction_date=selected_prediction_date,
            latest_market_date=latest_market_date,
            latest_available_date=latest_available_date,
        )

        was_correct = (
            prediction.predicted_class == actual_class
            if actual_class is not None
            else None
        )

        explanation = self._build_explanation(
            prediction_date=selected_prediction_date,
            prediction=prediction,
            selected_lookback=selected_lookback,
            evaluation=selected_evaluation,
            recommendation=recommendation,
            actual_class=actual_class,
            was_correct=was_correct,
        )

        return FrameworkResult(
            selected_lookback=selected_lookback,
            latest_available_date=latest_available_date,
            latest_market_date=latest_market_date,
            prediction_date=selected_prediction_date,
            prediction=prediction,
            selected_evaluation=selected_evaluation,
            recommendation=recommendation,
            actual_class=actual_class,
            was_correct=was_correct,
            explanation=explanation,
            local_explanation=local_explanation,
            local_explanation_error=local_explanation_error,
            diagnostics=diagnostics,
        )

    def _get_actual_class(
        self,
        prediction_date: pd.Timestamp,
        latest_market_date: pd.Timestamp,
        latest_available_date: pd.Timestamp,
    ) -> int | None:
        """Return the known outcome when the selected date is historical."""
        if prediction_date > latest_available_date:
            return None

        previous_rows = self.market_data[
            self.market_data["date"] == latest_market_date
        ]
        prediction_rows = self.market_data[
            self.market_data["date"] == prediction_date
        ]

        if previous_rows.empty or prediction_rows.empty:
            return None

        previous_close = float(previous_rows.iloc[0]["close"])
        prediction_close = float(prediction_rows.iloc[0]["close"])

        return int(prediction_close > previous_close)

    @staticmethod
    def _build_explanation(
        prediction_date: pd.Timestamp,
        prediction: PredictionResult,
        selected_lookback: int,
        evaluation: EvaluationResult,
        recommendation: RecommendationResult | None,
        actual_class: int | None,
        was_correct: bool | None,
    ) -> str:
        """Generate a plain-language explanation of the framework result."""
        prediction_count = len(evaluation.predictions)

        if recommendation is not None:
            selection_text = (
                f"The framework compared all available analysis windows and "
                f"selected the {selected_lookback}-day analysis window "
                f"because it ranked highest using recent out-of-sample "
                f"accuracy."
            )
        else:
            selection_text = (
                f"The {selected_lookback}-day analysis window was selected "
                f"manually and evaluated using the same walk-forward "
                f"method."
            )

        explanation = (
            f"{selection_text} The model was tested on "
            f"{prediction_count} sequential predictions made using only "
            f"information available before each prediction date. It achieved "
            f"{evaluation.accuracy:.1%} analysis window accuracy. The final "
            f"model "
            f"predicts a {prediction.label.lower()} close for "
            f"{prediction_date.strftime('%B %d, %Y')} with "
            f"{prediction.probability:.1%} estimated confidence. This "
            f"confidence represents the proportion of Random Forest trees "
            f"that supported the predicted class; it is not a guarantee."
        )

        if actual_class is not None:
            actual_label = (
                "bullish" if actual_class == 1 else "bearish"
            )
            correctness = "correct" if was_correct else "incorrect"

            explanation += (
                f" Because this is a historical prediction date, the known "
                f"outcome can be checked: the actual result was "
                f"{actual_label}, so the prediction was {correctness}."
            )
        else:
            explanation += (
                " The actual outcome is not yet present in the dataset, so "
                "the prediction cannot currently be marked correct or "
                "incorrect."
            )

        return explanation

    @staticmethod
    def _validate_custom_lookback(lookback: int) -> None:
        """Validate a user-selected lookback window."""
        if not MIN_CUSTOM_LOOKBACK <= lookback <= MAX_CUSTOM_LOOKBACK:
            raise ValueError(
                "Custom analysis window must be between "
                f"{MIN_CUSTOM_LOOKBACK} and {MAX_CUSTOM_LOOKBACK} days."
            )

    def _validate_prediction_date(
        self,
        prediction_date: pd.Timestamp,
        latest_available_date: pd.Timestamp,
    ) -> None:
        """Validate the requested historical or next-day prediction date."""
        earliest_available_date = pd.Timestamp(
            self.market_data["date"].min()
        ).normalize()

        latest_allowed_date = latest_available_date + timedelta(days=1)

        if prediction_date <= earliest_available_date:
            raise ValueError(
                "The prediction date must occur after the dataset begins."
            )

        if prediction_date > latest_allowed_date:
            raise ValueError(
                "The prediction date cannot be later than the day after "
                "the newest available candle."
            )
