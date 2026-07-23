"""Walk-forward evaluation for financial direction models."""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import (
    MINIMUM_TRAINING_ROWS,
    ROLLING_EVALUATION_WINDOW,
    TARGET_COLUMN,
)
from src.feature_engineering import get_feature_columns
from src.models.random_forest import (
    predict_latest,
    train_random_forest,
)


@dataclass
class EvaluationResult:
    """Summary and prediction history from walk-forward validation."""

    lookback: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions: pd.DataFrame


def walk_forward_evaluate(
    dataframe: pd.DataFrame,
    lookback: int,
    evaluation_window: int = ROLLING_EVALUATION_WINDOW,
    prediction_cutoff: pd.Timestamp | None = None,
) -> EvaluationResult:
    """
    Evaluate the model using expanding-window walk-forward validation.

    When prediction_cutoff is supplied, only outcomes known before that
    prediction date are eligible for evaluation.
    """
    if evaluation_window < 1:
        raise ValueError("Evaluation window must be at least 1.")

    feature_columns = get_feature_columns(dataframe)

    if not feature_columns:
        raise ValueError("No model features were found.")

    evaluation_data = (
        dataframe
        .dropna(subset=feature_columns + [TARGET_COLUMN])
        .reset_index(drop=True)
    )

    if prediction_cutoff is not None:
        cutoff = pd.Timestamp(prediction_cutoff).normalize()

        # A row dated June 15 predicts June 16. For a June 17 prediction,
        # June 16 is the most recent outcome that may be evaluated.
        latest_feature_date = cutoff - timedelta(days=1)

        evaluation_data = (
            evaluation_data[
                evaluation_data["date"] < latest_feature_date
            ]
            .reset_index(drop=True)
        )

    required_rows = MINIMUM_TRAINING_ROWS + evaluation_window

    if len(evaluation_data) < required_rows:
        raise ValueError(
            f"At least {required_rows} usable rows are required, "
            f"but only {len(evaluation_data)} were available."
        )

    test_start = len(evaluation_data) - evaluation_window
    prediction_records = []

    for test_index in range(test_start, len(evaluation_data)):
        training_data = evaluation_data.iloc[:test_index]
        test_row = evaluation_data.iloc[[test_index]]

        model, trained_feature_columns = train_random_forest(
            training_data
        )

        prediction = predict_latest(
            model=model,
            dataframe=test_row,
            feature_columns=trained_feature_columns,
        )

        actual_class = int(test_row[TARGET_COLUMN].iloc[0])
        feature_date = pd.Timestamp(test_row["date"].iloc[0])
        evaluated_prediction_date = feature_date + timedelta(days=1)

        prediction_records.append(
            {
                "prediction_date": evaluated_prediction_date,
                "feature_date": feature_date,
                "actual": actual_class,
                "predicted": prediction.predicted_class,
                "probability": prediction.probability,
                "correct": (
                    prediction.predicted_class == actual_class
                ),
            }
        )

    predictions = pd.DataFrame(prediction_records)

    actual_values = predictions["actual"]
    predicted_values = predictions["predicted"]

    return EvaluationResult(
        lookback=lookback,
        accuracy=float(
            accuracy_score(actual_values, predicted_values)
        ),
        precision=float(
            precision_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        f1_score=float(
            f1_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        predictions=predictions,
    )