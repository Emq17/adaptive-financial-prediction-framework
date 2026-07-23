"""Walk-forward evaluation for financial direction models."""

from dataclasses import dataclass

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
) -> EvaluationResult:
    """
    Evaluate the model using expanding-window walk-forward validation.

    Each observation is predicted using only data available before its date.
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

        prediction_records.append(
            {
                "date": test_row["date"].iloc[0],
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