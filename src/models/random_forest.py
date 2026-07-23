"""Random Forest model training and prediction."""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.constants import N_ESTIMATORS, RANDOM_STATE, TARGET_COLUMN
from src.feature_engineering import get_feature_columns


@dataclass
class PredictionResult:
    """Prediction output returned by the model."""

    predicted_class: int
    probability: float
    label: str


def build_random_forest() -> RandomForestClassifier:
    """Create a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def train_random_forest(
    dataframe: pd.DataFrame,
) -> tuple[RandomForestClassifier, list[str]]:
    """Train a Random Forest using the supplied featured dataset."""
    feature_columns = get_feature_columns(dataframe)

    if not feature_columns:
        raise ValueError("No valid feature columns were found.")

    training_data = dataframe.dropna(
        subset=feature_columns + [TARGET_COLUMN]
    )

    if training_data.empty:
        raise ValueError("No valid training rows were found.")

    features = training_data[feature_columns]
    target = training_data[TARGET_COLUMN].astype(int)

    model = build_random_forest()
    model.fit(features, target)

    return model, feature_columns


def predict_latest(
    model: RandomForestClassifier,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> PredictionResult:
    """Predict the direction represented by the latest feature row."""
    if dataframe.empty:
        raise ValueError("Prediction data is empty.")

    latest_features = dataframe.iloc[[-1]][feature_columns]

    predicted_class = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]

    class_index = list(model.classes_).index(predicted_class)
    probability = float(probabilities[class_index])

    label = "Bullish" if predicted_class == 1 else "Bearish"

    return PredictionResult(
        predicted_class=predicted_class,
        probability=probability,
        label=label,
    )