"""Random Forest model training and prediction."""

import math
from dataclasses import dataclass

import numpy as np
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


@dataclass
class LocalPredictionExplanation:
    """Validated SHAP explanation for the displayed prediction."""

    contributions: pd.DataFrame
    predicted_class: int
    class_index: int
    classes: list[int]
    base_value: float
    reconstructed_output: float
    model_output: str
    output_scale: str
    shap_output_shape: tuple[int, ...]


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


def explain_current_prediction(
    model: RandomForestClassifier,
    prediction_row: pd.DataFrame,
    feature_columns: list[str],
    prediction: PredictionResult,
) -> LocalPredictionExplanation:
    """Explain the exact displayed prediction with TreeExplainer."""
    import shap

    model_row = prediction_row.iloc[[-1]][feature_columns]
    fitted_feature_names = list(
        getattr(model, "feature_names_in_", [])
    )

    if fitted_feature_names != feature_columns:
        raise ValueError(
            "The current feature names or order do not match the fitted "
            "model."
        )

    if list(model_row.columns) != feature_columns:
        raise ValueError(
            "The current prediction row does not match the training "
            "feature order."
        )

    classes = [int(value) for value in model.classes_]

    if classes != [0, 1]:
        raise ValueError(
            "SHAP direction mapping requires model classes [0, 1]."
        )

    if prediction.predicted_class not in classes:
        raise ValueError(
            "The displayed prediction class is absent from the fitted model."
        )

    class_index = classes.index(prediction.predicted_class)
    probabilities = model.predict_proba(model_row)[0]

    if not math.isclose(
        float(probabilities[class_index]),
        prediction.probability,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "The displayed confidence does not match the fitted model."
        )

    explainer = shap.TreeExplainer(model)
    explanation = explainer(model_row)
    output_values = np.asarray(explanation.values, dtype=float)
    output_shape = tuple(output_values.shape)

    if output_values.shape == (1, len(feature_columns), len(classes)):
        class_contributions = output_values[0, :, class_index]
        all_class_contributions = output_values[0]
    elif output_values.shape == (len(feature_columns), len(classes)):
        class_contributions = output_values[:, class_index]
        all_class_contributions = output_values
    else:
        raise ValueError(
            "Unexpected SHAP output shape for binary classification: "
            f"{output_shape}."
        )

    base_values = np.asarray(explanation.base_values, dtype=float)

    if base_values.shape == (1, len(classes)):
        class_base_value = float(base_values[0, class_index])
    elif base_values.shape == (len(classes),):
        class_base_value = float(base_values[class_index])
    else:
        raise ValueError(
            "Unexpected SHAP base-value shape for binary classification: "
            f"{tuple(base_values.shape)}."
        )

    if len(class_contributions) != len(feature_columns):
        raise ValueError(
            "SHAP contribution count does not match the model features."
        )

    feature_values = model_row.iloc[0].to_numpy(dtype=float)

    if not (
        np.isfinite(class_contributions).all()
        and np.isfinite(feature_values).all()
        and math.isfinite(class_base_value)
    ):
        raise ValueError(
            "The local explanation contains nonfinite values."
        )

    if not np.allclose(
        all_class_contributions[:, 0],
        -all_class_contributions[:, 1],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Binary SHAP contributions are not complementary, so bullish "
            "and bearish directions cannot be mapped safely."
        )

    reconstructed_output = float(
        class_base_value + class_contributions.sum()
    )
    predicted_probability = float(probabilities[class_index])

    if not math.isclose(
        reconstructed_output,
        predicted_probability,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "SHAP values did not reconstruct the explained model output."
        )

    directional_contributions = (
        class_contributions
        if prediction.predicted_class == 1
        else -class_contributions
    )

    contributions = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": feature_values,
            "shap_value": class_contributions,
            "directional_contribution": directional_contributions,
        }
    )

    return LocalPredictionExplanation(
        contributions=contributions,
        predicted_class=prediction.predicted_class,
        class_index=class_index,
        classes=classes,
        base_value=class_base_value,
        reconstructed_output=reconstructed_output,
        model_output=str(explainer.model.model_output),
        output_scale="class probability",
        shap_output_shape=output_shape,
    )
