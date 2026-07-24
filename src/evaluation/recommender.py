"""Compare candidate lookbacks and recommend the strongest performer."""

from dataclasses import dataclass

import pandas as pd

from src.constants import (
    CANDIDATE_LOOKBACKS,
    ROLLING_EVALUATION_WINDOW,
)
from src.evaluation.evaluator import (
    EvaluationResult,
    walk_forward_evaluate,
)
from src.feature_engineering import create_features


@dataclass
class RecommendationResult:
    """Ranked lookback evaluation results."""

    recommended_lookback: int
    candidate_scores: pd.DataFrame
    rankings: pd.DataFrame
    evaluations: dict[int, EvaluationResult]


def recommend_lookback(
    market_data: pd.DataFrame,
    prediction_date: pd.Timestamp,
    lookbacks: list[int] | None = None,
    evaluation_window: int = ROLLING_EVALUATION_WINDOW,
) -> RecommendationResult:
    """
    Evaluate candidate lookbacks using only information available before
    the selected prediction date.
    """
    candidate_lookbacks = lookbacks or CANDIDATE_LOOKBACKS

    if not candidate_lookbacks:
        raise ValueError("At least one candidate lookback is required.")

    evaluations: dict[int, EvaluationResult] = {}
    ranking_records: list[dict[str, float | int]] = []

    for lookback in candidate_lookbacks:
        featured_data = create_features(
            dataframe=market_data,
            lookback=lookback,
        )

        evaluation = walk_forward_evaluate(
            dataframe=featured_data,
            lookback=lookback,
            evaluation_window=evaluation_window,
            prediction_cutoff=prediction_date,
        )

        evaluations[lookback] = evaluation

        ranking_records.append(
            {
                "lookback": lookback,
                "accuracy": evaluation.accuracy,
                "average_confidence": (
                    evaluation.predictions["probability"].mean()
                ),
                "out_of_sample_predictions": len(
                    evaluation.predictions
                ),
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1_score": evaluation.f1_score,
            }
        )

    candidate_scores = pd.DataFrame(ranking_records)

    rankings = (
        candidate_scores
        .sort_values(
            by=[
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "lookback",
            ],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )

    rankings.insert(0, "rank", range(1, len(rankings) + 1))

    recommended_lookback = int(rankings.iloc[0]["lookback"])

    return RecommendationResult(
        recommended_lookback=recommended_lookback,
        candidate_scores=candidate_scores,
        rankings=rankings,
        evaluations=evaluations,
    )
