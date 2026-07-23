"""Compare candidate lookbacks and recommend the strongest performer."""

from dataclasses import dataclass

import pandas as pd

from src.constants import CANDIDATE_LOOKBACKS
from src.evaluation.evaluator import EvaluationResult, walk_forward_evaluate
from src.feature_engineering import create_features


@dataclass
class RecommendationResult:
    """Ranked lookback evaluation results."""

    recommended_lookback: int
    rankings: pd.DataFrame
    evaluations: dict[int, EvaluationResult]


def recommend_lookback(
    market_data: pd.DataFrame,
    lookbacks: list[int] | None = None,
) -> RecommendationResult:
    """
    Evaluate candidate lookbacks and recommend the highest-ranked option.

    Models are ranked primarily by accuracy, then F1 score, precision,
    recall, and finally the shorter lookback when scores are tied.
    """
    candidate_lookbacks = lookbacks or CANDIDATE_LOOKBACKS

    if not candidate_lookbacks:
        raise ValueError("At least one candidate lookback is required.")

    evaluations: dict[int, EvaluationResult] = {}
    ranking_records: list[dict[str, float | int]] = []

    for lookback in candidate_lookbacks:
        featured_data = create_features(market_data, lookback)
        evaluation = walk_forward_evaluate(
            dataframe=featured_data,
            lookback=lookback,
        )

        evaluations[lookback] = evaluation

        ranking_records.append(
            {
                "lookback": lookback,
                "accuracy": evaluation.accuracy,
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1_score": evaluation.f1_score,
            }
        )

    rankings = (
        pd.DataFrame(ranking_records)
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
        rankings=rankings,
        evaluations=evaluations,
    )