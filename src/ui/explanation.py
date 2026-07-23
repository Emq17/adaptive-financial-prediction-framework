"""Explanation tab components."""

from typing import Any

import streamlit as st

from src.ui.formatting import prepare_advanced_metrics


def render_explanation(result: Any) -> None:
    """Explain how the result was generated and interpreted."""
    st.subheader("Why This Result?")

    evaluation = result.selected_evaluation
    prediction_count = len(evaluation.predictions)

    if result.recommendation is not None:
        selection_text = (
            f"The framework compared "
            f"{len(result.recommendation.rankings)} candidate lookbacks "
            f"using {prediction_count} recent out-of-sample predictions "
            f"for each model. The {result.selected_lookback}-day lookback "
            f"ranked first and was selected automatically."
        )
    else:
        selection_text = (
            f"The {result.selected_lookback}-day lookback was chosen in "
            f"Custom mode and evaluated using {prediction_count} "
            f"sequential out-of-sample predictions."
        )

    st.info(
        f"**Why was this lookback used?**\n\n{selection_text}"
    )

    st.info(
        "**How was the prediction generated?**\n\n"
        f"The final Random Forest was trained using completed daily "
        f"candles through "
        f"**{result.latest_market_date.strftime('%B %d, %Y')}**. "
        "Data after the prediction date was excluded."
    )

    st.info(
        f"**What does {result.prediction.probability:.1%} confidence "
        f"mean?**\n\nApproximately "
        f"{result.prediction.probability:.1%} of the Random Forest trees "
        f"supported the **{result.prediction.label.lower()}** result. "
        "This represents model agreement, not certainty."
    )

    if result.actual_class is None:
        st.warning(
            "**Can the prediction be verified?**\n\n"
            "Not yet. The actual outcome is not present in the dataset."
        )
    else:
        actual_label = (
            "Bullish" if result.actual_class == 1 else "Bearish"
        )

        status = (
            "Correct ✅"
            if result.was_correct
            else "Incorrect ❌"
        )

        st.success(
            "**Can the prediction be verified?**\n\n"
            f"Actual outcome: **{actual_label}**  \n"
            f"Prediction: **{result.prediction.label}**  \n"
            f"Result: **{status}**"
        )

    with st.expander("Complete Generated Explanation"):
        st.write(result.explanation)

    if result.recommendation is not None:
        with st.expander("Advanced Recommendation Metrics"):
            st.dataframe(
                prepare_advanced_metrics(
                    result.recommendation.rankings
                ),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Bullish Prediction Precision measures how often bullish "
                "predictions were correct. Bullish Detection Rate measures "
                "how many actually bullish days were identified. Balanced "
                "F1 Score combines both."
            )

    with st.expander("How to Interpret This Dashboard"):
        st.markdown(
            """
            **Prediction Lookback**  
            Number of historical daily observations represented in the
            model's lagged features.

            **Lookback Accuracy**  
            Percentage of recent out-of-sample predictions that were correct
            for a specific lookback.

            **Walk-Forward Validation**  
            The model repeatedly trains on past observations and predicts the
            next unseen observation.

            **Out-of-Sample Prediction**  
            A prediction made for data excluded from model training.

            **Estimated Confidence**  
            Percentage of trees supporting the predicted class. This measures
            model agreement, not certainty.

            **Adaptive Selection**  
            Comparing multiple lookbacks and automatically selecting the
            strongest recent performer.
            """
        )