"""Feature engineering for daily Bitcoin direction prediction."""

import pandas as pd

from src.constants import CLOSE_COLUMN, TARGET_COLUMN


def create_features(
    dataframe: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """
    Create historical features for a selected lookback window.

    Rows with complete model features are preserved even when the next-day
    target is unknown. This keeps the latest completed candle available for
    a future prediction.
    """
    if lookback < 1:
        raise ValueError("Lookback must be at least 1 day.")

    featured_data = dataframe.copy()

    featured_data["daily_return"] = (
        featured_data[CLOSE_COLUMN].pct_change()
    )

    featured_data["price_range"] = (
        featured_data["high"] - featured_data["low"]
    ) / featured_data["open"]

    featured_data["body_size"] = (
        featured_data[CLOSE_COLUMN] - featured_data["open"]
    ) / featured_data["open"]

    featured_data["volume_change"] = (
        featured_data["volume"].pct_change()
    )

    featured_data[f"rolling_mean_{lookback}"] = (
        featured_data[CLOSE_COLUMN]
        .rolling(window=lookback)
        .mean()
    )

    featured_data[f"rolling_std_{lookback}"] = (
        featured_data[CLOSE_COLUMN]
        .rolling(window=lookback)
        .std()
    )

    featured_data[f"momentum_{lookback}"] = (
        featured_data[CLOSE_COLUMN]
        / featured_data[CLOSE_COLUMN].shift(lookback)
        - 1
    )

    for lag in range(1, lookback + 1):
        featured_data[f"return_lag_{lag}"] = (
            featured_data["daily_return"].shift(lag)
        )

    next_close = featured_data[CLOSE_COLUMN].shift(-1)

    featured_data[TARGET_COLUMN] = (
        next_close > featured_data[CLOSE_COLUMN]
    ).astype("Int64")

    featured_data.loc[
        next_close.isna(),
        TARGET_COLUMN,
    ] = pd.NA

    feature_columns = get_feature_columns(featured_data)

    return (
        featured_data
        .dropna(subset=feature_columns)
        .reset_index(drop=True)
    )


def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return numeric model inputs while excluding identifiers and target."""
    excluded_columns = {
        "date",
        TARGET_COLUMN,
        "close_time",
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]