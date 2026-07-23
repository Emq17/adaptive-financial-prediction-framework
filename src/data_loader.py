"""Load and validate the Binance Bitcoin daily dataset."""

from pathlib import Path

import pandas as pd

from src.constants import DATE_COLUMN, REQUIRED_COLUMNS


def load_market_data(file_path: str | Path) -> pd.DataFrame:
    """Load cleaned daily Bitcoin OHLCV data from the Binance CSV."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    dataframe.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in dataframe.columns
    ]

    dataframe = dataframe.rename(
        columns={
            "open_time": "date",
        }
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    dataframe[DATE_COLUMN] = pd.to_datetime(
        dataframe[DATE_COLUMN],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != DATE_COLUMN
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = (
        dataframe.dropna(subset=REQUIRED_COLUMNS)
        .drop_duplicates(subset=[DATE_COLUMN])
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if dataframe.empty:
        raise ValueError("No valid rows remained after data cleaning.")

    return dataframe