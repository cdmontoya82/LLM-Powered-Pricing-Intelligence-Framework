"""
Data Ingestion Pipeline
-----------------------
Loads raw time-series CSV, applies IQR-based outlier detection,
imputes missing values, and serializes to compressed Parquet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load raw CSV. Expects at least columns: ds (date), y (numeric)."""
    df = pd.read_csv(filepath, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows from {filepath}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str = "y", multiplier: float = 1.5) -> pd.DataFrame:
    """
    IQR-based outlier removal.
    Values outside [Q1 - k*IQR, Q3 + k*IQR] are replaced with NaN for imputation.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    n_before = df[column].notna().sum()
    df[column] = df[column].where(df[column].between(lower, upper), other=np.nan)
    n_after = df[column].notna().sum()

    logger.info(f"Outlier removal: {n_before - n_after} values flagged (IQR x{multiplier})")
    logger.info(f"Bounds → lower: {lower:.2f}, upper: {upper:.2f}")
    return df


def impute_missing(df: pd.DataFrame, column: str = "y", method: str = "interpolate") -> pd.DataFrame:
    """
    Impute NaN values.
    method: 'interpolate' (linear) | 'forward_fill' | 'median'
    """
    n_missing = df[column].isna().sum()
    if n_missing == 0:
        logger.info("No missing values detected.")
        return df

    if method == "interpolate":
        df[column] = df[column].interpolate(method="linear").bfill().ffill()
    elif method == "forward_fill":
        df[column] = df[column].ffill().bfill()
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    logger.info(f"Imputed {n_missing} missing values using '{method}'")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for downstream model context."""
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def save_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save processed DataFrame to compressed Parquet."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)
    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"Saved to {output_path} ({size_kb:.1f} KB, snappy compressed)")


def run_pipeline(
    input_path: str,
    output_path: str = "data/processed/clean.parquet",
    iqr_multiplier: float = 1.5,
    imputation_method: str = "interpolate",
) -> pd.DataFrame:
    """
    Full ingestion pipeline:
    CSV → clean → outlier removal → imputation → feature engineering → Parquet
    """
    df = load_csv(input_path)
    df = remove_outliers_iqr(df, multiplier=iqr_multiplier)
    df = impute_missing(df, method=imputation_method)
    df = add_features(df)
    save_parquet(df, output_path)

    logger.info(f"Pipeline complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument("--input", type=str, default="data/raw/sample.csv")
    parser.add_argument("--output", type=str, default="data/processed/clean.parquet")
    parser.add_argument("--iqr", type=float, default=1.5)
    parser.add_argument("--impute", type=str, default="interpolate")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.iqr, args.impute)
