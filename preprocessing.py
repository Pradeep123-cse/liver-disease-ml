"""
Preprocessing utilities for the Indian Liver Patient Dataset (ILPD).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Dataset",
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the ILPD CSV file and assign column names.

    Args:
        filepath: Path to the CSV file (no header row).

    Returns:
        Raw DataFrame with labelled columns.
    """
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Clean, encode, scale and split the data.

    Steps:
        1. Drop duplicate rows.
        2. Encode ``Gender`` as a binary integer (Male=1, Female=0).
        3. Fill missing values in ``Albumin_and_Globulin_Ratio`` with the median.
        4. Convert the target (``Dataset``) to binary: 1 → 1 (liver patient),
           2 → 0 (healthy).
        5. Standard-scale all feature columns.
        6. Split into train / test sets.

    Args:
        df: Raw DataFrame returned by :func:`load_data`.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test, scaler)``.
    """
    df = df.drop_duplicates().copy()

    # Encode gender
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Impute missing values
    agr_median = df["Albumin_and_Globulin_Ratio"].median()
    df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
        agr_median
    )

    # Binary target: 1 = liver patient, 0 = healthy
    df["target"] = (df["Dataset"] == 1).astype(int)

    feature_cols = [c for c in COLUMN_NAMES if c != "Dataset"]
    X = df[feature_cols].values.astype(float)
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names used after preprocessing."""
    return [c for c in COLUMN_NAMES if c != "Dataset"]
