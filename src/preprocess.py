"""Preprocessing utilities for the Adult Income dataset."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1,
}


def normalize_column_name(column_name: str) -> str:
    """Convert a column name to lowercase snake_case."""
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    return normalized


def clean_categorical_value(value: object) -> object:
    """
    Clean one categorical value.

    Missing values, question marks, and blank strings are converted
    to NumPy NaN so that scikit-learn's SimpleImputer can process them.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        cleaned_value = value.strip()

        if cleaned_value in {"", "?"}:
            return np.nan

        return cleaned_value

    return value


def clean_target_value(value: object) -> object:
    """Clean one target value and remove a possible trailing period."""
    if pd.isna(value):
        return np.nan

    cleaned_value = str(value).strip().rstrip(".")

    if cleaned_value in {"", "?"}:
        return np.nan

    return cleaned_value


def clean_adult_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "income",
) -> pd.DataFrame:
    """
    Clean the Adult Income dataframe without modifying the original.

    Cleaning includes:
    - Standardizing column names
    - Trimming whitespace from categorical values
    - Replacing '?', blank strings, pd.NA, and None with np.nan
    - Removing trailing periods from target labels
    - Encoding the target as 0 or 1
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    cleaned = dataframe.copy(deep=True)

    cleaned.columns = [
        normalize_column_name(column)
        for column in cleaned.columns
    ]

    normalized_target = normalize_column_name(target_column)

    if normalized_target not in cleaned.columns:
        raise ValueError(
            f"Target column '{normalized_target}' was not found."
        )

    categorical_columns = cleaned.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        cleaned[column] = (
            cleaned[column]
            .map(clean_categorical_value)
            .astype("object")
        )

    cleaned[normalized_target] = (
        cleaned[normalized_target]
        .map(clean_target_value)
        .astype("object")
    )

    observed_target_values = set(
        cleaned[normalized_target].dropna().unique()
    )

    expected_target_values = set(TARGET_MAPPING.keys())
    unexpected_values = observed_target_values - expected_target_values

    if unexpected_values:
        raise ValueError(
            "Unexpected target values found: "
            f"{sorted(unexpected_values)}"
        )

    cleaned[normalized_target] = cleaned[
        normalized_target
    ].map(TARGET_MAPPING)

    cleaned = cleaned.dropna(
        subset=[normalized_target]
    ).reset_index(drop=True)

    cleaned[normalized_target] = cleaned[
        normalized_target
    ].astype("int64")

    return cleaned


def prepare_training_data(
    dataframe: pd.DataFrame,
    target_column: str = "income",
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and separate features from the target."""
    cleaned = clean_adult_dataframe(
        dataframe=dataframe,
        target_column=target_column,
    )

    normalized_target = normalize_column_name(target_column)

    features = cleaned.drop(columns=[normalized_target])
    target = cleaned[normalized_target].copy()

    return features, target


def get_feature_groups(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numerical and categorical feature names."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    numerical_features = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = features.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if not numerical_features:
        raise ValueError("No numerical features were found.")

    if not categorical_features:
        raise ValueError("No categorical features were found.")

    return numerical_features, categorical_features


def build_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build preprocessing pipelines for numerical and categorical data.

    Numerical features:
    - Median imputation
    - Standard scaling

    Categorical features:
    - Most-frequent imputation
    - One-hot encoding
    """
    numerical_features, categorical_features = get_feature_groups(
        features
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    missing_values=np.nan,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=np.nan,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor