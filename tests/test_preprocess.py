"""Tests for the Adult Income preprocessing pipeline."""

import copy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.preprocess import (
    build_preprocessor,
    clean_adult_dataframe,
    get_feature_groups,
    prepare_training_data,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a small Adult Income-style dataframe for testing."""
    return pd.DataFrame(
        {
            "age": [25, 38, 45, 52],
            "hours-per-week": [40, 50, np.nan, 35],
            "workclass": ["Private", " ?", "Self-emp-not-inc", None],
            "education": ["Bachelors", "HS-grad", " Masters ", ""],
            "income": ["<=50K", ">50K.", ">50K", "<=50K."],
        }
    )


def test_cleaning_handles_missing_categorical_values() -> None:
    """Question marks, blanks, and None should become missing values."""
    dataframe = create_sample_dataframe()

    cleaned = clean_adult_dataframe(dataframe)

    assert pd.isna(cleaned.loc[1, "workclass"])
    assert pd.isna(cleaned.loc[3, "workclass"])
    assert pd.isna(cleaned.loc[3, "education"])


def test_cleaning_trims_categorical_whitespace() -> None:
    """Whitespace should be removed from categorical values."""
    dataframe = create_sample_dataframe()

    cleaned = clean_adult_dataframe(dataframe)

    assert cleaned.loc[2, "education"] == "Masters"


def test_target_is_encoded_as_binary() -> None:
    """Income labels should be converted to integers 0 and 1."""
    dataframe = create_sample_dataframe()

    cleaned = clean_adult_dataframe(dataframe)

    assert cleaned["income"].tolist() == [0, 1, 1, 0]
    assert cleaned["income"].dtype == "int64"


def test_cleaning_does_not_modify_original_dataframe() -> None:
    """The preprocessing function must preserve the original dataframe."""
    dataframe = create_sample_dataframe()
    original = copy.deepcopy(dataframe)

    clean_adult_dataframe(dataframe)

    assert_frame_equal(dataframe, original)


def test_preprocessor_removes_all_missing_values() -> None:
    """The transformed feature matrix should not contain missing values."""
    dataframe = create_sample_dataframe()

    features, _ = prepare_training_data(dataframe)
    preprocessor = build_preprocessor(features)

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(dataframe)
    assert np.isnan(transformed).sum() == 0


def test_numerical_features_are_standardized() -> None:
    """Numerical transformed features should have means near zero."""
    dataframe = create_sample_dataframe()

    features, _ = prepare_training_data(dataframe)
    numerical_features, _ = get_feature_groups(features)

    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)

    transformed_numerical = transformed[:, : len(numerical_features)]
    numerical_means = transformed_numerical.mean(axis=0)

    assert np.allclose(numerical_means, 0.0, atol=1e-7)