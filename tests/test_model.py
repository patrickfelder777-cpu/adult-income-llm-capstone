"""Validation tests for the trained Adult Income model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import download_adult_dataset
from src.preprocess import prepare_training_data
from src.train import train_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_test_config() -> dict[str, Any]:
    """Load the project configuration for model testing."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    return config


@pytest.fixture(scope="session")
def trained_model() -> Pipeline:
    """
    Load the trained model.

    If the model artifact does not exist, run the training workflow
    before loading it.
    """
    if not MODEL_PATH.exists():
        train_models()

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "The saved model must be a scikit-learn Pipeline."
        )

    return model


@pytest.fixture(scope="session")
def test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the same held-out test split used during training."""
    config = load_test_config()

    if not DATA_PATH.exists():
        download_adult_dataset(
            output_path=DATA_PATH
        )

    dataframe = pd.read_csv(DATA_PATH)

    features, target = prepare_training_data(
        dataframe=dataframe,
        target_column=config["data"]["target_column"],
    )

    _, features_test, _, target_test = train_test_split(
        features,
        target,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["project"]["random_state"]),
        stratify=target,
    )

    return features_test, target_test


def test_model_prediction_shape_and_type(
    trained_model: Pipeline,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions should have the correct shape and binary values."""
    features_test, _ = test_data
    sample = features_test.head(25)

    predictions = trained_model.predict(sample)

    assert predictions.shape == (25,)
    assert np.issubdtype(
        predictions.dtype,
        np.integer,
    )
    assert set(np.unique(predictions)).issubset(
        {0, 1}
    )


def test_model_probability_shape_and_range(
    trained_model: Pipeline,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Probabilities should be valid for both income classes."""
    features_test, _ = test_data
    sample = features_test.head(25)

    probabilities = trained_model.predict_proba(sample)

    assert probabilities.shape == (25, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    probability_sums = probabilities.sum(axis=1)

    assert np.allclose(
        probability_sums,
        1.0,
        atol=1e-7,
    )


def test_model_meets_minimum_roc_auc(
    trained_model: Pipeline,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The selected model should achieve acceptable test ROC-AUC."""
    features_test, target_test = test_data

    positive_probabilities = trained_model.predict_proba(
        features_test
    )[:, 1]

    roc_auc = roc_auc_score(
        target_test,
        positive_probabilities,
    )

    assert roc_auc >= 0.85, (
        f"Expected ROC-AUC of at least 0.85, "
        f"but received {roc_auc:.4f}."
    )