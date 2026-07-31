"""Train Adult Income models and track experiments with MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import calculate_classification_metrics
from src.preprocess import build_preprocessor, prepare_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the YAML project configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {
        "project",
        "data",
        "mlflow",
        "models",
    }

    missing_sections = required_sections - set(config)

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    if len(config["models"]) < 5:
        raise ValueError(
            "At least five model configurations are required."
        )

    return config


def resolve_project_path(path_value: str) -> Path:
    """Convert a configured relative path into a project path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_model(
    model_type: str,
    parameters: dict[str, Any],
    random_state: int,
) -> Any:
    """Create a supported classification model."""
    model_parameters = parameters.copy()

    if model_type == "logistic_regression":
        model_parameters.setdefault(
            "random_state",
            random_state,
        )
        return LogisticRegression(**model_parameters)

    if model_type == "random_forest":
        model_parameters.setdefault(
            "random_state",
            random_state,
        )
        return RandomForestClassifier(**model_parameters)

    if model_type == "gradient_boosting":
        model_parameters.setdefault(
            "random_state",
            random_state,
        )
        return GradientBoostingClassifier(**model_parameters)

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def flatten_model_parameters(
    model_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Prepare model parameters for MLflow logging."""
    logged_parameters: dict[str, Any] = {
        "model_type": model_type,
    }

    for parameter_name, parameter_value in parameters.items():
        logged_parameters[
            f"model__{parameter_name}"
        ] = parameter_value

    return logged_parameters


def display_run_results(
    run_name: str,
    metrics: dict[str, float],
) -> None:
    """Print one experiment's evaluation metrics."""
    print("\n" + "=" * 65)
    print(f"RUN: {run_name}")
    print("=" * 65)

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name:>10}: "
            f"{metric_value:.4f}"
        )


def create_mlflow_input_example(
    features_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a realistic input example for MLflow model logging.

    Numeric columns are converted to float64 so they can support
    missing values during future inference.
    """
    input_example = features_test.head(5).copy()

    numeric_columns = input_example.select_dtypes(
        include=["number"]
    ).columns

    input_example[numeric_columns] = input_example[
        numeric_columns
    ].astype("float64")

    return input_example


def train_models(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """
    Train all configured models and track them with MLflow.

    Returns
    -------
    pandas.DataFrame
        Metrics and MLflow identifiers for every experiment.
    """
    config = load_config(config_path)

    random_state = int(
        config["project"]["random_state"]
    )

    target_column = config["data"]["target_column"]
    test_size = float(config["data"]["test_size"])

    raw_data_path = resolve_project_path(
        config["data"]["raw_path"]
    )

    if not raw_data_path.exists():
        raise FileNotFoundError(
            "Raw dataset was not found. Run "
            "'python -m src.data_loader' first."
        )

    dataframe = pd.read_csv(raw_data_path)

    features, target = prepare_training_data(
        dataframe=dataframe,
        target_column=target_column,
    )

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    selection_metric = config["mlflow"]["selection_metric"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    models_directory = PROJECT_ROOT / "models"
    reports_directory = PROJECT_ROOT / "reports"

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_description = {
        "dataset_name": "UCI Adult Income",
        "dataset_source": "UCI Machine Learning Repository",
        "dataset_rows": int(dataframe.shape[0]),
        "original_feature_count": int(features.shape[1]),
        "target_column": target_column,
        "positive_class": ">50K",
        "negative_class": "<=50K",
        "train_rows": int(features_train.shape[0]),
        "test_rows": int(features_test.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "data_version": "UCI Adult dataset ID 2",
    }

    experiment_results: list[dict[str, Any]] = []

    best_metric_value = float("-inf")
    best_pipeline: Pipeline | None = None
    best_run_information: dict[str, Any] | None = None

    for model_config in config["models"]:
        run_name = model_config["run_name"]
        model_type = model_config["model_type"]
        model_parameters = model_config["parameters"]

        model = build_model(
            model_type=model_type,
            parameters=model_parameters,
            random_state=random_state,
        )

        preprocessor = build_preprocessor(
            features_train
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print(
            f"\nTraining: {run_name} "
            f"({model_type})..."
        )

        with mlflow.start_run(
            run_name=run_name
        ) as active_run:
            pipeline.fit(
                features_train,
                target_train,
            )

            predictions = pipeline.predict(
                features_test
            )

            probabilities = pipeline.predict_proba(
                features_test
            )[:, 1]

            metrics = calculate_classification_metrics(
                y_true=target_test,
                y_pred=predictions,
                y_probability=probabilities,
            )

            logged_parameters = flatten_model_parameters(
                model_type=model_type,
                parameters=model_parameters,
            )

            logged_parameters.update(
                {
                    "random_state": random_state,
                    "test_size": test_size,
                    "target_column": target_column,
                    "selection_metric": selection_metric,
                }
            )

            mlflow.log_params(logged_parameters)
            mlflow.log_metrics(metrics)

            mlflow.log_dict(
                dataset_description,
                "dataset/dataset_description.json",
            )

            mlflow.log_dict(
                model_config,
                "configuration/model_configuration.json",
            )

            input_example = create_mlflow_input_example(
                features_test
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            run_id = active_run.info.run_id

            run_result = {
                "run_name": run_name,
                "run_id": run_id,
                "model_type": model_type,
                **metrics,
            }

            experiment_results.append(run_result)

            display_run_results(
                run_name=run_name,
                metrics=metrics,
            )

            current_metric_value = metrics[
                selection_metric
            ]

            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_pipeline = pipeline
                best_run_information = run_result.copy()

    results_dataframe = pd.DataFrame(
        experiment_results
    ).sort_values(
        by=selection_metric,
        ascending=False,
    )

    results_path = (
        reports_directory
        / "training_results.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    if (
        best_pipeline is None
        or best_run_information is None
    ):
        raise RuntimeError(
            "Training completed without selecting a best model."
        )

    best_model_path = (
        models_directory
        / "best_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        best_model_path,
    )

    metadata_path = (
        models_directory
        / "best_model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            best_run_information,
            metadata_file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print("\nExperiment ranking:")

    print(
        results_dataframe[
            [
                "run_name",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nBest run: "
        f"{best_run_information['run_name']}"
    )

    print(
        f"Best {selection_metric}: "
        f"{best_metric_value:.4f}"
    )

    print(
        f"Best model saved to: "
        f"{best_model_path}"
    )

    print(
        f"Best model metadata saved to: "
        f"{metadata_path}"
    )

    print(
        f"Results saved to: "
        f"{results_path}"
    )

    return results_dataframe


if __name__ == "__main__":
    train_models()