"""Compare MLflow experiments and identify the best model run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the project YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if "mlflow" not in config:
        raise ValueError(
            "The configuration is missing the 'mlflow' section."
        )

    return config


def compare_experiments(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """
    Query MLflow runs, rank them, and identify the best experiment.

    Returns
    -------
    pandas.DataFrame
        A ranked table of completed experiment runs.
    """
    config = load_config(config_path)

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    selection_metric = config["mlflow"]["selection_metric"]

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' was not found. "
            "Run 'python -m src.train' first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
    )

    if runs.empty:
        raise ValueError(
            "No completed MLflow runs were found."
        )

    metric_column = f"metrics.{selection_metric}"

    if metric_column not in runs.columns:
        raise ValueError(
            f"Selection metric '{selection_metric}' was not logged."
        )

    required_metric_columns = [
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
    ]

    missing_metric_columns = [
        column
        for column in required_metric_columns
        if column not in runs.columns
    ]

    if missing_metric_columns:
        raise ValueError(
            "The following required metrics are missing: "
            f"{missing_metric_columns}"
        )

    ranked_runs = runs.sort_values(
        by=metric_column,
        ascending=False,
    ).reset_index(drop=True)

    display_columns = [
        "run_id",
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
        "artifact_uri",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in ranked_runs.columns
    ]

    comparison = ranked_runs[
        available_columns
    ].copy()

    comparison = comparison.rename(
        columns={
            "tags.mlflow.runName": "run_name",
            "params.model_type": "model_type",
            "metrics.accuracy": "accuracy",
            "metrics.precision": "precision",
            "metrics.recall": "recall",
            "metrics.f1": "f1",
            "metrics.roc_auc": "roc_auc",
        }
    )

    reports_directory = PROJECT_ROOT / "reports"

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        reports_directory
        / "mlflow_experiment_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    best_run = comparison.iloc[0]

    best_run_summary = {
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "selection_metric": selection_metric,
        "run_id": str(best_run["run_id"]),
        "run_name": str(best_run["run_name"]),
        "model_type": str(best_run["model_type"]),
        "accuracy": float(best_run["accuracy"]),
        "precision": float(best_run["precision"]),
        "recall": float(best_run["recall"]),
        "f1": float(best_run["f1"]),
        "roc_auc": float(best_run["roc_auc"]),
        "artifact_uri": str(best_run["artifact_uri"]),
    }

    best_run_path = (
        reports_directory
        / "best_mlflow_run.json"
    )

    with best_run_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            best_run_summary,
            output_file,
            indent=2,
        )

    print("\n" + "=" * 75)
    print("MLFLOW EXPERIMENT COMPARISON")
    print("=" * 75)

    print(
        f"\nExperiment: {experiment_name}"
    )

    print(
        f"Completed runs: {len(comparison)}"
    )

    print(
        f"Selection metric: {selection_metric}"
    )

    print("\nRanked experiments:")

    metric_display_columns = [
        "run_name",
        "model_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    print(
        comparison[
            metric_display_columns
        ].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("BEST MLFLOW RUN")
    print("=" * 75)

    print(
        f"Run name: {best_run_summary['run_name']}"
    )

    print(
        f"Model type: {best_run_summary['model_type']}"
    )

    print(
        f"Run ID: {best_run_summary['run_id']}"
    )

    print(
        f"Best {selection_metric}: "
        f"{best_run_summary[selection_metric]:.4f}"
    )

    print(
        f"Comparison saved to: {comparison_path}"
    )

    print(
        f"Best-run summary saved to: {best_run_path}"
    )

    return comparison


if __name__ == "__main__":
    compare_experiments()