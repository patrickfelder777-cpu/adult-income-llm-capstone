"""Download and inspect the UCI Adult Income dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"


def download_adult_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, str]:
    """
    Download the Adult Income dataset from the UCI repository.

    Parameters
    ----------
    output_path:
        Location where the combined feature and target data will be saved.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The combined dataset and target-column name.
    """
    adult = fetch_ucirepo(id=2)

    features = adult.data.features.copy()
    targets = adult.data.targets.copy()

    if targets.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, but found {targets.shape[1]}."
        )

    target_column = targets.columns[0]
    dataset = pd.concat([features, targets], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    return dataset, target_column


def display_dataset_summary(
    dataset: pd.DataFrame,
    target_column: str,
) -> None:
    """Print basic information for initial dataset validation."""
    print("=" * 60)
    print("ADULT INCOME DATASET")
    print("=" * 60)

    print(f"Rows: {dataset.shape[0]:,}")
    print(f"Columns: {dataset.shape[1]}")
    print(f"Target column: {target_column}")

    print("\nColumn names:")
    for column in dataset.columns:
        print(f"  - {column}")

    print("\nTarget values:")
    print(dataset[target_column].value_counts(dropna=False))

    print("\nMissing values by column:")
    print(dataset.isna().sum().sort_values(ascending=False))

    print("\nFirst five rows:")
    print(dataset.head())


if __name__ == "__main__":
    dataframe, target = download_adult_dataset()
    display_dataset_summary(dataframe, target)

    print(f"\nDataset saved to:\n{DEFAULT_OUTPUT_PATH}")