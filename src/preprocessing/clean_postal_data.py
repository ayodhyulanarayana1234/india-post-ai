import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "Dataset1.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "postal_network.csv"


def clean_coordinates(df):
    """
    Clean and validate latitude/longitude values.

    Categories:
    - Normal coordinates: keep
    - Reversed coordinates: swap
    - Missing/zero: set to NaN
    - Other corrupted values: set to NaN
    """

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    valid_lat = df["latitude"].between(6, 38)
    valid_lon = df["longitude"].between(68, 98)

    normal = valid_lat & valid_lon

    reversed_coords = (
        df["latitude"].between(68, 98)
        & df["longitude"].between(6, 38)
    )

    missing_or_zero = (
        df["latitude"].isna()
        | df["longitude"].isna()
        | (df["latitude"] == 0)
        | (df["longitude"] == 0)
    )

    # Swap only strongly identified reversed coordinates
    df.loc[reversed_coords, ["latitude", "longitude"]] = (
        df.loc[reversed_coords, ["longitude", "latitude"]].values
    )

    # Recalculate validity after swapping
    valid_after_swap = (
        df["latitude"].between(6, 38)
        & df["longitude"].between(68, 98)
    )

    # Everything that is not valid after processing becomes missing
    df.loc[~valid_after_swap, ["latitude", "longitude"]] = pd.NA

    return df


def clean_data():
    """Load, clean and save the postal network dataset."""

    print(f"Loading dataset: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    print(f"Original shape: {df.shape}")

    # Remove exact duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    print(f"Removed duplicate rows: {duplicates}")

    # Clean coordinates
    df = clean_coordinates(df)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Save processed dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed shape: {df.shape}")
    print(f"Saved processed dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_data()