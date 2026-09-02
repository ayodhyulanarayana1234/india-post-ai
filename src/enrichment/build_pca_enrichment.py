from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PCA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset2.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
KEY_COLUMNS = ["State", "District"]
PCA_FILTER_COLUMNS = ["Level", "TRU"]
NAME_COLUMNS = ["State Name", "District Name"]


def require_columns(dataframe, columns, source_name):
    missing_columns = set(columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def build_pca_enrichment():
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"Dataset2 file not found: {PCA_PATH}")
    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(
            f"Geographic crosswalk file not found: {CROSSWALK_PATH}"
        )

    pca = pd.read_csv(PCA_PATH, low_memory=False)
    crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)

    require_columns(
        pca,
        KEY_COLUMNS + PCA_FILTER_COLUMNS,
        "Dataset2.csv",
    )
    require_columns(
        crosswalk,
        KEY_COLUMNS + NAME_COLUMNS,
        "geographic_crosswalk.csv",
    )

    pca_district_total = pca.loc[
        pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total")
    ].copy()
    input_rows = len(pca_district_total)
    if input_rows != 640:
        raise ValueError(
            f"Expected 640 PCA district-total rows, found {input_rows}."
        )

    crosswalk_rows = len(crosswalk)
    if crosswalk_rows != 640:
        raise ValueError(
            f"Expected 640 geographic crosswalk rows, found {crosswalk_rows}."
        )

    pca_duplicate_keys = int(pca_district_total.duplicated(KEY_COLUMNS).sum())
    crosswalk_duplicate_keys = int(crosswalk.duplicated(KEY_COLUMNS).sum())
    if pca_duplicate_keys != 0:
        raise ValueError(
            f"Dataset2 contains duplicate State + District keys: "
            f"{pca_duplicate_keys}"
        )
    if crosswalk_duplicate_keys != 0:
        raise ValueError(
            "geographic_crosswalk.csv contains duplicate State + District "
            f"keys: {crosswalk_duplicate_keys}"
        )

    original_codes = pca_district_total[KEY_COLUMNS].reset_index(drop=True)
    mapped = pca_district_total.merge(
        crosswalk[KEY_COLUMNS + NAME_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    mapped_rows = int(mapped["_merge"].eq("both").sum())
    unmapped_rows = int(mapped["_merge"].eq("left_only").sum())
    output_rows = len(mapped)
    duplicate_output_keys = int(mapped.duplicated(KEY_COLUMNS).sum())
    missing_state_names = int(mapped["State Name"].isna().sum())
    missing_district_names = int(mapped["District Name"].isna().sum())

    if output_rows != input_rows:
        raise ValueError(
            "Unexpected row multiplication or loss: "
            f"input={input_rows}, output={output_rows}"
        )
    if mapped_rows != input_rows or unmapped_rows != 0:
        raise ValueError(
            "Incomplete geographic mapping: "
            f"mapped={mapped_rows}, unmapped={unmapped_rows}"
        )
    if duplicate_output_keys != 0:
        raise ValueError(
            "Output contains duplicate State + District keys: "
            f"{duplicate_output_keys}"
        )
    if not mapped[KEY_COLUMNS].reset_index(drop=True).equals(original_codes):
        raise ValueError("State or District code values changed during the join.")
    if missing_state_names != 0:
        raise ValueError(
            f"Output contains missing State Names: {missing_state_names}"
        )
    if missing_district_names != 0:
        raise ValueError(
            f"Output contains missing District Names: {missing_district_names}"
        )

    result = mapped.drop(columns="_merge")
    if len(result.columns) < len(pca_district_total.columns) + len(NAME_COLUMNS):
        raise ValueError("One or more PCA demographic columns were dropped.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=== PCA Demographic Enrichment Validation Report ===")
    print(f"Input PCA district-total rows: {input_rows}")
    print(f"Crosswalk rows: {crosswalk_rows}")
    print(f"Output rows: {output_rows}")
    print(f"Unmapped rows: {unmapped_rows}")
    print(f"Duplicate State + District keys: {duplicate_output_keys}")
    print(f"Missing State Names: {missing_state_names}")
    print(f"Missing District Names: {missing_district_names}")
    print("Unexpected row multiplication: no")
    print("State and District codes unchanged: yes")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pca_enrichment()
