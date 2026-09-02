from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET3_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset3.xlsx"
PCA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset2.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"

DATASET3_COLUMNS = {
    "state_code": "State Code",
    "district_code": "District Code",
    "subdistrict_code": "Sub District Code",
    "town_village_code": "Town-Village Code",
    "location_name": "Town-Village Name",
}
KEY_COLUMNS = ["State", "District"]
OUTPUT_COLUMNS = ["State", "District", "State Name", "District Name"]


def require_columns(dataframe, columns, source_name):
    missing_columns = set(columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def build_state_names(dataset3):
    columns = DATASET3_COLUMNS
    state_rows = dataset3.loc[
        dataset3[columns["district_code"]].eq(0)
        & dataset3[columns["subdistrict_code"]].eq(0)
        & dataset3[columns["town_village_code"]].eq(0),
        [columns["state_code"], columns["location_name"]],
    ].rename(
        columns={
            columns["state_code"]: "State",
            columns["location_name"]: "State Name",
        }
    )

    state_name_counts = state_rows.groupby("State")["State Name"].nunique(
        dropna=False
    )
    ambiguous_states = state_name_counts[state_name_counts != 1]
    if not ambiguous_states.empty:
        raise ValueError(
            "Dataset3 has missing or conflicting state names for codes: "
            f"{list(ambiguous_states.index)}"
        )

    duplicate_state_rows = state_rows.duplicated("State", keep=False)
    if duplicate_state_rows.any():
        raise ValueError("Dataset3 has duplicate state hierarchy rows.")

    return state_rows


def build_district_crosswalk(dataset3):
    columns = DATASET3_COLUMNS
    district_rows = dataset3.loc[
        dataset3[columns["district_code"]].ne(0)
        & dataset3[columns["subdistrict_code"]].eq(0)
        & dataset3[columns["town_village_code"]].eq(0),
        [
            columns["state_code"],
            columns["district_code"],
            columns["location_name"],
        ],
    ].rename(
        columns={
            columns["state_code"]: "State",
            columns["district_code"]: "District",
            columns["location_name"]: "District Name",
        }
    )

    duplicate_keys = district_rows.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Dataset3 has duplicate State + District keys after hierarchy "
            "reduction."
        )

    district_name_counts = district_rows.groupby(KEY_COLUMNS)["District Name"].nunique(
        dropna=False
    )
    ambiguous_districts = district_name_counts[district_name_counts != 1]
    if not ambiguous_districts.empty:
        raise ValueError(
            "Dataset3 has missing or conflicting district names for keys: "
            f"{list(ambiguous_districts.index)}"
        )

    state_names = build_state_names(dataset3)
    crosswalk = district_rows.merge(
        state_names,
        on="State",
        how="left",
        validate="many_to_one",
    )

    if crosswalk[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError("Dataset3 crosswalk contains missing geographic names.")

    return crosswalk[OUTPUT_COLUMNS]


def build_geographic_crosswalk():
    if not DATASET3_PATH.exists():
        raise FileNotFoundError(f"Dataset3 file not found: {DATASET3_PATH}")
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"Dataset2 file not found: {PCA_PATH}")

    dataset3 = pd.read_excel(DATASET3_PATH)
    require_columns(
        dataset3,
        DATASET3_COLUMNS.values(),
        "Dataset3.xlsx",
    )

    pca = pd.read_csv(PCA_PATH, low_memory=False)
    require_columns(pca, ["State", "District", "Level", "TRU"], "Dataset2.csv")

    pca_district_total = pca.loc[
        pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total"),
        KEY_COLUMNS,
    ].copy()
    input_rows = len(pca_district_total)
    if input_rows != 640:
        raise ValueError(
            f"Expected 640 PCA district-total rows, found {input_rows}."
        )

    if pca_district_total.duplicated(KEY_COLUMNS).any():
        raise ValueError("Dataset2 contains duplicate State + District keys.")

    crosswalk = build_district_crosswalk(dataset3)
    duplicate_crosswalk_keys = crosswalk.duplicated(KEY_COLUMNS).sum()
    if duplicate_crosswalk_keys != 0:
        raise ValueError(
            "Crosswalk reduction left duplicate State + District keys: "
            f"{duplicate_crosswalk_keys}"
        )

    mapped = pca_district_total.merge(
        crosswalk,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    mapped_rows = int(mapped["_merge"].eq("both").sum())
    unmapped_rows = int(mapped["_merge"].eq("left_only").sum())

    if len(mapped) != input_rows:
        raise ValueError(
            "Crosswalk join unexpectedly changed the PCA row count: "
            f"{input_rows} -> {len(mapped)}"
        )
    if mapped_rows != input_rows or unmapped_rows != 0:
        raise ValueError(
            "Geographic crosswalk is incomplete: "
            f"mapped={mapped_rows}, unmapped={unmapped_rows}"
        )

    if not mapped[KEY_COLUMNS].reset_index(drop=True).equals(
        pca_district_total[KEY_COLUMNS].reset_index(drop=True)
    ):
        raise ValueError("State or District codes changed during the crosswalk join.")

    result = mapped[OUTPUT_COLUMNS]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=== Geographic Crosswalk Validation Report ===")
    print(f"Dataset2 PCA district-total input rows: {input_rows}")
    print(f"Dataset3 crosswalk rows after reduction: {len(crosswalk)}")
    print(f"Duplicate State + District keys after reduction: {duplicate_crosswalk_keys}")
    print(f"Mapped rows: {mapped_rows}")
    print(f"Unmapped rows: {unmapped_rows}")
    print(f"Output rows: {len(result)}")
    print("Unexpected row multiplication: no")
    print("State and District codes consistent: yes")
    print(f"Saved geographic crosswalk: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_geographic_crosswalk()
