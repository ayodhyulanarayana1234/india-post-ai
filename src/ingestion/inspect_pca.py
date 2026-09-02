from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PCA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset2.csv"
DATASET3_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset3.xlsx"
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"


def matching_columns(columns, patterns, exclude_patterns=None):
    """Return original column names matching the supplied patterns."""
    exclude_patterns = exclude_patterns or []
    matches = []

    for column in columns:
        name = str(column).strip().lower()

        if any(re.search(pattern, name) for pattern in patterns):
            if not any(re.search(pattern, name) for pattern in exclude_patterns):
                matches.append(column)

    return matches


def preferred_column(columns, patterns):
    """Select the most likely column while retaining the exact original name."""
    matches = matching_columns(columns, patterns)

    if not matches:
        return None

    for column in matches:
        normalized = str(column).strip().lower()
        if normalized in {"state", "district"}:
            return column

    return matches[0]


def print_join_key_check(pca_df, pca_state_col, pca_district_col):
    print("\n=== Geographic Join-Key Check ===")

    if not POSTAL_PATH.exists():
        print(f"Postal network file not found: {POSTAL_PATH}")
        return

    postal_df = pd.read_csv(POSTAL_PATH, low_memory=False)
    print(f"Loaded postal network for inspection only: {POSTAL_PATH}")

    postal_state_col = preferred_column(
        postal_df.columns,
        [r"\bstate\b", r"state[_\s-]*(name|code)?"],
    )
    postal_district_col = preferred_column(
        postal_df.columns,
        [r"\bdistrict\b", r"district[_\s-]*(name|code)?"],
    )

    print(f"Postal State column: {postal_state_col}")
    print(f"Postal District column: {postal_district_col}")

    if not pca_state_col or not postal_state_col:
        print("State join-key comparison cannot be performed.")
    else:
        pca_states = {
            str(value).strip().casefold()
            for value in pca_df[pca_state_col].dropna()
        }
        postal_states = {
            str(value).strip().casefold()
            for value in postal_df[postal_state_col].dropna()
        }
        state_overlap = pca_states & postal_states

        print(f"PCA unique states: {len(pca_states)}")
        print(f"Postal unique states: {len(postal_states)}")
        print(f"Matching state values: {len(state_overlap)}")
        print(f"State key potentially usable: {bool(state_overlap)}")

    if not pca_district_col or not postal_district_col:
        print("District join-key comparison cannot be performed.")
    else:
        pca_districts = {
            str(value).strip().casefold()
            for value in pca_df[pca_district_col].dropna()
        }
        postal_districts = {
            str(value).strip().casefold()
            for value in postal_df[postal_district_col].dropna()
        }
        district_overlap = pca_districts & postal_districts

        print(f"PCA unique districts: {len(pca_districts)}")
        print(f"Postal unique districts: {len(postal_districts)}")
        print(f"Matching district values: {len(district_overlap)}")
        print(f"District key potentially usable: {bool(district_overlap)}")

    if (
        pca_state_col
        and pca_district_col
        and postal_state_col
        and postal_district_col
    ):
        pca_pairs = {
            (
                str(state).strip().casefold(),
                str(district).strip().casefold(),
            )
            for state, district in zip(
                pca_df[pca_state_col],
                pca_df[pca_district_col],
            )
            if pd.notna(state) and pd.notna(district)
        }

        postal_pairs = {
            (
                str(state).strip().casefold(),
                str(district).strip().casefold(),
            )
            for state, district in zip(
                postal_df[postal_state_col],
                postal_df[postal_district_col],
            )
            if pd.notna(state) and pd.notna(district)
        }

        pair_overlap = pca_pairs & postal_pairs

        print(f"Matching State-District pairs: {len(pair_overlap)}")
        print(
            "State-District composite key potentially usable: "
            f"{bool(pair_overlap)}"
        )

    print("No datasets were merged or modified.")


def main():
    inspect_dataset3()


def inspect_dataset3():
    print("=== Dataset3 Geographic Crosswalk Inspection ===")
    print(f"Excel file: {DATASET3_PATH}")

    if not DATASET3_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {DATASET3_PATH}")

    workbook = pd.ExcelFile(DATASET3_PATH)
    print(f"\nSheet names: {workbook.sheet_names}")

    sheet_tables = {}
    geographic_sheet_names = []

    for sheet_name in workbook.sheet_names:
        sheet_df = pd.read_excel(DATASET3_PATH, sheet_name=sheet_name)
        sheet_tables[sheet_name] = sheet_df

        print(f"\n--- Sheet: {sheet_name} ---")
        print(f"Rows: {sheet_df.shape[0]}")
        print(f"Columns: {sheet_df.shape[1]}")
        print(f"Column names: {list(sheet_df.columns)}")

        if not sheet_df.empty:
            print("First 10 rows:")
            print(sheet_df.head(10).to_string(index=False))

        geographic_columns = matching_columns(
            sheet_df.columns,
            [r"state", r"district", r"code", r"town[-\s]?village"],
        )
        if geographic_columns:
            geographic_sheet_names.append(sheet_name)

    print("\n=== Geographic Location-Code Sheet ===")
    if not geographic_sheet_names:
        print("No sheet with geographic location-code information was found.")
        return

    print(f"Sheet(s): {geographic_sheet_names}")
    crosswalk_sheet = sheet_tables[geographic_sheet_names[0]]
    inspect_dataset3_columns_and_keys(crosswalk_sheet)


def inspect_dataset3_columns_and_keys(crosswalk_df):
    state_code_col = preferred_column(
        crosswalk_df.columns,
        [r"^state\s*code$", r"state.*code"],
    )
    state_name_col = preferred_column(
        crosswalk_df.columns,
        [r"^state\s*name$", r"state.*name"],
    )
    district_code_col = preferred_column(
        crosswalk_df.columns,
        [r"^district\s*code$", r"district.*code"],
    )
    district_name_col = preferred_column(
        crosswalk_df.columns,
        [r"^district\s*name$", r"district.*name"],
    )
    location_name_col = preferred_column(
        crosswalk_df.columns,
        [r"town[-\s]?village.*name", r"location.*name"],
    )

    if state_name_col is None and location_name_col is not None:
        state_name_col = (
            f"{location_name_col} (where District Code == 0 and "
            "Town-Village Code == 0)"
        )
    if district_name_col is None and location_name_col is not None:
        district_name_col = (
            f"{location_name_col} (where District Code != 0, "
            "Sub District Code == 0, and Town-Village Code == 0)"
        )

    identified_columns = {
        "State code": state_code_col,
        "State name": state_name_col,
        "District code": district_code_col,
        "District name": district_name_col,
    }

    print("\n=== Geographic Column Identification ===")
    for role, column in identified_columns.items():
        print(f"{role}: {column}")

    required_columns = [state_code_col, district_code_col]
    if any(column is None for column in required_columns):
        print("\nUnique geographic counts cannot be calculated.")
        return

    key_columns = [state_code_col, district_code_col]
    unique_keys = crosswalk_df[key_columns].drop_duplicates()
    duplicate_rows = crosswalk_df.duplicated(key_columns, keep=False)

    print("\n=== Dataset3 Geographic Key Counts ===")
    print(f"Unique State codes: {crosswalk_df[state_code_col].nunique(dropna=True)}")
    print(
        "Unique District codes: "
        f"{crosswalk_df[district_code_col].nunique(dropna=True)}"
    )
    print(f"Unique State + District combinations: {len(unique_keys)}")
    print(f"State + District unique across rows: {not duplicate_rows.any()}")
    print(f"Rows participating in duplicate combinations: {int(duplicate_rows.sum())}")

    inspect_dataset2_crosswalk(
        crosswalk_df,
        state_code_col,
        district_code_col,
    )


def inspect_dataset2_crosswalk(crosswalk_df, state_code_col, district_code_col):
    print("\n=== Dataset2 DISTRICT + Total Crosswalk Check ===")

    if not PCA_PATH.exists():
        print(f"Dataset2 source file not found: {PCA_PATH}")
        return

    pca_df = pd.read_csv(PCA_PATH, low_memory=False)
    district_total_df = pca_df.loc[
        pca_df["Level"].eq("DISTRICT") & pca_df["TRU"].eq("Total")
    ]

    pca_key_rows = district_total_df.dropna(subset=["State", "District"])
    crosswalk_key_rows = crosswalk_df.dropna(
        subset=[state_code_col, district_code_col]
    )
    pca_keys = set(zip(pca_key_rows["State"], pca_key_rows["District"]))
    crosswalk_keys = set(
        zip(
            crosswalk_key_rows[state_code_col],
            crosswalk_key_rows[district_code_col],
        )
    )
    matched_keys = pca_keys & crosswalk_keys
    missing_keys = pca_keys - crosswalk_keys

    print(f"Dataset2 DISTRICT + Total records: {len(district_total_df)}")
    print(f"Dataset2 geographic keys: {len(pca_keys)}")
    print(f"Matching crosswalk keys: {len(matched_keys)}")
    print(f"Unmatched Dataset2 keys: {len(missing_keys)}")
    print(f"All 640 records can be mapped by State + District codes: {not missing_keys}")
    print("No datasets were merged or modified.")


def inspect_district_total_records(pca_df):
    print("\n=== DISTRICT / Total Record Inspection ===")

    required_columns = {"State", "District", "Name", "Level", "TRU"}
    missing_columns = required_columns - set(pca_df.columns)

    if missing_columns:
        print(f"Required columns not found: {sorted(missing_columns)}")
        return

    district_total_df = pca_df.loc[
        pca_df["Level"].eq("DISTRICT") & pca_df["TRU"].eq("Total")
    ].copy()

    print(f"Filtered rows: {len(district_total_df)}")
    print("\nFirst 30 district-total records:")
    print(
        district_total_df[["State", "District", "Name", "Level", "TRU"]]
        .head(30)
        .to_string(index=False)
    )

    print(f"\nUnique State codes: {district_total_df['State'].nunique(dropna=True)}")
    print(
        "Unique District codes: "
        f"{district_total_df['District'].nunique(dropna=True)}"
    )

    geographic_keys = ["State", "District"]
    unique_combinations = district_total_df[geographic_keys].drop_duplicates()
    duplicate_mask = district_total_df.duplicated(geographic_keys, keep=False)
    duplicate_combinations = district_total_df.loc[
        duplicate_mask, geographic_keys
    ].drop_duplicates()

    print(f"Unique State + District combinations: {len(unique_combinations)}")
    print(
        "Duplicate State + District combinations: "
        f"{duplicate_combinations.shape[0]} unique combinations "
        f"({duplicate_mask.sum()} records)"
    )

    print("\nFive Name examples for each State code:")
    for state_code, state_rows in district_total_df.groupby("State", sort=True):
        names = state_rows["Name"].head(5).tolist()
        print(f"State={state_code!r}: {names}")

    names = district_total_df["Name"].dropna().astype(str)
    contains_total = names.str.contains(r"\btotal\b", case=False, regex=True)
    contains_state_and_district = names.str.contains(
        r"\bstate\b.*\bdistrict\b|\bdistrict\b.*\bstate\b",
        case=False,
        regex=True,
    )

    if contains_state_and_district.all():
        name_format = "state name + district name"
    elif contains_total.all():
        name_format = 'district name + "Total"'
    elif not contains_total.any() and not contains_state_and_district.any():
        name_format = "district name only"
    else:
        name_format = "another format or mixed formats"

    print("\nName column format check:")
    print(f"Appears to contain: {name_format}")
    print(f'Names containing "Total": {int(contains_total.sum())}')
    print(
        "Names containing both state and district labels: "
        f"{int(contains_state_and_district.sum())}"
    )

    print("\n20 representative Name values:")
    print(names.head(20).to_string(index=False))


def inspect_level_and_tru(pca_df):
    print("\n=== Level and TRU Inspection ===")

    required_columns = {"Level", "TRU"}
    missing_columns = required_columns - set(pca_df.columns)

    if missing_columns:
        print(f"Required columns not found: {sorted(missing_columns)}")
        return

    print("\nUnique values and counts for Level:")
    print(pca_df["Level"].value_counts(dropna=False).to_string())

    print("\nUnique values and counts for TRU:")
    print(pca_df["TRU"].value_counts(dropna=False).to_string())

    print("\nUnique Level/TRU combinations:")
    level_tru_counts = (
        pca_df.groupby(["Level", "TRU"], dropna=False)
        .size()
        .reset_index(name="RowCount")
        .sort_values(["Level", "TRU"], na_position="last")
    )
    print(level_tru_counts.to_string(index=False))

    print("\nRows for each Level value:")
    level_counts = pca_df["Level"].value_counts(dropna=False)
    for level, count in level_counts.items():
        print(f"Level={level!r}: {count} rows")

    level_text = pca_df["Level"].astype("string").str.strip().str.casefold()
    district_mask = level_text.str.contains(
        r"district|dist\b| तहसील |tehsil",
        regex=True,
        na=False,
    )

    district_rows = pca_df.loc[district_mask]

    print("\n=== Potential District-Level Records ===")
    if district_rows.empty:
        print(
            "No Level values explicitly containing district-related text were found."
        )
        print("Level values available for manual review:")
        print(pca_df["Level"].drop_duplicates().to_string(index=False))
        return

    print(f"Potential district-level rows found: {len(district_rows)}")
    print("\nSample potential district-level rows:")

    display_columns = [
        column
        for column in ["State", "District", "Level", "Name", "TRU"]
        if column in district_rows.columns
    ]

    print(district_rows[display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()