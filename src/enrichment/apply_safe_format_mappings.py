from pathlib import Path

import pandas as pd

from diagnose_name_normalization import deterministic_normalize
from validate_geographic_candidates import (
    DATASET3_COLUMNS,
    build_candidate_records,
    normalize_for_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET3_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset3.xlsx"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"
EXPECTED_PCA_ROWS = 640
EXPECTED_APPROVED = 8
KEY_COLUMNS = ["State", "District"]
NAME_COLUMNS = ["State Name", "District Name"]
POSTAL_NAME_COLUMNS = ["Postal State", "Postal District"]


def require_columns(dataframe, columns, source_name):
    missing_columns = set(columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing_columns)}"
        )


def collect_approved_records(dataset3, pca, postal, crosswalk):
    candidates, _ = build_candidate_records(postal, pca)
    candidate_to_postal = {}
    postal_to_pca = {}
    for candidate in candidates:
        candidate_to_postal.setdefault(candidate["pca_key"], set()).add(
            candidate["postal_key"]
        )
        postal_to_pca.setdefault(candidate["postal_key"], set()).add(
            candidate["pca_key"]
        )

    one_to_one_candidates = [
        candidate
        for candidate in candidates
        if len(candidate_to_postal[candidate["pca_key"]]) == 1
        and len(postal_to_pca[candidate["postal_key"]]) == 1
    ]

    pca_district_total = pca.loc[
        pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total")
    ]
    pca_rows_by_name = {
        (
            normalize_for_comparison(row["State Name"]),
            normalize_for_comparison(row["District Name"]),
        ): row
        for _, row in pca_district_total.iterrows()
    }
    crosswalk_keys = set(zip(crosswalk["State"], crosswalk["District"]))
    columns = DATASET3_COLUMNS
    district_hierarchy = dataset3.loc[
        dataset3[columns["district_code"]].ne(0)
        & dataset3[columns["subdistrict_code"]].eq(0)
        & dataset3[columns["town_village_code"]].eq(0)
    ]
    census_names_by_code = {}
    for _, row in district_hierarchy.iterrows():
        code_key = (row[columns["state_code"]], row[columns["district_code"]])
        census_names_by_code.setdefault(code_key, []).append(
            row[columns["location_name"]]
        )

    approved = []
    for candidate in one_to_one_candidates:
        pca_key = (candidate["PCA State"], candidate["PCA District"])
        pca_row = pca_rows_by_name.get(pca_key)
        if pca_row is None:
            continue

        code_key = (pca_row["State"], pca_row["District"])
        census_names = census_names_by_code.get(code_key, [])
        if len(census_names) != 1:
            continue

        normalized_pca = deterministic_normalize(candidate["PCA District Name"])
        normalized_postal = deterministic_normalize(
            candidate["Candidate Postal District"]
        )
        normalized_census = deterministic_normalize(census_names[0])
        pca_code_unique = (
            pca_district_total[["State", "District"]]
            .eq([pca_row["State"], pca_row["District"]])
            .all(axis=1)
            .sum()
            == 1
        )
        conditions = {
            "A State code unchanged": pca_row["State"] == code_key[0],
            "B District code unchanged": pca_row["District"] == code_key[1],
            "C Census hierarchy unique": code_key in crosswalk_keys,
            "D PCA code unique": pca_code_unique,
            "E Postal relationship one-to-one": (
                len(candidate_to_postal[candidate["pca_key"]]) == 1
                and len(postal_to_pca[candidate["postal_key"]]) == 1
            ),
            "F Postal normalized equals Census": normalized_postal == normalized_census,
            "G PCA normalized equals Census": normalized_pca == normalized_census,
            "H No competing postal candidate": len(candidate_to_postal[candidate["pca_key"]]) == 1,
        }
        if all(conditions.values()):
            approved.append(
                {
                    "candidate": candidate,
                    "pca_row": pca_row,
                    "code_key": code_key,
                    "census_name": census_names[0],
                    "conditions": conditions,
                }
            )

    if len(approved) != EXPECTED_APPROVED:
        raise ValueError(
            f"Expected {EXPECTED_APPROVED} SAFE_FORMAT_MAPPING records, "
            f"reproduced {len(approved)}."
        )
    return approved, pca_district_total


def apply_safe_format_mappings():
    for path, label in [
        (DATASET3_PATH, "Dataset3"),
        (PCA_PATH, "PCA enrichment"),
        (POSTAL_PATH, "postal network"),
        (CROSSWALK_PATH, "geographic crosswalk"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    dataset3 = pd.read_excel(DATASET3_PATH)
    pca = pd.read_csv(PCA_PATH, low_memory=False)
    postal = pd.read_csv(POSTAL_PATH, low_memory=False)
    original_crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)

    require_columns(dataset3, DATASET3_COLUMNS.values(), "Dataset3.xlsx")
    require_columns(pca, KEY_COLUMNS + ["Level", "TRU", "State Name", "District Name"], "pca_district_enrichment.csv")
    require_columns(postal, ["statename", "district"], "postal_network.csv")
    require_columns(original_crosswalk, KEY_COLUMNS + NAME_COLUMNS, "geographic_crosswalk.csv")

    original_rows = len(original_crosswalk)
    original_keys = original_crosswalk[KEY_COLUMNS].copy()
    if original_rows != EXPECTED_PCA_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_PCA_ROWS} original crosswalk rows, found {original_rows}."
        )
    if original_crosswalk.duplicated(KEY_COLUMNS).any():
        raise ValueError("Original geographic crosswalk contains duplicate keys.")

    approved, pca_district_total = collect_approved_records(
        dataset3,
        pca,
        postal,
        original_crosswalk,
    )
    approved_keys = {item["code_key"] for item in approved}

    final_crosswalk = original_crosswalk.copy()
    for column in POSTAL_NAME_COLUMNS:
        if column not in final_crosswalk.columns:
            final_crosswalk[column] = pd.NA

    for item in approved:
        state_code, district_code = item["code_key"]
        matching_rows = (
            final_crosswalk["State"].eq(state_code)
            & final_crosswalk["District"].eq(district_code)
        )
        if int(matching_rows.sum()) != 1:
            raise ValueError(
                f"Approved key {item['code_key']!r} is not unique in the crosswalk."
            )
        candidate = item["candidate"]
        final_crosswalk.loc[matching_rows, "Postal State"] = candidate[
            "Candidate Postal State"
        ]
        final_crosswalk.loc[matching_rows, "Postal District"] = candidate[
            "Candidate Postal District"
        ]

    final_rows = len(final_crosswalk)
    duplicate_keys = int(final_crosswalk.duplicated(KEY_COLUMNS).sum())
    pca_keys = set(zip(pca_district_total["State"], pca_district_total["District"]))
    final_keys = set(zip(final_crosswalk["State"], final_crosswalk["District"]))
    pca_coverage = len(pca_keys & final_keys)
    unresolved_pca_records = len(pca_keys - approved_keys)
    row_multiplication = final_rows != original_rows
    codes_unchanged = (
        final_crosswalk[KEY_COLUMNS]
        .reset_index(drop=True)
        .equals(original_keys.reset_index(drop=True))
    )

    if final_rows != original_rows:
        raise ValueError(
            f"Unexpected final crosswalk row count: {original_rows} -> {final_rows}"
        )
    if duplicate_keys != 0:
        raise ValueError(f"Final crosswalk contains duplicate keys: {duplicate_keys}")
    if pca_coverage != EXPECTED_PCA_ROWS:
        raise ValueError(
            f"Final crosswalk does not retain all PCA keys: {pca_coverage}/"
            f"{EXPECTED_PCA_ROWS}"
        )
    if not codes_unchanged:
        raise ValueError("State or District codes changed during implementation.")

    final_crosswalk.to_csv(CROSSWALK_PATH, index=False)

    print("=== Safe Format Mapping Implementation Report ===")
    print(f"Original crosswalk rows: {original_rows}")
    print(f"Final crosswalk rows: {final_rows}")
    print(f"Approved mappings added: {len(approved)}")
    print(f"Duplicate State + District keys: {duplicate_keys}")
    print(f"PCA district coverage: {pca_coverage}/{EXPECTED_PCA_ROWS}")
    print(f"Unresolved PCA records without approved format mapping: {unresolved_pca_records}")
    print(f"Unexpected row multiplication: {'yes' if row_multiplication else 'no'}")
    print(f"State and District codes unchanged: {'yes' if codes_unchanged else 'no'}")
    print(f"Output crosswalk: {CROSSWALK_PATH}")
    print("Backup: none; no existing project backup convention was identified.")
    print("Only the 8 validated SAFE_FORMAT_MAPPING relationships were incorporated.")


if __name__ == "__main__":
    apply_safe_format_mappings()
