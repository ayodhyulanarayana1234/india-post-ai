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
EXPECTED_FORMAT_ONLY = 8


def validate_format_only_candidates():
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
    crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)

    required_dataset3_columns = set(DATASET3_COLUMNS.values())
    missing_dataset3_columns = required_dataset3_columns - set(dataset3.columns)
    if missing_dataset3_columns:
        raise ValueError(
            "Dataset3 is missing required columns: "
            f"{sorted(missing_dataset3_columns)}"
        )

    required_crosswalk_columns = {"State", "District", "State Name", "District Name"}
    missing_crosswalk_columns = required_crosswalk_columns - set(crosswalk.columns)
    if missing_crosswalk_columns:
        raise ValueError(
            "geographic_crosswalk.csv is missing required columns: "
            f"{sorted(missing_crosswalk_columns)}"
        )

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

    format_only_candidates = []
    for candidate in one_to_one_candidates:
        pca_row = pca_rows_by_name[
            (candidate["PCA State"], candidate["PCA District"])
        ]
        pca_name = candidate["PCA District Name"]
        postal_name = candidate["Candidate Postal District"]
        code_key = (pca_row["State"], pca_row["District"])
        census_names = []

        columns = DATASET3_COLUMNS
        for _, dataset3_row in dataset3.loc[
            dataset3[columns["state_code"]].eq(code_key[0])
            & dataset3[columns["district_code"]].eq(code_key[1])
            & dataset3[columns["subdistrict_code"]].eq(0)
            & dataset3[columns["town_village_code"]].eq(0)
        ].iterrows():
            census_names.append(dataset3_row[columns["location_name"]])

        if len(census_names) != 1:
            continue
        census_name = census_names[0]
        normalized_postal = deterministic_normalize(postal_name)
        normalized_census = deterministic_normalize(census_name)
        if normalized_postal == normalized_census:
            format_only_candidates.append(
                {
                    "candidate": candidate,
                    "pca_row": pca_row,
                    "code_key": code_key,
                    "census_name": census_name,
                    "normalized_postal": normalized_postal,
                    "normalized_census": normalized_census,
                }
            )

    if len(format_only_candidates) != EXPECTED_FORMAT_ONLY:
        raise ValueError(
            f"Expected {EXPECTED_FORMAT_ONLY} FORMAT_ONLY_DIFFERENCE candidates, "
            f"found {len(format_only_candidates)}."
        )

    crosswalk_keys = set(zip(crosswalk["State"], crosswalk["District"]))
    review_rows = []
    for item in format_only_candidates:
        candidate = item["candidate"]
        pca_row = item["pca_row"]
        code_key = item["code_key"]
        census_name = item["census_name"]
        census_state_code, census_district_code = code_key
        census_state_names = dataset3.loc[
            dataset3[columns["state_code"]].eq(census_state_code)
            & dataset3[columns["district_code"]].eq(0)
            & dataset3[columns["subdistrict_code"]].eq(0)
            & dataset3[columns["town_village_code"]].eq(0),
            columns["location_name"],
        ].dropna().drop_duplicates()
        census_identity_rows = dataset3.loc[
            dataset3[columns["state_code"]].eq(census_state_code)
            & dataset3[columns["district_code"]].eq(census_district_code)
            & dataset3[columns["subdistrict_code"]].eq(0)
            & dataset3[columns["town_village_code"]].eq(0)
        ]

        pca_code_unique = (
            pca_district_total[["State", "District"]]
            .eq([pca_row["State"], pca_row["District"]])
            .all(axis=1)
            .sum()
            == 1
        )
        code_matches_census = (
            pca_row["State"] == census_state_code
            and pca_row["District"] == census_district_code
        )
        hierarchy_unique = (
            len(census_identity_rows) == 1
            and len(census_identity_rows[columns["location_name"]].drop_duplicates()) == 1
            and code_key in crosswalk_keys
        )
        pca_candidate_count = len(candidate_to_postal[candidate["pca_key"]])
        postal_candidate_count = len(postal_to_pca[candidate["postal_key"]])
        relationship_one_to_one = (
            pca_candidate_count == 1 and postal_candidate_count == 1
        )
        no_competing_postal = pca_candidate_count == 1
        state_same = candidate["PCA State"] == normalize_for_comparison(
            candidate["Candidate Postal State"]
        )
        normalized_pca = deterministic_normalize(candidate["PCA District Name"])
        normalized_census = item["normalized_census"]
        normalized_postal = item["normalized_postal"]
        postal_equals_census = normalized_postal == normalized_census
        pca_equals_census = normalized_pca == normalized_census

        conditions = {
            "A PCA State code == Census State code": code_matches_census,
            "B PCA District code == Census District code": code_matches_census,
            "C exactly one Dataset3 hierarchy identity": hierarchy_unique,
            "D PCA State + District code unique": pca_code_unique,
            "E candidate relationship one-to-one": relationship_one_to_one,
            "F normalized Postal == normalized Census": postal_equals_census,
            "G normalized PCA == normalized Census": pca_equals_census,
            "H no competing postal candidate": no_competing_postal,
        }
        failed_conditions = [name for name, passed in conditions.items() if not passed]
        category = "SAFE_FORMAT_MAPPING" if not failed_conditions else "NOT_SUFFICIENT"

        review_rows.append(
            {
                "PCA State Name": candidate["PCA State Name"],
                "PCA District Name": candidate["PCA District Name"],
                "PCA State code": pca_row["State"],
                "PCA District code": pca_row["District"],
                "Postal State": candidate["Candidate Postal State"],
                "Postal District": candidate["Candidate Postal District"],
                "Census State name": census_state_names.iloc[0] if len(census_state_names) == 1 else None,
                "Census District name": census_name,
                "Census State code": census_state_code,
                "Census District code": census_district_code,
                "Dataset3 hierarchy record count": len(census_identity_rows),
                "Exact name difference": (
                    f"PCA={candidate['PCA District Name']!r}; "
                    f"Postal={candidate['Candidate Postal District']!r}; "
                    f"Census={census_name!r}"
                ),
                "State identical after standardization": state_same,
                "PCA code uniquely identifies Census district": hierarchy_unique,
                "Postal name exactly equals Census name": candidate["Candidate Postal District"] == census_name,
                "Normalized postal district name": normalized_postal,
                "Normalized Census district name": normalized_census,
                "A": conditions["A PCA State code == Census State code"],
                "B": conditions["B PCA District code == Census District code"],
                "C": conditions["C exactly one Dataset3 hierarchy identity"],
                "D": conditions["D PCA State + District code unique"],
                "E": conditions["E candidate relationship one-to-one"],
                "F": conditions["F normalized Postal == normalized Census"],
                "G": conditions["G normalized PCA == normalized Census"],
                "H": conditions["H no competing postal candidate"],
                "Validation Category": category,
                "Failed Conditions": "; ".join(failed_conditions) or "None",
            }
        )

    review = pd.DataFrame(review_rows)
    safe_count = int((review["Validation Category"] == "SAFE_FORMAT_MAPPING").sum())
    not_sufficient_count = int(
        (review["Validation Category"] == "NOT_SUFFICIENT").sum()
    )

    print("=== FORMAT_ONLY Candidate Evidence Validation ===")
    print("No fuzzy matching, edit distance, similarity scoring, or inference was used.")
    print(f"Total FORMAT_ONLY candidates: {len(format_only_candidates)}")
    print(f"SAFE_FORMAT_MAPPING: {safe_count}")
    print(f"NOT_SUFFICIENT: {not_sufficient_count}")
    print("\nComplete validation table:")
    print(review.to_string(index=False))
    print("\nFailed validation conditions:")
    failed = review.loc[review["Failed Conditions"] != "None"]
    print(failed[["PCA State Name", "PCA District Name", "Failed Conditions"]].to_string(index=False) if not failed.empty else "None")
    print("\nNo geographic mappings were created.")


if __name__ == "__main__":
    validate_format_only_candidates()
