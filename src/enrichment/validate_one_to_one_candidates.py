from pathlib import Path

import pandas as pd

from validate_geographic_candidates import (
    DATASET3_COLUMNS,
    build_candidate_records,
    identify_column,
    normalize_for_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET3_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset3.xlsx"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"
EXPECTED_ONE_TO_ONE = 32


def validate_one_to_one_candidates():
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
    one_to_one_pca_keys = {candidate["pca_key"] for candidate in one_to_one_candidates}
    if len(one_to_one_pca_keys) != EXPECTED_ONE_TO_ONE:
        raise ValueError(
            f"Expected {EXPECTED_ONE_TO_ONE} one-to-one candidates, "
            f"found {len(one_to_one_pca_keys)}."
        )

    columns = DATASET3_COLUMNS
    state_hierarchy = dataset3.loc[
        dataset3[columns["district_code"]].eq(0)
        & dataset3[columns["subdistrict_code"]].eq(0)
        & dataset3[columns["town_village_code"]].eq(0)
    ]
    district_hierarchy = dataset3.loc[
        dataset3[columns["district_code"]].ne(0)
        & dataset3[columns["subdistrict_code"]].eq(0)
        & dataset3[columns["town_village_code"]].eq(0)
    ]

    state_names = {
        row[columns["state_code"]]: row[columns["location_name"]]
        for _, row in state_hierarchy.iterrows()
    }
    district_evidence = {}
    for _, row in district_hierarchy.iterrows():
        key = (row[columns["state_code"]], row[columns["district_code"]])
        district_evidence.setdefault(key, []).append(row[columns["location_name"]])

    pca_district_total = pca.loc[
        pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total")
    ]
    pca_codes_by_name = {
        (
            normalize_for_comparison(row["State Name"]),
            normalize_for_comparison(row["District Name"]),
        ): row
        for _, row in pca_district_total.iterrows()
    }
    crosswalk_keys = set(zip(crosswalk["State"], crosswalk["District"]))

    review_rows = []
    for candidate in one_to_one_candidates:
        pca_name_key = (candidate["PCA State"], candidate["PCA District"])
        pca_row = pca_codes_by_name.get(pca_name_key)
        if pca_row is None:
            review_rows.append(
                build_review_row(
                    candidate,
                    None,
                    None,
                    None,
                    [],
                    False,
                    False,
                    "NOT_SUFFICIENT",
                    "Existing PCA enrichment does not provide codes for this candidate.",
                )
            )
            continue

        census_key = (pca_row["State"], pca_row["District"])
        census_names = district_evidence.get(census_key, [])
        census_state_name = state_names.get(pca_row["State"])
        census_district_name = (
            census_names[0]
            if census_names and len(set(census_names)) == 1
            else None
        )
        code_identity_supported = (
            census_key in crosswalk_keys
            and len(census_names) == 1
            and census_district_name is not None
        )
        state_same = candidate["PCA State"] == normalize_for_comparison(
            candidate["Candidate Postal State"]
        )
        postal_name_exact = (
            code_identity_supported
            and candidate["Candidate Postal District"] == census_district_name
        )

        if code_identity_supported and postal_name_exact:
            category = "CODE_AND_NAME_SUPPORTED"
            reason = "Census code identity and exact postal/Census district name agree."
        elif code_identity_supported:
            category = "CODE_SUPPORTED_NAME_DIFFERENCE"
            reason = (
                "Census code identity is uniquely supported, but the postal district "
                "name differs from the Census district name; equivalence is not proven."
            )
        else:
            category = "NOT_SUFFICIENT"
            reason = (
                "The available code and hierarchy evidence does not uniquely establish "
                "the geographic identity."
            )

        review_rows.append(
            build_review_row(
                candidate,
                pca_row,
                census_key,
                census_state_name,
                census_names,
                state_same,
                postal_name_exact,
                category,
                reason,
                census_district_name=census_district_name,
                code_identity_supported=code_identity_supported,
            )
        )

    review = pd.DataFrame(review_rows)
    totals = review["Validation Category"].value_counts()

    print("=== One-to-One Geographic Candidate Validation ===")
    print("Comparison normalization: string conversion, outer strip, casefold")
    print("Fuzzy matching: not used")
    print("Historical equivalence inferred from names: no")
    print(f"Total one-to-one candidates: {len(one_to_one_pca_keys)}")
    print(f"CODE_AND_NAME_SUPPORTED: {int(totals.get('CODE_AND_NAME_SUPPORTED', 0))}")
    print(
        "CODE_SUPPORTED_NAME_DIFFERENCE: "
        f"{int(totals.get('CODE_SUPPORTED_NAME_DIFFERENCE', 0))}"
    )
    print(f"NOT_SUFFICIENT: {int(totals.get('NOT_SUFFICIENT', 0))}")
    print("\nComplete validation table:")
    print(review.to_string(index=False))
    print("\nNo geographic mappings were created.")
    print("No datasets were merged, modified, dropped, or overwritten.")


def build_review_row(
    candidate,
    pca_row,
    census_key,
    census_state_name,
    census_names,
    state_same,
    postal_name_exact,
    category,
    reason,
    census_district_name=None,
    code_identity_supported=False,
):
    return {
        "PCA State Name": candidate["PCA State Name"],
        "PCA District Name": candidate["PCA District Name"],
        "PCA State code": pca_row["State"] if pca_row is not None else None,
        "PCA District code": pca_row["District"] if pca_row is not None else None,
        "Postal State": candidate["Candidate Postal State"],
        "Postal District": candidate["Candidate Postal District"],
        "Census State code": census_key[0] if census_key else None,
        "Census District code": census_key[1] if census_key else None,
        "Census State name": census_state_name,
        "Census District name": census_district_name,
        "Dataset3 hierarchy record count": len(census_names),
        "Exact name difference": (
            f"PCA={candidate['PCA District Name']!r}; "
            f"Postal={candidate['Candidate Postal District']!r}; "
            f"Census={census_district_name!r}"
        ),
        "State identical after standardization": state_same,
        "PCA code uniquely identifies Census district": code_identity_supported,
        "Postal equals Census district exactly": postal_name_exact,
        "Relationship proven by available data": category == "CODE_AND_NAME_SUPPORTED",
        "Validation Category": category,
        "Evidence": reason,
    }


if __name__ == "__main__":
    validate_one_to_one_candidates()
