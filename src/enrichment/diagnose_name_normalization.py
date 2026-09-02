from pathlib import Path
import unicodedata

import pandas as pd

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
EXPECTED_CANDIDATES = 32


def deterministic_normalize(value):
    """Normalize representational formatting without changing letters."""
    if pd.isna(value):
        return ""

    text = " ".join(str(value).strip().casefold().split())
    return "".join(
        character
        for character in text
        if not character.isspace()
        and character != "-"
        and not unicodedata.category(character).startswith("P")
    )


def validate_name_normalization():
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
    one_to_one_keys = {candidate["pca_key"] for candidate in one_to_one_candidates}
    if len(one_to_one_keys) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATES} one-to-one candidates, "
            f"found {len(one_to_one_keys)}."
        )

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

    crosswalk_keys = set(zip(crosswalk["State"], crosswalk["District"]))
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

    report_rows = []
    for candidate in one_to_one_candidates:
        pca_name_key = (candidate["PCA State"], candidate["PCA District"])
        pca_row = pca_rows_by_name.get(pca_name_key)
        if pca_row is None:
            raise ValueError(
                "A one-to-one candidate has no corresponding PCA enrichment row "
                f"for names {pca_name_key!r}."
            )

        code_key = (pca_row["State"], pca_row["District"])
        census_names = census_names_by_code.get(code_key, [])
        census_name = (
            census_names[0]
            if len(census_names) == 1
            else None
        )
        code_supported = code_key in crosswalk_keys and census_name is not None
        normalized_pca = deterministic_normalize(candidate["PCA District Name"])
        normalized_postal = deterministic_normalize(
            candidate["Candidate Postal District"]
        )
        normalized_census = deterministic_normalize(census_name)
        postal_equals_census = (
            code_supported and normalized_postal == normalized_census
        )
        pca_equals_census = code_supported and normalized_pca == normalized_census

        category = (
            "FORMAT_ONLY_DIFFERENCE"
            if postal_equals_census
            else "NAME_REMAINS_DIFFERENT"
        )
        report_rows.append(
            {
                "PCA State": candidate["PCA State Name"],
                "PCA District": candidate["PCA District Name"],
                "Postal State": candidate["Candidate Postal State"],
                "Postal District": candidate["Candidate Postal District"],
                "Census District": census_name,
                "Original PCA District Name": candidate["PCA District Name"],
                "Original Postal District Name": candidate["Candidate Postal District"],
                "Original Census District Name": census_name,
                "Normalized PCA Name": normalized_pca,
                "Normalized Postal Name": normalized_postal,
                "Normalized Census Name": normalized_census,
                "Normalized Postal == Census": postal_equals_census,
                "Normalized PCA == Census": pca_equals_census,
                "Diagnostic Classification": category,
            }
        )

    report = pd.DataFrame(report_rows)
    totals = report["Diagnostic Classification"].value_counts()

    print("=== Name Normalization Diagnostic ===")
    print("Only the 32 one-to-one CODE_SUPPORTED_NAME_DIFFERENCE candidates were reviewed.")
    print("Normalization: string conversion, strip, casefold, whitespace collapse, "
          "space/hyphen/punctuation removal.")
    print("Fuzzy matching, edit distance, similarity scoring, and historical inference: not used.")
    print(f"Total candidates: {len(one_to_one_keys)}")
    print(f"FORMAT_ONLY_DIFFERENCE: {int(totals.get('FORMAT_ONLY_DIFFERENCE', 0))}")
    print(f"NAME_REMAINS_DIFFERENT: {int(totals.get('NAME_REMAINS_DIFFERENT', 0))}")
    print("\nComplete 32-row diagnostic table:")
    print(report.to_string(index=False))
    print("\nNo geographic mappings were created.")


if __name__ == "__main__":
    validate_name_normalization()
