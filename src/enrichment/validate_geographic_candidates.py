from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET3_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset3.xlsx"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"
EXPECTED_CANDIDATES = 43
EXPECTED_UNMATCHED = 134

DATASET3_COLUMNS = {
    "state_code": "State Code",
    "district_code": "District Code",
    "subdistrict_code": "Sub District Code",
    "town_village_code": "Town-Village Code",
    "location_name": "Town-Village Name",
}


def identify_column(dataframe, role, candidates):
    normalized_columns = {
        str(column).strip().casefold(): column for column in dataframe.columns
    }
    matches = [normalized_columns[name] for name in candidates if name in normalized_columns]
    if len(matches) != 1:
        raise ValueError(
            f"Could not identify exactly one {role} column. Candidates: {matches}"
        )
    return matches[0]


def normalize_for_comparison(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def remove_punctuation_and_whitespace(value):
    return re.sub(r"[^a-z0-9]", "", value)


def simple_name_reason(pca_name, postal_name):
    if pca_name == postal_name:
        return None
    if remove_punctuation_and_whitespace(pca_name) == remove_punctuation_and_whitespace(
        postal_name
    ):
        return "same letters and numbers after removing punctuation and whitespace"

    pca_tokens = pca_name.split()
    postal_tokens = postal_name.split()
    if len(pca_tokens) != len(postal_tokens):
        return None

    differences = [
        (first, second)
        for first, second in zip(pca_tokens, postal_tokens)
        if first != second
    ]
    if len(differences) != 1:
        return None

    first, second = differences[0]
    if len(first) <= 3 and second.startswith(first):
        return "possible abbreviation expansion"
    if len(second) <= 3 and first.startswith(second):
        return "possible abbreviation"
    if first[0] == second[0] and first[-1] == second[-1]:
        return "possible spelling difference"
    return None


def build_normalized_pairs(dataframe, state_column, district_column):
    pairs = dataframe[[state_column, district_column]].drop_duplicates().copy()
    pairs["_state"] = pairs[state_column].map(normalize_for_comparison)
    pairs["_district"] = pairs[district_column].map(normalize_for_comparison)
    return pairs.loc[pairs["_state"].ne("") & pairs["_district"].ne("")].copy()


def build_candidate_records(postal, pca):
    postal_state_column = identify_column(
        postal, "postal State name", ["statename", "state_name", "state name"]
    )
    postal_district_column = identify_column(
        postal,
        "postal District name",
        ["district", "districtname", "district_name", "district name"],
    )
    pca_state_column = identify_column(
        pca, "PCA State name", ["state name", "statename", "state_name"]
    )
    pca_district_column = identify_column(
        pca,
        "PCA District name",
        ["district name", "districtname", "district_name"],
    )

    postal_pairs = build_normalized_pairs(
        postal, postal_state_column, postal_district_column
    )
    pca_pairs = build_normalized_pairs(pca, pca_state_column, pca_district_column)
    postal_keys = set(zip(postal_pairs["_state"], postal_pairs["_district"]))
    unmatched_pca = pca_pairs.loc[
        ~pca_pairs.apply(
            lambda row: (row["_state"], row["_district"]) in postal_keys,
            axis=1,
        )
    ].sort_values(["_state", "_district"])

    if len(unmatched_pca) != EXPECTED_UNMATCHED:
        raise ValueError(f"Expected {EXPECTED_UNMATCHED} unmatched PCA pairs, found {len(unmatched_pca)}.")

    postal_names_by_state = {
        state: sorted(group["_district"].unique())
        for state, group in postal_pairs.groupby("_state")
    }
    postal_exact_names = {
        (row["_state"], row["_district"]): (
            row[postal_state_column],
            row[postal_district_column],
        )
        for _, row in postal_pairs.iterrows()
    }
    pca_exact_names = {
        (row["_state"], row["_district"]): (
            row[pca_state_column],
            row[pca_district_column],
        )
        for _, row in unmatched_pca.iterrows()
    }

    candidates = []
    for _, pca_row in unmatched_pca.iterrows():
        pca_key = (pca_row["_state"], pca_row["_district"])
        for postal_district in postal_names_by_state.get(pca_row["_state"], []):
            reason = simple_name_reason(pca_row["_district"], postal_district)
            if reason:
                postal_key = (pca_row["_state"], postal_district)
                candidates.append(
                    {
                        "pca_key": pca_key,
                        "postal_key": postal_key,
                        "PCA State Name": pca_exact_names[pca_key][0],
                        "PCA District Name": pca_exact_names[pca_key][1],
                        "PCA State": pca_row["_state"],
                        "PCA District": pca_row["_district"],
                        "Candidate Postal State": postal_exact_names[postal_key][0],
                        "Candidate Postal District": postal_exact_names[postal_key][1],
                        "Reason": reason,
                    }
                )

    candidate_pca_keys = {candidate["pca_key"] for candidate in candidates}
    if len(candidate_pca_keys) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATES} SIMPLE_NAME_VARIATION candidates, "
            f"found {len(candidate_pca_keys)}."
        )
    return candidates, candidate_pca_keys


def validate_candidates():
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

    required_crosswalk_columns = {"State", "District", "State Name", "District Name"}
    missing_crosswalk_columns = required_crosswalk_columns - set(crosswalk.columns)
    if missing_crosswalk_columns:
        raise ValueError(
            "geographic_crosswalk.csv is missing required columns: "
            f"{sorted(missing_crosswalk_columns)}"
        )

    required_dataset3_columns = set(DATASET3_COLUMNS.values())
    missing_dataset3_columns = required_dataset3_columns - set(dataset3.columns)
    if missing_dataset3_columns:
        raise ValueError(
            f"Dataset3 is missing required columns: {sorted(missing_dataset3_columns)}"
        )

    candidates, candidate_pca_keys = build_candidate_records(postal, pca)
    candidate_to_postal = {}
    postal_to_candidate = {}
    for candidate in candidates:
        candidate_to_postal.setdefault(candidate["pca_key"], set()).add(
            candidate["postal_key"]
        )
        postal_to_candidate.setdefault(candidate["postal_key"], set()).add(
            candidate["pca_key"]
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

    pca_code_by_name = {
        (
            normalize_for_comparison(row["State Name"]),
            normalize_for_comparison(row["District Name"]),
        ): row
        for _, row in pca.loc[
            pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total")
        ].iterrows()
    }
    crosswalk_code_keys = set(zip(crosswalk["State"], crosswalk["District"]))

    review_rows = []
    for candidate in candidates:
        pca_key = candidate["pca_key"]
        pca_name_key = (
            candidate["PCA State"],
            candidate["PCA District"],
        )
        pca_code_row = pca_code_by_name.get(pca_name_key)
        if pca_code_row is None:
            pca_code_row = None

        census_key = (
            pca_code_row["State"],
            pca_code_row["District"],
        ) if pca_code_row is not None else None
        crosswalk_key_supported = census_key in crosswalk_code_keys

        census_names = district_evidence.get(census_key, []) if crosswalk_key_supported else []
        census_state_name = (
            state_names.get(pca_code_row["State"])
            if crosswalk_key_supported
            else None
        )
        census_district_name = census_names[0] if len(set(census_names)) == 1 and census_names else None
        pca_count = len(candidate_to_postal[pca_key])
        postal_count = len(postal_to_candidate[candidate["postal_key"]])
        uniquely_identified = (
            crosswalk_key_supported
            and len(census_names) == 1
            and census_district_name is not None
        )
        state_same = candidate["PCA State"] == normalize_for_comparison(
            candidate["Candidate Postal State"]
        )

        if pca_count > 1 or postal_count > 1:
            category = "AMBIGUOUS"
        else:
            category = "CODE_NOT_SUFFICIENT"

        if category == "AMBIGUOUS":
            evidence_assessment = (
                "Multiple candidate relationships exist; safe equivalence "
                "cannot be determined from these datasets."
            )
        elif pca_code_row is None:
            evidence_assessment = (
                "The existing PCA enrichment/crosswalk relationship does not "
                "provide a code for this candidate."
            )
        elif not crosswalk_key_supported:
            evidence_assessment = (
                "The PCA code is not present in the existing geographic "
                "crosswalk."
            )
        elif not uniquely_identified:
            evidence_assessment = (
                "Dataset3 does not uniquely identify one district hierarchy "
                "record for the PCA codes."
            )
        else:
            evidence_assessment = (
                "Codes uniquely identify the PCA/Census district, but Dataset3 "
                "does not provide a postal code for the candidate; equivalence "
                "is not proven."
            )

        review_rows.append(
            {
                "PCA State Name": candidate["PCA State Name"],
                "PCA District Name": candidate["PCA District Name"],
                "PCA State code": pca_code_row["State"] if pca_code_row is not None else None,
                "PCA District code": pca_code_row["District"] if pca_code_row is not None else None,
                "Candidate Postal State": candidate["Candidate Postal State"],
                "Candidate Postal District": candidate["Candidate Postal District"],
                "Census State code": census_key[0] if census_key else None,
                "Census District code": census_key[1] if census_key else None,
                "Census State name": census_state_name,
                "Census District name": census_district_name,
                "Dataset3 district hierarchy records": len(census_names),
                "District uniquely identified by Census codes": uniquely_identified,
                "State same after standardization": state_same,
                "PCA candidate postal districts": pca_count,
                "Postal candidate PCA districts": postal_count,
                "Validation category": category,
                "Evidence assessment": evidence_assessment,
                "Name variation reason": candidate["Reason"],
            }
        )

    review = pd.DataFrame(review_rows)
    totals = review["Validation category"].value_counts()
    code_supported = int(totals.get("CODE_SUPPORTED", 0))
    code_not_sufficient = int(totals.get("CODE_NOT_SUFFICIENT", 0))
    ambiguous = int(totals.get("AMBIGUOUS", 0))

    print("=== Geographic Candidate Code/Hierarchy Validation ===")
    print("Dataset3 used as the authoritative Census hierarchy source.")
    print("No fuzzy matching or name-only historical inference was used.")
    print(f"Unmatched PCA pairs reproduced: {EXPECTED_UNMATCHED}")
    print(f"Total SIMPLE_NAME_VARIATION candidates: {len(candidate_pca_keys)}")

    print("\nOne-to-many candidate details:")
    one_to_many = review.loc[review["PCA candidate postal districts"] > 1]
    if one_to_many.empty:
        print("None")
    else:
        print(one_to_many.to_string(index=False))
        print(
            "Every listed postal candidate above is shown with the same PCA/Census "
            "code evidence; no candidate was selected."
        )

    print("\nOne-to-one candidate evidence:")
    one_to_one = review.loc[
        (review["PCA candidate postal districts"] == 1)
        & (review["Postal candidate PCA districts"] == 1)
    ]
    print(one_to_one.to_string(index=False))
    print(
        "One-to-one relationships remain CODE_NOT_SUFFICIENT unless the supplied "
        "data proves postal equivalence; Dataset3 codes identify PCA/Census rows "
        "but do not provide postal codes."
    )

    print("\nValidation totals:")
    print(f"  Total candidates: {len(candidate_pca_keys)}")
    print(f"  CODE_SUPPORTED: {code_supported}")
    print(f"  CODE_NOT_SUFFICIENT: {code_not_sufficient}")
    print(f"  AMBIGUOUS: {ambiguous}")

    print("\nComplete candidate validation table:")
    print(review.to_string(index=False))
    print("\nNo geographic mapping was created.")
    print("No datasets were merged, modified, dropped, or overwritten.")


if __name__ == "__main__":
    validate_candidates()
