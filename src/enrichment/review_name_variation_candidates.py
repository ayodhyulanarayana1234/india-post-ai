from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
EXPECTED_UNMATCHED_PAIRS = 134
EXPECTED_SIMPLE_CANDIDATES = 43


def identify_column(dataframe, role, candidates):
    normalized_columns = {
        str(column).strip().casefold(): column for column in dataframe.columns
    }
    matches = [normalized_columns[name] for name in candidates if name in normalized_columns]

    if len(matches) != 1:
        raise ValueError(
            f"Could not identify exactly one {role} column. "
            f"Candidates found: {matches}"
        )

    return matches[0]


def normalize_for_comparison(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def remove_punctuation_and_whitespace(value):
    return re.sub(r"[^a-z0-9]", "", value)


def name_tokens(value):
    return value.split()


def simple_name_reason(pca_name, postal_name):
    if pca_name == postal_name:
        return None

    if remove_punctuation_and_whitespace(pca_name) == remove_punctuation_and_whitespace(
        postal_name
    ):
        return "same letters and numbers after removing punctuation and whitespace"

    pca_tokens = name_tokens(pca_name)
    postal_tokens = name_tokens(postal_name)
    if len(pca_tokens) != len(postal_tokens):
        return None

    differences = [
        (pca_token, postal_token)
        for pca_token, postal_token in zip(pca_tokens, postal_tokens)
        if pca_token != postal_token
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


def review_name_variation_candidates():
    if not POSTAL_PATH.exists():
        raise FileNotFoundError(f"Postal network file not found: {POSTAL_PATH}")
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"PCA enrichment file not found: {PCA_PATH}")

    postal = pd.read_csv(POSTAL_PATH, low_memory=False)
    pca = pd.read_csv(PCA_PATH, low_memory=False)

    postal_state_column = identify_column(
        postal,
        "postal State name",
        ["statename", "state_name", "state name"],
    )
    postal_district_column = identify_column(
        postal,
        "postal District name",
        ["district", "districtname", "district_name", "district name"],
    )
    pca_state_column = identify_column(
        pca,
        "PCA State name",
        ["state name", "statename", "state_name"],
    )
    pca_district_column = identify_column(
        pca,
        "PCA District name",
        ["district name", "districtname", "district_name"],
    )

    postal_pairs = build_normalized_pairs(
        postal,
        postal_state_column,
        postal_district_column,
    )
    pca_pairs = build_normalized_pairs(
        pca,
        pca_state_column,
        pca_district_column,
    )
    postal_key_set = set(zip(postal_pairs["_state"], postal_pairs["_district"]))

    unmatched_pca = pca_pairs.loc[
        ~pca_pairs.apply(
            lambda row: (row["_state"], row["_district"]) in postal_key_set,
            axis=1,
        )
    ].sort_values(["_state", "_district"])
    if len(unmatched_pca) != EXPECTED_UNMATCHED_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_UNMATCHED_PAIRS} unmatched PCA pairs, "
            f"found {len(unmatched_pca)}."
        )

    postal_names_by_state = {
        state: sorted(group["_district"].unique())
        for state, group in postal_pairs.groupby("_state")
    }
    candidate_rows = []
    candidate_key_set = set()

    for _, pca_row in unmatched_pca.iterrows():
        pca_key = (pca_row["_state"], pca_row["_district"])
        for postal_district in postal_names_by_state.get(pca_row["_state"], []):
            reason = simple_name_reason(pca_row["_district"], postal_district)
            if reason:
                candidate_key = (pca_key, (pca_row["_state"], postal_district))
                candidate_key_set.add(candidate_key)
                candidate_rows.append(
                    {
                        "PCA State": pca_row[pca_state_column],
                        "PCA District": pca_row[pca_district_column],
                        "Candidate Postal State": pca_row[pca_state_column],
                        "Candidate Postal District": postal_pairs.loc[
                            (postal_pairs["_state"] == pca_row["_state"])
                            & (postal_pairs["_district"] == postal_district),
                            postal_district_column,
                        ].iloc[0],
                        "Exact PCA District Name": pca_row[pca_district_column],
                        "Exact Postal District Name": postal_pairs.loc[
                            (postal_pairs["_state"] == pca_row["_state"])
                            & (postal_pairs["_district"] == postal_district),
                            postal_district_column,
                        ].iloc[0],
                        "Reason": reason,
                        "_pca_key": pca_key,
                        "_postal_key": (pca_row["_state"], postal_district),
                    }
                )

    candidate_pca_keys = {key[0] for key in candidate_key_set}
    if len(candidate_pca_keys) != EXPECTED_SIMPLE_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_SIMPLE_CANDIDATES} SIMPLE_NAME_VARIATION "
            f"candidates, found {len(candidate_pca_keys)}."
        )

    candidate_postal_keys = {key[1] for key in candidate_key_set}
    pca_to_postal = {}
    postal_to_pca = {}
    for pca_key, postal_key in candidate_key_set:
        pca_to_postal.setdefault(pca_key, set()).add(postal_key)
        postal_to_pca.setdefault(postal_key, set()).add(pca_key)

    reviewed_rows = []
    for row in candidate_rows:
        pca_key = row["_pca_key"]
        postal_key = row["_postal_key"]
        pca_count = len(pca_to_postal[pca_key])
        postal_count = len(postal_to_pca[postal_key])
        one_to_one = pca_count == 1 and postal_count == 1

        if pca_count > 1 and postal_count > 1:
            relationship = "AMBIGUOUS: one-to-many and many-to-one"
        elif pca_count > 1:
            relationship = "ONE_TO_MANY"
        elif postal_count > 1:
            relationship = "MANY_TO_ONE"
        elif one_to_one:
            relationship = "ONE_TO_ONE_CANDIDATE_ONLY"
        else:
            relationship = "AMBIGUOUS"

        reviewed_rows.append(
            {
                "PCA State": row["PCA State"],
                "PCA District": row["PCA District"],
                "Candidate Postal State": row["Candidate Postal State"],
                "Candidate Postal District": row["Candidate Postal District"],
                "Exact PCA District Name": row["Exact PCA District Name"],
                "Exact Postal District Name": row["Exact Postal District Name"],
                "Reason": row["Reason"],
                "State Same After Standardization": pca_key[0] == postal_key[0],
                "PCA Candidate Count": pca_count,
                "Postal Candidate Count": postal_count,
                "One-to-One Relationship": one_to_one,
                "Relationship Review": relationship,
            }
        )

    review = pd.DataFrame(reviewed_rows)
    one_to_one_candidates = {
        key for key in candidate_pca_keys if len(pca_to_postal[key]) == 1
        and len(postal_to_pca[next(iter(pca_to_postal[key]))]) == 1
    }
    one_to_many_candidates = {
        key for key in candidate_pca_keys if len(pca_to_postal[key]) > 1
    }
    many_to_one_candidates = {
        key for key in candidate_pca_keys
        if any(len(postal_to_pca[postal_key]) > 1 for postal_key in pca_to_postal[key])
    }
    ambiguous_candidates = candidate_pca_keys - one_to_one_candidates

    print("=== Name Variation Candidate Review ===")
    print("Comparison normalization: string conversion, outer strip, casefold")
    print("Fuzzy matching: not used")
    print("Mappings assigned or saved: none")
    print(f"Unmatched PCA pairs reproduced: {len(unmatched_pca)}")
    print(f"SIMPLE_NAME_VARIATION candidates: {len(candidate_pca_keys)}")

    print("\nCandidate details:")
    print(review.to_string(index=False))

    print("\nRelationship flags:")
    print("  ONE_TO_MANY: one PCA district has multiple postal candidates.")
    print("  MANY_TO_ONE: multiple PCA districts share one postal candidate.")
    print("  AMBIGUOUS: relationship cannot be determined safely.")
    print("  ONE_TO_ONE_CANDIDATE_ONLY is not an automatic approval.")

    print("\nFinal summary:")
    print(f"  Total SIMPLE_NAME_VARIATION candidates: {len(candidate_pca_keys)}")
    print(f"  One-to-one candidates: {len(one_to_one_candidates)}")
    print(f"  One-to-many candidates: {len(one_to_many_candidates)}")
    print(f"  Many-to-one candidates: {len(many_to_one_candidates)}")
    print(f"  Ambiguous candidates: {len(ambiguous_candidates)}")
    print("\nNo datasets were merged, modified, dropped, mapped, or saved.")


if __name__ == "__main__":
    review_name_variation_candidates()
