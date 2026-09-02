from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
EXPECTED_UNMATCHED_PAIRS = 134
CATEGORIES = [
    "SIMPLE_NAME_VARIATION",
    "HISTORICAL_OR_BOUNDARY_DIFFERENCE",
    "POSSIBLE_RENAMING",
    "POSSIBLE_SPLIT_OR_REORGANIZATION",
    "UNCLEAR",
]


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


def classify_geographic_mismatches():
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
    report_rows = []
    simple_candidates = []

    for _, row in unmatched_pca.iterrows():
        state = row["_state"]
        pca_district = row["_district"]
        state_candidates = []

        for postal_district in postal_names_by_state.get(state, []):
            reason = simple_name_reason(pca_district, postal_district)
            if reason:
                state_candidates.append((postal_district, reason))

        if state_candidates:
            category = "SIMPLE_NAME_VARIATION"
            evidence = "; ".join(
                f"{postal_district!r}: {reason}"
                for postal_district, reason in state_candidates
            )
            for postal_district, reason in state_candidates:
                simple_candidates.append(
                    {
                        "PCA State": state,
                        "PCA District": pca_district,
                        "Candidate Postal State": state,
                        "Candidate Postal District": postal_district,
                        "Reason": reason,
                    }
                )
        else:
            category = "UNCLEAR"
            evidence = (
                "No deterministic same-state name variation supported by "
                "the inspected values; official geographic evidence is required."
            )

        report_rows.append(
            {
                "PCA State": state,
                "PCA District": pca_district,
                "Diagnostic Category": category,
                "Evidence": evidence,
            }
        )

    report = pd.DataFrame(report_rows)
    category_totals = report["Diagnostic Category"].value_counts().reindex(
        CATEGORIES,
        fill_value=0,
    )

    print("=== Geographic Mismatch Classification Report ===")
    print("\nComparison normalization: string conversion, outer strip, casefold")
    print("Fuzzy matching: not used")
    print("Mappings assigned: none")

    print("\n1. All unmatched PCA State + District pairs:")
    print(
        report[["PCA State", "PCA District"]]
        .to_string(index=False)
    )

    print("\n2. SIMPLE_NAME_VARIATION candidates only:")
    if simple_candidates:
        print(pd.DataFrame(simple_candidates).to_string(index=False))
    else:
        print("No supported simple name-variation candidates found.")

    print("\n3. Historical, renaming, split, and reorganization assessment:")
    print(
        "No unmatched pairs were assigned to HISTORICAL_OR_BOUNDARY_DIFFERENCE, "
        "POSSIBLE_RENAMING, or POSSIBLE_SPLIT_OR_REORGANIZATION based on names "
        "alone. A reliable classification or mapping for such cases requires an "
        "official geographic crosswalk or other authoritative source."
    )

    print("\n4. Category totals:")
    for category, count in category_totals.items():
        print(f"  {category}: {count}")

    print("\n5. Final classification table for all unmatched pairs:")
    print(report.to_string(index=False))
    print("\nNo datasets were merged, modified, dropped, or mapped.")


if __name__ == "__main__":
    classify_geographic_mismatches()
