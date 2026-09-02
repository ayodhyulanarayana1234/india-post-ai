from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTAL_PATH = PROJECT_ROOT / "data" / "processed" / "postal_network.csv"
PCA_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"


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


def normalize_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def without_punctuation(name):
    return re.sub(r"[^a-z0-9\s]", "", name).split()


def name_tokens(name):
    return name.split()


def abbreviation_candidate(pca_name, postal_name):
    pca_tokens = name_tokens(pca_name)
    postal_tokens = name_tokens(postal_name)
    if len(pca_tokens) != len(postal_tokens):
        return False

    different_tokens = [
        (pca_token, postal_token)
        for pca_token, postal_token in zip(pca_tokens, postal_tokens)
        if pca_token != postal_token
    ]
    if len(different_tokens) != 1:
        return False

    short_token, long_token = different_tokens[0]
    return (
        (len(short_token) <= 3 and long_token.startswith(short_token))
        or (len(long_token) <= 3 and short_token.startswith(long_token))
    )


def spelling_candidate(pca_name, postal_name):
    pca_tokens = name_tokens(pca_name)
    postal_tokens = name_tokens(postal_name)
    if len(pca_tokens) != len(postal_tokens):
        return False

    different_tokens = 0
    for pca_token, postal_token in zip(pca_tokens, postal_tokens):
        if pca_token == postal_token:
            continue
        if not pca_token or not postal_token:
            return False
        if pca_token[0] != postal_token[0] or pca_token[-1] != postal_token[-1]:
            return False
        different_tokens += 1

    return different_tokens > 0


def subset_candidate(pca_name, postal_name):
    pca_tokens = set(name_tokens(pca_name))
    postal_tokens = set(name_tokens(postal_name))
    if not pca_tokens or not postal_tokens or pca_tokens == postal_tokens:
        return False
    return pca_tokens.issubset(postal_tokens) or postal_tokens.issubset(pca_tokens)


def classify_candidate(pca_name, postal_name):
    reasons = []
    pca_without_punctuation = " ".join(without_punctuation(pca_name))
    postal_without_punctuation = " ".join(without_punctuation(postal_name))

    if (
        pca_name != postal_name
        and pca_without_punctuation
        and pca_without_punctuation == postal_without_punctuation
    ):
        reasons.append("punctuation difference")
    if abbreviation_candidate(pca_name, postal_name):
        reasons.append("possible abbreviation")
    if spelling_candidate(pca_name, postal_name):
        reasons.append("possible spelling difference")
    if subset_candidate(pca_name, postal_name):
        reasons.append("possible old/new, renamed, split, or reorganized district")

    return reasons


def analyze_geographic_mismatches():
    if not POSTAL_PATH.exists():
        raise FileNotFoundError(f"Postal network file not found: {POSTAL_PATH}")
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"PCA enrichment file not found: {PCA_PATH}")

    postal = pd.read_csv(POSTAL_PATH, low_memory=False)
    pca = pd.read_csv(PCA_PATH, low_memory=False)

    postal_state_col = identify_column(
        postal,
        "postal State name",
        ["statename", "state_name", "state name"],
    )
    postal_district_col = identify_column(
        postal,
        "postal District name",
        ["district", "districtname", "district_name", "district name"],
    )
    pca_state_col = identify_column(
        pca,
        "PCA State name",
        ["state name", "statename", "state_name"],
    )
    pca_district_col = identify_column(
        pca,
        "PCA District name",
        ["district name", "districtname", "district_name"],
    )

    postal_pairs = postal[[postal_state_col, postal_district_col]].drop_duplicates()
    postal_pairs = postal_pairs.assign(
        _state=postal_pairs[postal_state_col].map(normalize_name),
        _district=postal_pairs[postal_district_col].map(normalize_name),
    )
    postal_pairs = postal_pairs.loc[
        postal_pairs["_state"].ne("") & postal_pairs["_district"].ne("")
    ]

    pca_pairs = pca[[pca_state_col, pca_district_col]].drop_duplicates()
    pca_pairs = pca_pairs.assign(
        _state=pca_pairs[pca_state_col].map(normalize_name),
        _district=pca_pairs[pca_district_col].map(normalize_name),
    )
    pca_pairs = pca_pairs.loc[
        pca_pairs["_state"].ne("") & pca_pairs["_district"].ne("")
    ]

    postal_key_set = set(zip(postal_pairs["_state"], postal_pairs["_district"]))
    unmatched_pca = pca_pairs.loc[
        ~pca_pairs.apply(
            lambda row: (row["_state"], row["_district"]) in postal_key_set,
            axis=1,
        )
    ].copy()
    unmatched_pca = unmatched_pca.sort_values(["_state", "_district"])

    postal_names_by_state = {
        state: sorted(group["_district"].unique())
        for state, group in postal_pairs.groupby("_state")
    }
    candidates = []
    for _, row in unmatched_pca.iterrows():
        pca_name = row["_district"]
        for postal_name in postal_names_by_state.get(row["_state"], []):
            reasons = classify_candidate(pca_name, postal_name)
            if reasons:
                candidates.append(
                    {
                        "state": row["_state"],
                        "pca_district": pca_name,
                        "postal_district_candidate": postal_name,
                        "reasons": reasons,
                    }
                )

    candidate_by_pca = {}
    for candidate in candidates:
        candidate_by_pca.setdefault(
            (candidate["state"], candidate["pca_district"]), []
        ).append(candidate)

    unmatched_by_state = unmatched_pca.groupby("_state").size().sort_index()
    simple_difference_keys = {
        (candidate["state"], candidate["pca_district"])
        for candidate in candidates
        if any(
            reason in candidate["reasons"]
            for reason in [
                "punctuation difference",
                "possible abbreviation",
                "possible spelling difference",
            ]
        )
    }
    historical_difference_keys = {
        (candidate["state"], candidate["pca_district"])
        for candidate in candidates
        if "possible old/new, renamed, split, or reorganized district"
        in candidate["reasons"]
    }
    all_candidate_keys = set(candidate_by_pca)
    unclear_keys = {
        (row["_state"], row["_district"])
        for _, row in unmatched_pca.iterrows()
    } - all_candidate_keys

    print("=== Geographic Mismatch Diagnostic Report ===")
    print("\nCompared columns:")
    print(f"  Postal State: {postal_state_col}")
    print(f"  Postal District: {postal_district_col}")
    print(f"  PCA State Name: {pca_state_col}")
    print(f"  PCA District Name: {pca_district_col}")
    print("  Comparison only: string conversion, strip, casefold")
    print("  Fuzzy matching: not used")

    print("\n1. All unmatched PCA State + District pairs:")
    for _, row in unmatched_pca.iterrows():
        print(f"  State={row[pca_state_col]!r}, District={row[pca_district_col]!r}")

    print("\n2. Unmatched PCA districts grouped by State:")
    for state, count in unmatched_by_state.items():
        print(f"  State={state!r}: {count}")

    print("\n3. Unmatched PCA district names:")
    for _, row in unmatched_pca.iterrows():
        print(
            f"  State Name={row[pca_state_col]!r}, "
            f"District Name={row[pca_district_col]!r}"
        )

    print("\n4-6. Candidate name variations within the same State:")
    if not candidates:
        print("  No deterministic name-variation candidates found.")
    else:
        for candidate in candidates:
            print(
                f"  State={candidate['state']!r}; "
                f"PCA={candidate['pca_district']!r}; "
                f"Postal candidate={candidate['postal_district_candidate']!r}; "
                f"Candidate reasons={', '.join(candidate['reasons'])}"
            )
    print("  Candidates are diagnostic only; no mappings were assigned.")

    print("\n7. Summary")
    print(f"  Likely simple naming-difference candidates: {len(simple_difference_keys)}")
    print(
        "  Likely historical boundary or administrative-change candidates: "
        f"{len(historical_difference_keys)}"
    )
    print(f"  Unclear cases with no candidate: {len(unclear_keys)}")
    print(f"  Total unmatched PCA pairs analyzed: {len(unmatched_pca)}")
    print("\nNo datasets were merged, modified, dropped, or manually mapped.")


if __name__ == "__main__":
    analyze_geographic_mismatches()
