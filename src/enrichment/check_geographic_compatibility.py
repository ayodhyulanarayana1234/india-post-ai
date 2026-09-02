from pathlib import Path

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


def standardize_for_comparison(series):
    return series.astype("string").str.strip().str.casefold()


def build_pair_series(state_values, district_values):
    return pd.Series(
        [
            (state, district)
            if pd.notna(state) and pd.notna(district)
            else None
            for state, district in zip(state_values, district_values)
        ],
        index=state_values.index,
        dtype="object",
    )


def print_pair_examples(title, pairs):
    print(f"\n{title}")
    examples = sorted(pairs)[:20]
    if examples:
        for state, district in examples:
            print(f"  State={state!r}, District={district!r}")
    else:
        print("  None")


def check_geographic_compatibility():
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

    postal_state = standardize_for_comparison(postal[postal_state_col])
    postal_district = standardize_for_comparison(postal[postal_district_col])
    pca_state = standardize_for_comparison(pca[pca_state_col])
    pca_district = standardize_for_comparison(pca[pca_district_col])

    postal_pair_series = build_pair_series(postal_state, postal_district)
    pca_pair_series = build_pair_series(pca_state, pca_district)
    postal_pairs = set(postal_pair_series.dropna())
    pca_pairs = set(pca_pair_series.dropna())
    postal_states = set(postal_state.dropna())
    pca_states = set(pca_state.dropna())

    matching_states = postal_states & pca_states
    postal_states_missing_from_pca = postal_states - pca_states
    pca_states_missing_from_postal = pca_states - postal_states
    matching_pairs = postal_pairs & pca_pairs
    postal_pairs_missing_from_pca = postal_pairs - pca_pairs
    pca_pairs_missing_from_postal = pca_pairs - postal_pairs

    pca_duplicate_pair_rows = int(pca_pair_series.duplicated().sum())
    pca_pairs_are_unique = pca_duplicate_pair_rows == 0
    postal_rows_with_pca = int(postal_pair_series.isin(pca_pairs).sum())
    postal_rows_without_pca = len(postal) - postal_rows_with_pca
    coverage_percentage = (
        100 * postal_rows_with_pca / len(postal) if len(postal) else 0.0
    )

    print("=== Geographic Compatibility Validation Report ===")
    print("\nIdentified columns:")
    print(f"  Postal State: {postal_state_col}")
    print(f"  Postal District: {postal_district_col}")
    print(f"  PCA State Name: {pca_state_col}")
    print(f"  PCA District Name: {pca_district_col}")
    print("  Comparison standardization: string, strip, casefold")

    print("\nA. Postal network")
    print(f"  Rows: {len(postal)}")
    print(f"  Unique states: {len(postal_states)}")
    print(f"  Unique districts: {len(set(postal_district.dropna()))}")
    print(f"  Unique State + District pairs: {len(postal_pairs)}")

    print("\nB. PCA enrichment")
    print(f"  Rows: {len(pca)}")
    print(f"  Unique states: {len(pca_states)}")
    print(f"  Unique districts: {len(set(pca_district.dropna()))}")
    print(f"  Unique State + District pairs: {len(pca_pairs)}")

    print("\nC. State-level compatibility")
    print(f"  Matching states: {len(matching_states)}")
    print(
        "  States in postal network but missing from PCA: "
        f"{len(postal_states_missing_from_pca)}"
    )
    print(
        "  States in PCA but missing from postal network: "
        f"{len(pca_states_missing_from_postal)}"
    )

    print("\nD. District-level compatibility")
    print(f"  Matching State + District pairs: {len(matching_pairs)}")
    print(
        "  Postal State + District pairs missing from PCA: "
        f"{len(postal_pairs_missing_from_pca)}"
    )
    print(
        "  PCA State + District pairs missing from postal network: "
        f"{len(pca_pairs_missing_from_postal)}"
    )

    print_pair_examples(
        "E. Unmatched postal State + District examples (maximum 20):",
        postal_pairs_missing_from_pca,
    )
    print_pair_examples(
        "E. Unmatched PCA State + District examples (maximum 20):",
        pca_pairs_missing_from_postal,
    )

    print("\nF. PCA State + District uniqueness")
    print(f"  Every PCA pair is unique: {pca_pairs_are_unique}")
    print(f"  Duplicate PCA pair rows: {pca_duplicate_pair_rows}")

    print("\nG. Postal row coverage by PCA pair")
    print(f"  Postal rows with a matching PCA record: {postal_rows_with_pca}")
    print(f"  Postal rows with no PCA record: {postal_rows_without_pca}")

    print("\nH. Postal rows covered by PCA demographics")
    print(f"  Coverage percentage: {coverage_percentage:.2f}%")
    print("\nNo datasets were merged, modified, or dropped.")


if __name__ == "__main__":
    check_geographic_compatibility()
