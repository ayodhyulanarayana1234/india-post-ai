from pathlib import Path

import pandas as pd

from apply_safe_format_mappings import (
    DATASET3_PATH,
    EXPECTED_APPROVED,
    KEY_COLUMNS,
    POSTAL_PATH,
    collect_approved_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PCA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset2.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic_crosswalk.csv"
PCA_ENRICHMENT_PATH = PROJECT_ROOT / "data" / "processed" / "pca_district_enrichment.csv"
EXPECTED_PCA_ROWS = 640
POSTAL_EVIDENCE_COLUMNS = ["Postal State", "Postal District"]


def nonempty_values(dataframe, column):
    return dataframe[column].notna() & dataframe[column].astype("string").str.strip().ne("")


def validate_final_geographic_crosswalk():
    paths = [
        (PCA_PATH, "Dataset2"),
        (CROSSWALK_PATH, "geographic crosswalk"),
        (PCA_ENRICHMENT_PATH, "PCA enrichment"),
        (POSTAL_PATH, "postal network"),
        (DATASET3_PATH, "Dataset3"),
    ]
    for path, label in paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    pca = pd.read_csv(PCA_PATH, low_memory=False)
    final_crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)
    pca_enrichment = pd.read_csv(PCA_ENRICHMENT_PATH, low_memory=False)
    postal = pd.read_csv(POSTAL_PATH, low_memory=False)
    dataset3 = pd.read_excel(DATASET3_PATH)

    required_pca_columns = set(KEY_COLUMNS + ["Level", "TRU"])
    missing_pca_columns = required_pca_columns - set(pca.columns)
    if missing_pca_columns:
        raise ValueError(f"Dataset2 is missing columns: {sorted(missing_pca_columns)}")
    required_crosswalk_columns = set(KEY_COLUMNS + ["State Name", "District Name"])
    missing_crosswalk_columns = required_crosswalk_columns - set(final_crosswalk.columns)
    if missing_crosswalk_columns:
        raise ValueError(
            "Final crosswalk is missing columns: "
            f"{sorted(missing_crosswalk_columns)}"
        )

    pca_district_total = pca.loc[
        pca["Level"].eq("DISTRICT") & pca["TRU"].eq("Total")
    ].copy()
    pca_rows = len(pca_district_total)
    final_rows = len(final_crosswalk)
    pca_keys = set(zip(pca_district_total["State"], pca_district_total["District"]))
    crosswalk_keys = set(zip(final_crosswalk["State"], final_crosswalk["District"]))
    missing_pca_keys = pca_keys - crosswalk_keys
    unexpected_crosswalk_keys = crosswalk_keys - pca_keys
    duplicate_keys = int(final_crosswalk.duplicated(KEY_COLUMNS).sum())
    missing_state_names = int((~nonempty_values(final_crosswalk, "State Name")).sum())
    missing_district_names = int((~nonempty_values(final_crosswalk, "District Name")).sum())

    state_codes_unchanged = not missing_pca_keys and not unexpected_crosswalk_keys
    district_codes_unchanged = state_codes_unchanged
    no_row_multiplication = final_rows == pca_rows == EXPECTED_PCA_ROWS
    no_pca_record_dropped = pca_rows == EXPECTED_PCA_ROWS and not missing_pca_keys

    approved, _ = collect_approved_records(
        dataset3,
        pca_enrichment,
        postal,
        final_crosswalk,
    )
    approved_keys = {item["code_key"] for item in approved}
    approved_key_names = {
        item["code_key"]: (
            item["candidate"]["Candidate Postal State"],
            item["candidate"]["Candidate Postal District"],
        )
        for item in approved
    }
    approved_count = len(approved)

    postal_columns_present = all(
        column in final_crosswalk.columns for column in POSTAL_EVIDENCE_COLUMNS
    )
    postal_columns_absent = all(
        column not in final_crosswalk.columns for column in POSTAL_EVIDENCE_COLUMNS
    )
    postal_evidence_valid = postal_columns_absent
    evidence_keys = set()
    unexpected_mappings = set()
    if postal_columns_present:
        state_present = nonempty_values(final_crosswalk, "Postal State")
        district_present = nonempty_values(final_crosswalk, "Postal District")
        partial_evidence = state_present ^ district_present
        evidence_rows = final_crosswalk.loc[state_present & district_present]
        evidence_keys = set(zip(evidence_rows["State"], evidence_rows["District"]))
        unexpected_mappings = evidence_keys - approved_keys
        wrong_approved_evidence = set()
        for _, row in evidence_rows.iterrows():
            key = (row["State"], row["District"])
            expected_names = approved_key_names.get(key)
            actual_names = (row["Postal State"], row["Postal District"])
            if expected_names != actual_names:
                wrong_approved_evidence.add(key)
        postal_evidence_valid = (
            not partial_evidence.any()
            and not unexpected_mappings
            and not wrong_approved_evidence
            and evidence_keys == approved_keys
        )
    elif not postal_columns_absent:
        postal_evidence_valid = False

    approved_mappings_present = (
        approved_count == EXPECTED_APPROVED
        and approved_keys.issubset(crosswalk_keys)
        and postal_evidence_valid
    )
    unresolved_candidate_mapped = bool(unexpected_mappings)
    if postal_columns_present:
        unresolved_candidate_mapped = unresolved_candidate_mapped or not (
            evidence_keys <= approved_keys
        )
    else:
        unresolved_candidate_mapped = False

    conditions = {
        "Dataset2 has exactly 640 district-total records": pca_rows == EXPECTED_PCA_ROWS,
        "Final crosswalk has exactly 640 rows": final_rows == EXPECTED_PCA_ROWS,
        "Every PCA key exists in final crosswalk": not missing_pca_keys,
        "Every final key corresponds to a PCA district-total record": not unexpected_crosswalk_keys,
        "State + District keys are unique": duplicate_keys == 0,
        "State codes unchanged": state_codes_unchanged,
        "District codes unchanged": district_codes_unchanged,
        "State names populated": missing_state_names == 0,
        "District names populated": missing_district_names == 0,
        "Postal evidence contains only approved relationships": postal_evidence_valid,
        "All 8 approved mappings are present": approved_mappings_present,
        "No unresolved candidate was mapped": not unresolved_candidate_mapped,
        "No unexpected row multiplication": no_row_multiplication,
        "No PCA district-total record was dropped": no_pca_record_dropped,
    }
    integrity_status = "PASS" if all(conditions.values()) else "FAIL"

    print("=== Final Geographic Crosswalk Integrity Report ===")
    print(f"Dataset2 PCA district-total rows: {pca_rows}")
    print(f"Final crosswalk rows: {final_rows}")
    print(f"Missing PCA keys: {len(missing_pca_keys)}")
    print(f"Unexpected crosswalk keys: {len(unexpected_crosswalk_keys)}")
    print(f"Duplicate keys: {duplicate_keys}")
    print(f"Missing State names: {missing_state_names}")
    print(f"Missing District names: {missing_district_names}")
    print(f"State codes unchanged: {'yes' if state_codes_unchanged else 'no'}")
    print(f"District codes unchanged: {'yes' if district_codes_unchanged else 'no'}")
    print(f"Approved format mappings present: {approved_count}/{EXPECTED_APPROVED}")
    print(f"Unexpected mappings: {len(unexpected_mappings)}")
    print(f"PCA district coverage: {len(pca_keys & crosswalk_keys)}/{EXPECTED_PCA_ROWS}")
    print(f"Unresolved PCA records: {len(missing_pca_keys)}")
    print(f"No unexpected row multiplication: {'yes' if no_row_multiplication else 'no'}")
    print(f"No PCA district-total record dropped: {'yes' if no_pca_record_dropped else 'no'}")
    print(f"Final integrity status: {integrity_status}")

    if integrity_status == "FAIL":
        print("\nFailed conditions:")
        for condition, passed in conditions.items():
            if not passed:
                print(f"  {condition}")

    print("\nNo datasets were modified.")


if __name__ == "__main__":
    validate_final_geographic_crosswalk()
