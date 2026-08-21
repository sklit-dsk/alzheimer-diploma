"""E02 - Cohort construction and age control.

Builds the Age >= 60 cohort selected after E01 and validates the resulting
subject-level and scan-level metadata before downstream modeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


BINARY_CLASSES = ["Non Demented", "Very mild Dementia"]
CLASS_ORDER = ["Non Demented", "Very mild Dementia"]
NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
AGE_THRESHOLD = 60

EXPECTED_SUBJECT_COUNTS = {
    "Non Demented": 85,
    "Very mild Dementia": 58,
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
E01_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E01_clinical_metadata" / "results"
E01_SUBJECT_LEVEL_PATH = E01_RESULTS_DIR / "subject_level_metadata.csv"
E01_SCAN_LEVEL_PATH = E01_RESULTS_DIR / "scan_level_metadata.csv"
RESULTS_DIR = SCRIPT_DIR / "results"


def ensure_required_columns(
    frame: pd.DataFrame, required_columns: list[str], source_name: str
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_python(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def load_subject_metadata() -> pd.DataFrame:
    subject_level = pd.read_csv(E01_SUBJECT_LEVEL_PATH)
    ensure_required_columns(
        subject_level,
        [
            "subject_id",
            "class",
            "Age",
            "Sex",
            "CDR",
            "MMSE",
            "number_of_scans",
            "number_of_images",
        ],
        "E01 subject-level metadata",
    )
    return subject_level


def load_scan_metadata() -> pd.DataFrame:
    scan_level = pd.read_csv(E01_SCAN_LEVEL_PATH)
    ensure_required_columns(
        scan_level,
        ["scan_id", "subject_id", "class", "number_of_images", "Age", "Sex", "CDR", "MMSE"],
        "E01 scan-level metadata",
    )
    return scan_level


def build_subject_cohort(subject_level: pd.DataFrame) -> pd.DataFrame:
    binary_subjects = subject_level[subject_level["class"].isin(BINARY_CLASSES)].copy()
    binary_subjects["Age"] = pd.to_numeric(binary_subjects["Age"], errors="coerce")

    cohort = binary_subjects[binary_subjects["Age"] >= AGE_THRESHOLD].copy()
    cohort = cohort.sort_values(["class", "subject_id"]).reset_index(drop=True)
    validate_subject_cohort(cohort)
    return cohort


def validate_subject_cohort(cohort: pd.DataFrame) -> None:
    if cohort["subject_id"].duplicated().any():
        duplicates = cohort.loc[cohort["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicate subject_id rows found in E02 cohort: {duplicates}")

    if cohort["class"].isna().any():
        raise ValueError("E02 cohort contains missing class values.")

    invalid_classes = sorted(set(cohort["class"].dropna()) - set(BINARY_CLASSES))
    if invalid_classes:
        raise ValueError(f"E02 cohort contains unexpected classes: {invalid_classes}")

    counts = cohort.groupby("class")["subject_id"].nunique().to_dict()
    expected_total = sum(EXPECTED_SUBJECT_COUNTS.values())
    observed_total = int(cohort["subject_id"].nunique())
    if counts != EXPECTED_SUBJECT_COUNTS or observed_total != expected_total:
        raise ValueError(
            "Unexpected E02 cohort subject counts. "
            f"Expected {EXPECTED_SUBJECT_COUNTS} and total {expected_total}; "
            f"observed {counts} and total {observed_total}."
        )

    if cohort["Age"].isna().any():
        missing_subjects = cohort.loc[cohort["Age"].isna(), "subject_id"].tolist()
        raise ValueError(f"E02 cohort contains missing Age values: {missing_subjects}")

    if (cohort["Age"] < AGE_THRESHOLD).any():
        invalid_subjects = cohort.loc[cohort["Age"] < AGE_THRESHOLD, "subject_id"].tolist()
        raise ValueError(f"E02 cohort contains subjects younger than {AGE_THRESHOLD}: {invalid_subjects}")

    if cohort["Sex"].isna().any() or (cohort["Sex"].astype(str).str.strip() == "").any():
        missing_subjects = cohort.loc[
            cohort["Sex"].isna() | (cohort["Sex"].astype(str).str.strip() == ""),
            "subject_id",
        ].tolist()
        raise ValueError(f"E02 cohort contains missing Sex values: {missing_subjects}")

    cdr_consistency = create_cdr_consistency(cohort)
    inconsistent = cdr_consistency[cdr_consistency["inconsistent_cdr_count"] > 0]
    if not inconsistent.empty:
        raise ValueError(
            "CDR inconsistency detected in E02 cohort: "
            f"{inconsistent.to_dict(orient='records')}"
        )


def build_scan_cohort(scan_level: pd.DataFrame, subject_cohort: pd.DataFrame) -> pd.DataFrame:
    selected_subjects = set(subject_cohort["subject_id"])
    scan_cohort = scan_level[scan_level["subject_id"].isin(selected_subjects)].copy()
    scan_cohort = scan_cohort.sort_values(["class", "subject_id", "scan_id"]).reset_index(drop=True)
    validate_scan_cohort(scan_cohort, subject_cohort)
    return scan_cohort


def validate_scan_cohort(scan_cohort: pd.DataFrame, subject_cohort: pd.DataFrame) -> None:
    selected_subjects = set(subject_cohort["subject_id"])
    scan_subjects = set(scan_cohort["subject_id"])

    missing_scan_subjects = sorted(selected_subjects - scan_subjects)
    if missing_scan_subjects:
        raise ValueError(f"E02 subjects without scans: {missing_scan_subjects}")

    excluded_subjects = sorted(scan_subjects - selected_subjects)
    if excluded_subjects:
        raise ValueError(f"Scan cohort contains excluded subjects: {excluded_subjects}")

    if scan_cohort["scan_id"].isna().any():
        raise ValueError("E02 scan-level cohort contains missing scan_id values.")

    if scan_cohort["scan_id"].duplicated().any():
        duplicates = scan_cohort.loc[scan_cohort["scan_id"].duplicated(), "scan_id"].tolist()
        raise ValueError(f"Duplicate scan_id rows found in E02 scan cohort: {duplicates}")

    subject_counts_match = len(scan_subjects) == subject_cohort["subject_id"].nunique()
    if not subject_counts_match:
        raise ValueError(
            "Subject counts from scan-level and subject-level E02 cohorts do not match. "
            f"Scan-level subjects: {len(scan_subjects)}; "
            f"subject-level subjects: {subject_cohort['subject_id'].nunique()}."
        )

    expected_classes = subject_cohort[["subject_id", "class"]].rename(
        columns={"class": "subject_level_class"}
    )
    class_check = scan_cohort[["scan_id", "subject_id", "class"]].merge(
        expected_classes,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    mismatched = class_check[class_check["class"] != class_check["subject_level_class"]]
    if not mismatched.empty:
        raise ValueError(
            "Scan-level class values do not match subject-level class values: "
            f"{mismatched.to_dict(orient='records')}"
        )

    scans_per_subject = scan_cohort.groupby("subject_id")["scan_id"].nunique()
    expected_scan_counts = subject_cohort.set_index("subject_id")["number_of_scans"]
    scan_count_check = pd.concat(
        [expected_scan_counts.rename("subject_level_scans"), scans_per_subject.rename("scan_rows")],
        axis=1,
    ).fillna(0)
    mismatched_scan_counts = scan_count_check[
        scan_count_check["subject_level_scans"] != scan_count_check["scan_rows"]
    ]
    if not mismatched_scan_counts.empty:
        raise ValueError(
            "Subject-level number_of_scans does not match retained scan rows: "
            f"{mismatched_scan_counts.reset_index().to_dict(orient='records')}"
        )


def descriptive_summary(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        values = (
            pd.to_numeric(frame.loc[frame["class"] == class_name, value_column], errors="coerce")
            .dropna()
            .astype(float)
        )
        rows.append(
            {
                "class": class_name,
                "count": int(values.count()),
                "mean": values.mean() if not values.empty else np.nan,
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median": values.median() if not values.empty else np.nan,
                "min": values.min() if not values.empty else np.nan,
                "max": values.max() if not values.empty else np.nan,
                "Q1": values.quantile(0.25) if not values.empty else np.nan,
                "Q3": values.quantile(0.75) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cliffs_delta(group_a: pd.Series, group_b: pd.Series) -> float:
    a_values = group_a.to_numpy(dtype=float)
    b_values = group_b.to_numpy(dtype=float)
    if len(a_values) == 0 or len(b_values) == 0:
        return np.nan

    differences = a_values[:, None] - b_values[None, :]
    greater = np.sum(differences > 0)
    lesser = np.sum(differences < 0)
    return float((greater - lesser) / differences.size)


def compare_age_distributions(cohort: pd.DataFrame) -> pd.DataFrame:
    frame = cohort[["class", "Age"]].copy()
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    non_demented = frame.loc[frame["class"] == NEGATIVE_CLASS, "Age"].dropna()
    very_mild = frame.loc[frame["class"] == POSITIVE_CLASS, "Age"].dropna()

    mann_whitney = stats.mannwhitneyu(
        very_mild,
        non_demented,
        alternative="two-sided",
        method="auto",
    )

    return pd.DataFrame(
        [
            {
                "cohort": f"Age >= {AGE_THRESHOLD}",
                "comparison": "Very mild Dementia minus Non Demented",
                "non_demented_n": int(len(non_demented)),
                "very_mild_dementia_n": int(len(very_mild)),
                "difference_in_means": very_mild.mean() - non_demented.mean(),
                "difference_in_medians": very_mild.median() - non_demented.median(),
                "mann_whitney_u_statistic": float(mann_whitney.statistic),
                "mann_whitney_p_value": float(mann_whitney.pvalue),
                "effect_size": cliffs_delta(very_mild, non_demented),
                "effect_size_name": "Cliff's delta",
                "effect_size_direction": "positive means ages are higher in Very mild Dementia",
                "test_note": "Two-sided Mann-Whitney U; statistical significance is not clinical importance.",
            }
        ]
    )


def distribution_table(
    frame: pd.DataFrame, value_column: str, missing_label: str = "Missing"
) -> pd.DataFrame:
    data = frame[["class", value_column]].copy()
    data[value_column] = data[value_column].astype("object")
    data.loc[data[value_column].isna(), value_column] = missing_label
    data[value_column] = data[value_column].astype(str)

    counts = (
        data.groupby(["class", value_column], dropna=False)
        .size()
        .reset_index(name="subjects")
    )
    class_totals = data.groupby("class").size().rename("class_total").reset_index()
    counts = counts.merge(class_totals, on="class", how="left")
    counts["percent_within_class"] = (
        counts["subjects"] / counts["class_total"] * 100
    ).round(4)
    return counts.sort_values(["class", value_column])


def missing_values_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_subjects = len(cohort)
    for variable in ["Age", "Sex", "CDR", "MMSE"]:
        missing_count = int(cohort[variable].isna().sum())
        rows.append(
            {
                "variable": variable,
                "total_subjects": total_subjects,
                "available_count": total_subjects - missing_count,
                "missing_count": missing_count,
                "missing_percent": round((missing_count / total_subjects) * 100, 4),
            }
        )
    return pd.DataFrame(rows)


def create_cdr_consistency(cohort: pd.DataFrame) -> pd.DataFrame:
    expected_cdr = {NEGATIVE_CLASS: 0.0, POSITIVE_CLASS: 0.5}
    rows: list[dict[str, Any]] = []

    for class_name in CLASS_ORDER:
        class_rows = cohort[cohort["class"] == class_name].copy()
        cdr_values = pd.to_numeric(class_rows["CDR"], errors="coerce")
        expected = expected_cdr[class_name]
        missing_count = int(cdr_values.isna().sum())
        available = cdr_values.dropna()
        consistent_count = int(np.isclose(available, expected).sum())
        inconsistent_count = int(len(available) - consistent_count)
        inconsistent_subjects = class_rows.loc[
            cdr_values.notna() & ~np.isclose(cdr_values, expected), "subject_id"
        ].tolist()

        rows.append(
            {
                "class": class_name,
                "expected_cdr_when_available": expected,
                "subjects": int(len(class_rows)),
                "available_cdr_count": int(len(available)),
                "missing_cdr_count": missing_count,
                "consistent_cdr_count": consistent_count,
                "inconsistent_cdr_count": inconsistent_count,
                "inconsistent_subject_ids": ";".join(inconsistent_subjects),
                "note": "Missing CDR remains missing and is not interpreted as 0.",
            }
        )

    return pd.DataFrame(rows)


def retained_percentages(subject_level: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    binary_subjects = subject_level[subject_level["class"].isin(BINARY_CLASSES)].copy()
    original_counts = binary_subjects.groupby("class")["subject_id"].nunique().to_dict()
    retained_counts = cohort.groupby("class")["subject_id"].nunique().to_dict()

    rows = []
    for class_name in CLASS_ORDER:
        original = int(original_counts.get(class_name, 0))
        retained = int(retained_counts.get(class_name, 0))
        rows.append(
            {
                "class": class_name,
                "original_e01_binary_subjects": original,
                "retained_e02_subjects": retained,
                "retained_percent": round((retained / original) * 100, 4)
                if original
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_json_summary(
    subject_level: pd.DataFrame,
    subject_cohort: pd.DataFrame,
    scan_cohort: pd.DataFrame,
    age_summary: pd.DataFrame,
    age_comparison: pd.DataFrame,
    sex_distribution: pd.DataFrame,
    cdr_distribution: pd.DataFrame,
    cdr_consistency: pd.DataFrame,
    mmse_summary: pd.DataFrame,
    missing_values: pd.DataFrame,
    retained: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "experiment_id": "E02",
        "objective": "Cohort construction and age control",
        "primary_strategy": f"Age >= {AGE_THRESHOLD}",
        "binary_classes": BINARY_CLASSES,
        "subject_counts": {
            str(key): int(value)
            for key, value in subject_cohort.groupby("class")["subject_id"].nunique().items()
        },
        "total_subjects": int(subject_cohort["subject_id"].nunique()),
        "scan_counts": {
            "total_scans": int(scan_cohort["scan_id"].nunique()),
            "subjects_with_scans": int(scan_cohort["subject_id"].nunique()),
        },
        "retained_percentages_relative_to_e01_binary_cohort": retained.to_dict(
            orient="records"
        ),
        "age_statistics": age_summary.to_dict(orient="records"),
        "age_comparison": age_comparison.to_dict(orient="records"),
        "sex_distribution": sex_distribution.to_dict(orient="records"),
        "cdr_distribution": cdr_distribution.to_dict(orient="records"),
        "CDR_consistency": cdr_consistency.to_dict(orient="records"),
        "MMSE_summary": mmse_summary.to_dict(orient="records"),
        "missing_values": missing_values.to_dict(orient="records"),
        "validation": {
            "expected_subject_counts": EXPECTED_SUBJECT_COUNTS,
            "age_missing": int(subject_cohort["Age"].isna().sum()),
            "sex_missing": int(subject_cohort["Sex"].isna().sum()),
            "duplicate_subject_id_rows": int(subject_cohort["subject_id"].duplicated().sum()),
            "subjects_without_scans": int(
                len(set(subject_cohort["subject_id"]) - set(scan_cohort["subject_id"]))
            ),
            "excluded_subjects_in_scan_cohort": int(
                len(set(scan_cohort["subject_id"]) - set(subject_cohort["subject_id"]))
            ),
        },
        "interpretation_rules": [
            "E02 constructs the Age >= 60 cohort selected from E01.",
            "Missing CDR remains missing and is not interpreted as 0.",
            "The Mann-Whitney U p-value is not interpreted as clinical importance.",
            "Approximate age matching remains optional or a sensitivity analysis.",
            "No model is trained or evaluated in E02.",
        ],
        "input_subject_rows": int(len(subject_level)),
    }
    return to_plain_python(summary)


def print_summary(
    subject_cohort: pd.DataFrame,
    scan_cohort: pd.DataFrame,
    age_summary: pd.DataFrame,
    age_comparison: pd.DataFrame,
    retained: pd.DataFrame,
) -> None:
    subject_counts = subject_cohort.groupby("class")["subject_id"].nunique()
    scan_counts = scan_cohort.groupby("class")["scan_id"].nunique()

    print("\nE02 - Cohort construction / age control")
    print("=" * 48)
    print(f"Strategy: Age >= {AGE_THRESHOLD}")
    print(f"Total subjects: {subject_cohort['subject_id'].nunique()}")
    print(f"Total scans: {scan_cohort['scan_id'].nunique()}")

    print("\nSubjects by class:")
    for class_name in CLASS_ORDER:
        print(
            f"- {class_name}: {int(subject_counts.get(class_name, 0))} subjects, "
            f"{int(scan_counts.get(class_name, 0))} scans"
        )

    print("\nRetained relative to E01 binary cohort:")
    print(retained.to_string(index=False))

    print("\nAge summary:")
    print(age_summary.to_string(index=False))

    print("\nAge comparison:")
    print(age_comparison.to_string(index=False))

    print("\nE02 cohort files were written to experiments/E02_age_control/results/.")
    print("No model training was performed.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subject_level = load_subject_metadata()
    scan_level = load_scan_metadata()

    subject_cohort = build_subject_cohort(subject_level)
    scan_cohort = build_scan_cohort(scan_level, subject_cohort)

    output_subject_columns = [
        "subject_id",
        "class",
        "Age",
        "Sex",
        "CDR",
        "MMSE",
        "number_of_scans",
        "number_of_images",
    ]
    subject_cohort_output = subject_cohort[output_subject_columns].copy()

    age_summary = descriptive_summary(subject_cohort, "Age")
    age_comparison = compare_age_distributions(subject_cohort)
    sex_distribution = distribution_table(subject_cohort, "Sex")
    cdr_distribution = distribution_table(subject_cohort, "CDR")
    cdr_consistency = create_cdr_consistency(subject_cohort)
    mmse_summary = descriptive_summary(subject_cohort, "MMSE")
    missing_values = missing_values_summary(subject_cohort)
    retained = retained_percentages(subject_level, subject_cohort)

    subject_cohort_output.to_csv(RESULTS_DIR / "e02_subject_level_cohort.csv", index=False)
    scan_cohort.to_csv(RESULTS_DIR / "e02_scan_level_cohort.csv", index=False)
    age_summary.to_csv(RESULTS_DIR / "e02_age_summary.csv", index=False)
    age_comparison.to_csv(RESULTS_DIR / "e02_age_comparison.csv", index=False)
    sex_distribution.to_csv(RESULTS_DIR / "e02_sex_distribution.csv", index=False)
    cdr_distribution.to_csv(RESULTS_DIR / "e02_cdr_distribution.csv", index=False)
    cdr_consistency.to_csv(RESULTS_DIR / "e02_cdr_consistency.csv", index=False)
    mmse_summary.to_csv(RESULTS_DIR / "e02_mmse_summary.csv", index=False)
    missing_values.to_csv(RESULTS_DIR / "e02_missing_values.csv", index=False)
    retained.to_csv(RESULTS_DIR / "e02_retained_percentages.csv", index=False)

    summary = write_json_summary(
        subject_level=subject_level,
        subject_cohort=subject_cohort,
        scan_cohort=scan_cohort,
        age_summary=age_summary,
        age_comparison=age_comparison,
        sex_distribution=sex_distribution,
        cdr_distribution=cdr_distribution,
        cdr_consistency=cdr_consistency,
        mmse_summary=mmse_summary,
        missing_values=missing_values,
        retained=retained,
    )
    with (RESULTS_DIR / "e02_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print_summary(
        subject_cohort=subject_cohort,
        scan_cohort=scan_cohort,
        age_summary=age_summary,
        age_comparison=age_comparison,
        retained=retained,
    )


if __name__ == "__main__":
    main()
