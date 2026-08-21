"""E01 - Clinical metadata and confounder analysis.

This script connects ImagesOASIS scan identifiers from E00 with OASIS-1
clinical metadata, then creates scan-level and subject-level summaries for
the Non Demented vs Very mild Dementia binary scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BINARY_CLASSES = ["Non Demented", "Very mild Dementia"]
CLINICAL_COLUMNS = ["ID", "M/F", "Age", "MMSE", "CDR"]
SUBJECT_CLINICAL_COLUMNS = ["Age", "Sex", "CDR", "MMSE"]
CLASS_ORDER = ["Non Demented", "Very mild Dementia"]
POSITIVE_CLASS = "Very mild Dementia"
NEGATIVE_CLASS = "Non Demented"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
E00_METADATA_PATH = PROJECT_ROOT / "experiments" / "E00_baseline_rf" / "dataset_metadata.csv"
CLINICAL_METADATA_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)
RESULTS_DIR = SCRIPT_DIR / "results"


def ensure_required_columns(
    frame: pd.DataFrame, required_columns: list[str], source_name: str
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def numeric_values(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return sorted(pd.unique(values).tolist())


def text_values(series: pd.Series) -> list[str]:
    cleaned = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
    )
    return sorted(cleaned.unique().tolist())


def resolve_values(
    series: pd.Series,
    *,
    numeric: bool,
    level: str,
    entity_id: str,
    variable: str,
    conflict_rows: list[dict[str, Any]],
) -> Any:
    values: list[Any] = numeric_values(series) if numeric else text_values(series)

    if len(values) == 0:
        return pd.NA

    if len(values) == 1:
        return values[0]

    conflict_rows.append(
        {
            "level": level,
            "entity_id": entity_id,
            "variable": variable,
            "values": ";".join(str(value) for value in values),
            "n_distinct_non_missing_values": len(values),
        }
    )
    return pd.NA


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


def load_binary_imagesoasis_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(E00_METADATA_PATH)
    ensure_required_columns(
        metadata,
        ["path", "class", "subject_id", "scan_id", "filename"],
        "ImagesOASIS E00 metadata",
    )

    binary_metadata = metadata[metadata["class"].isin(BINARY_CLASSES)].copy()
    if binary_metadata.empty:
        raise ValueError("No binary-class rows were found in the E00 metadata.")
    return binary_metadata


def load_clinical_metadata() -> pd.DataFrame:
    clinical = pd.read_excel(CLINICAL_METADATA_PATH)
    ensure_required_columns(clinical, CLINICAL_COLUMNS, "OASIS-1 clinical metadata")
    clinical = clinical.copy()
    clinical["ID"] = clinical["ID"].astype(str).str.strip()
    clinical = clinical.rename(columns={"M/F": "Sex"})
    return clinical


def build_imagesoasis_scan_table(
    binary_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for scan_id, scan_rows in binary_metadata.groupby("scan_id", sort=True):
        subject_values = text_values(scan_rows["subject_id"])
        class_values = text_values(scan_rows["class"])

        if len(subject_values) != 1:
            conflicts.append(
                {
                    "level": "imagesoasis_scan",
                    "entity_id": scan_id,
                    "variable": "subject_id",
                    "values": ";".join(subject_values),
                    "n_distinct_non_missing_values": len(subject_values),
                }
            )
        if len(class_values) != 1:
            conflicts.append(
                {
                    "level": "imagesoasis_scan",
                    "entity_id": scan_id,
                    "variable": "class",
                    "values": ";".join(class_values),
                    "n_distinct_non_missing_values": len(class_values),
                }
            )

        rows.append(
            {
                "scan_id": scan_id,
                "subject_id": subject_values[0] if len(subject_values) == 1 else pd.NA,
                "class": class_values[0] if len(class_values) == 1 else pd.NA,
                "number_of_images": int(len(scan_rows)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(conflicts)


def build_clinical_scan_table(
    clinical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for clinical_id, clinical_rows in clinical.groupby("ID", sort=True):
        rows.append(
            {
                "clinical_id": clinical_id,
                "Age": resolve_values(
                    clinical_rows["Age"],
                    numeric=True,
                    level="clinical_scan_id",
                    entity_id=clinical_id,
                    variable="Age",
                    conflict_rows=conflicts,
                ),
                "Sex": resolve_values(
                    clinical_rows["Sex"],
                    numeric=False,
                    level="clinical_scan_id",
                    entity_id=clinical_id,
                    variable="Sex",
                    conflict_rows=conflicts,
                ),
                "CDR": resolve_values(
                    clinical_rows["CDR"],
                    numeric=True,
                    level="clinical_scan_id",
                    entity_id=clinical_id,
                    variable="CDR",
                    conflict_rows=conflicts,
                ),
                "MMSE": resolve_values(
                    clinical_rows["MMSE"],
                    numeric=True,
                    level="clinical_scan_id",
                    entity_id=clinical_id,
                    variable="MMSE",
                    conflict_rows=conflicts,
                ),
                "source_row_count": int(len(clinical_rows)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(conflicts)


def create_scan_level_metadata(
    imagesoasis_scans: pd.DataFrame, clinical_scans: pd.DataFrame
) -> pd.DataFrame:
    scan_level = imagesoasis_scans.merge(
        clinical_scans,
        how="left",
        left_on="scan_id",
        right_on="clinical_id",
        validate="one_to_one",
    )
    scan_level["matched_clinical_metadata"] = scan_level["clinical_id"].notna()
    ordered_columns = [
        "scan_id",
        "subject_id",
        "class",
        "number_of_images",
        "matched_clinical_metadata",
        "clinical_id",
        "Age",
        "Sex",
        "CDR",
        "MMSE",
        "source_row_count",
    ]
    return scan_level[ordered_columns].sort_values(["class", "subject_id", "scan_id"])


def create_subject_level_metadata(
    scan_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for subject_id, subject_scans in scan_level.groupby("subject_id", sort=True):
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "class": resolve_values(
                subject_scans["class"],
                numeric=False,
                level="subject",
                entity_id=subject_id,
                variable="class",
                conflict_rows=conflicts,
            ),
            "Age": resolve_values(
                subject_scans["Age"],
                numeric=True,
                level="subject",
                entity_id=subject_id,
                variable="Age",
                conflict_rows=conflicts,
            ),
            "Sex": resolve_values(
                subject_scans["Sex"],
                numeric=False,
                level="subject",
                entity_id=subject_id,
                variable="Sex",
                conflict_rows=conflicts,
            ),
            "CDR": resolve_values(
                subject_scans["CDR"],
                numeric=True,
                level="subject",
                entity_id=subject_id,
                variable="CDR",
                conflict_rows=conflicts,
            ),
            "MMSE": resolve_values(
                subject_scans["MMSE"],
                numeric=True,
                level="subject",
                entity_id=subject_id,
                variable="MMSE",
                conflict_rows=conflicts,
            ),
            "number_of_scans": int(subject_scans["scan_id"].nunique()),
            "number_of_images": int(subject_scans["number_of_images"].sum()),
            "scan_ids": ";".join(sorted(subject_scans["scan_id"].dropna().astype(str))),
        }
        subject_rows.append(row)

    subject_level = pd.DataFrame(subject_rows)

    for column in SUBJECT_CLINICAL_COLUMNS:
        conflict_ids = {
            row["entity_id"]
            for row in conflicts
            if row["level"] == "subject" and row["variable"] == column
        }
        subject_level[f"{column}_conflict"] = subject_level["subject_id"].isin(conflict_ids)

    ordered_columns = [
        "subject_id",
        "class",
        "Age",
        "Sex",
        "CDR",
        "MMSE",
        "number_of_scans",
        "number_of_images",
        "scan_ids",
        "Age_conflict",
        "Sex_conflict",
        "CDR_conflict",
        "MMSE_conflict",
    ]
    return subject_level[ordered_columns].sort_values(["class", "subject_id"]), pd.DataFrame(conflicts)


def create_missing_values_summary(subject_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_subjects = len(subject_level)
    for column in SUBJECT_CLINICAL_COLUMNS:
        missing_count = int(subject_level[column].isna().sum())
        conflict_column = f"{column}_conflict"
        rows.append(
            {
                "variable": column,
                "total_subjects": total_subjects,
                "available_count": int(total_subjects - missing_count),
                "missing_count": missing_count,
                "missing_percent": round((missing_count / total_subjects) * 100, 4),
                "conflict_count": int(subject_level[conflict_column].sum())
                if conflict_column in subject_level.columns
                else 0,
            }
        )
    return pd.DataFrame(rows)


def descriptive_summary(
    frame: pd.DataFrame, value_column: str, group_column: str = "class"
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        values = (
            pd.to_numeric(
                frame.loc[frame[group_column] == class_name, value_column],
                errors="coerce",
            )
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


def compare_age_distributions(
    subject_level: pd.DataFrame, cohort_label: str = "All binary subjects"
) -> pd.DataFrame:
    age_frame = subject_level[["class", "Age"]].copy()
    age_frame["Age"] = pd.to_numeric(age_frame["Age"], errors="coerce")
    non_demented = age_frame.loc[age_frame["class"] == NEGATIVE_CLASS, "Age"].dropna()
    very_mild = age_frame.loc[age_frame["class"] == POSITIVE_CLASS, "Age"].dropna()

    if len(non_demented) > 0 and len(very_mild) > 0:
        mann_whitney = stats.mannwhitneyu(
            very_mild,
            non_demented,
            alternative="two-sided",
            method="auto",
        )
        u_statistic = float(mann_whitney.statistic)
        p_value = float(mann_whitney.pvalue)
    else:
        u_statistic = np.nan
        p_value = np.nan

    return pd.DataFrame(
        [
            {
                "cohort": cohort_label,
                "comparison": "Very mild Dementia minus Non Demented",
                "non_demented_n": int(len(non_demented)),
                "very_mild_dementia_n": int(len(very_mild)),
                "difference_in_means": very_mild.mean() - non_demented.mean(),
                "difference_in_medians": very_mild.median() - non_demented.median(),
                "mann_whitney_u_statistic": u_statistic,
                "mann_whitney_p_value": p_value,
                "effect_size": cliffs_delta(very_mild, non_demented),
                "effect_size_name": "Cliff's delta",
                "effect_size_direction": "positive means ages are higher in Very mild Dementia",
                "test_note": "Two-sided Mann-Whitney U; statistical significance is not clinical importance.",
            }
        ]
    )


def make_distribution_table(
    subject_level: pd.DataFrame, value_column: str, missing_label: str = "Missing"
) -> pd.DataFrame:
    frame = subject_level[["class", value_column]].copy()
    frame[value_column] = frame[value_column].astype("object")
    frame.loc[frame[value_column].isna(), value_column] = missing_label
    frame[value_column] = frame[value_column].astype(str)

    counts = (
        frame.groupby(["class", value_column], dropna=False)
        .size()
        .reset_index(name="subjects")
    )
    class_totals = frame.groupby("class").size().rename("class_total").reset_index()
    counts = counts.merge(class_totals, on="class", how="left")
    counts["percent_within_class"] = (
        counts["subjects"] / counts["class_total"] * 100
    ).round(4)
    return counts.sort_values(["class", value_column])


def create_cdr_consistency(subject_level: pd.DataFrame) -> pd.DataFrame:
    expected_cdr = {NEGATIVE_CLASS: 0.0, POSITIVE_CLASS: 0.5}
    rows: list[dict[str, Any]] = []

    for class_name in CLASS_ORDER:
        class_rows = subject_level[subject_level["class"] == class_name].copy()
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
                "note": "Missing CDR is kept missing and is not interpreted as 0.",
            }
        )

    return pd.DataFrame(rows)


def create_age_60plus_summary(subject_level: pd.DataFrame) -> pd.DataFrame:
    frame = subject_level.copy()
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    original_counts = frame.groupby("class").size().to_dict()
    retained = frame[frame["Age"] >= 60].copy()
    stats_frame = descriptive_summary(retained, "Age")

    retained_counts = retained.groupby("class").size().to_dict()
    stats_frame["original_subjects"] = stats_frame["class"].map(original_counts).fillna(0).astype(int)
    stats_frame["retained_subjects"] = stats_frame["class"].map(retained_counts).fillna(0).astype(int)
    stats_frame["retained_percent"] = (
        stats_frame["retained_subjects"] / stats_frame["original_subjects"] * 100
    ).replace([np.inf, -np.inf], np.nan).round(4)
    stats_frame["note"] = (
        "Exploratory 60+ cohort summary; E01 does not create or validate the E02 cohort dataset."
    )

    first_columns = ["class", "original_subjects", "retained_subjects", "retained_percent"]
    other_columns = [column for column in stats_frame.columns if column not in first_columns]
    return stats_frame[first_columns + other_columns]


def create_age_matching_feasibility(subject_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = subject_level[["class", "Age"]].copy()
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = frame.dropna(subset=["Age"]).copy()
    frame["Age"] = frame["Age"].astype(int)
    frame["age_5year_bin"] = (frame["Age"] // 5) * 5

    exact_age = (
        frame.pivot_table(
            index="Age",
            columns="class",
            values="Age",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for class_name in CLASS_ORDER:
        if class_name not in exact_age.columns:
            exact_age[class_name] = 0
    exact_age["possible_exact_age_pairs"] = exact_age[CLASS_ORDER].min(axis=1)
    exact_age = exact_age[["Age", *CLASS_ORDER, "possible_exact_age_pairs"]].sort_values("Age")

    five_year = (
        frame.pivot_table(
            index="age_5year_bin",
            columns="class",
            values="Age",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for class_name in CLASS_ORDER:
        if class_name not in five_year.columns:
            five_year[class_name] = 0
    five_year["possible_5year_bin_pairs"] = five_year[CLASS_ORDER].min(axis=1)
    five_year = five_year[
        ["age_5year_bin", *CLASS_ORDER, "possible_5year_bin_pairs"]
    ].sort_values("age_5year_bin")

    return exact_age, five_year


def plot_age_distribution(subject_level: pd.DataFrame, output_path: Path) -> None:
    frame = subject_level[["class", "Age"]].copy()
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = frame.dropna(subset=["Age"])

    plt.figure(figsize=(9, 5.5))
    all_ages = frame["Age"].to_numpy(dtype=float)
    bins = np.arange(np.floor(all_ages.min()) - 0.5, np.ceil(all_ages.max()) + 1.5, 2)

    for class_name in CLASS_ORDER:
        values = frame.loc[frame["class"] == class_name, "Age"].to_numpy(dtype=float)
        plt.hist(values, bins=bins, alpha=0.55, label=f"{class_name} (n={len(values)})")

    plt.title("Subject-level age distribution by class")
    plt.xlabel("Age")
    plt.ylabel("Number of subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_sex_distribution(sex_distribution: pd.DataFrame, output_path: Path) -> None:
    pivot = sex_distribution.pivot_table(
        index="class",
        columns="Sex",
        values="subjects",
        fill_value=0,
        aggfunc="sum",
    )
    pivot = pivot.reindex(CLASS_ORDER)

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Subject-level sex distribution by class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of subjects")
    ax.legend(title="Sex")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def matching_summary(scan_level: pd.DataFrame) -> dict[str, Any]:
    total_scans = int(len(scan_level))
    matched_scans = int(scan_level["matched_clinical_metadata"].sum())
    total_subjects = int(scan_level["subject_id"].nunique())

    subject_match = (
        scan_level.groupby("subject_id")["matched_clinical_metadata"]
        .agg(["any", "all"])
        .reset_index()
    )
    subjects_with_any_match = int(subject_match["any"].sum())
    subjects_with_all_scans_matched = int(subject_match["all"].sum())

    return {
        "total_scans": total_scans,
        "matched_scans": matched_scans,
        "unmatched_scans": total_scans - matched_scans,
        "scan_coverage_percent": round((matched_scans / total_scans) * 100, 4)
        if total_scans
        else None,
        "total_subjects": total_subjects,
        "subjects_with_any_matched_scan": subjects_with_any_match,
        "subjects_with_all_scans_matched": subjects_with_all_scans_matched,
        "subjects_with_all_scans_matched_percent": round(
            (subjects_with_all_scans_matched / total_subjects) * 100, 4
        )
        if total_subjects
        else None,
    }


def write_json_summary(
    subject_level: pd.DataFrame,
    scan_level: pd.DataFrame,
    missing_values: pd.DataFrame,
    age_summary: pd.DataFrame,
    age_comparison: pd.DataFrame,
    cdr_consistency: pd.DataFrame,
    age_60plus_summary: pd.DataFrame,
    age_60plus_comparison: pd.DataFrame,
    clinical_conflicts: pd.DataFrame,
    imagesoasis_conflicts: pd.DataFrame,
    subject_conflicts: pd.DataFrame,
    exact_age_feasibility: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "experiment_id": "E01",
        "objective": "Clinical metadata and confounder analysis",
        "binary_classes": BINARY_CLASSES,
        "positive_class": POSITIVE_CLASS,
        "number_of_binary_subjects": int(subject_level["subject_id"].nunique()),
        "number_of_scans": int(scan_level["scan_id"].nunique()),
        "matching_coverage": matching_summary(scan_level),
        "subjects_per_class": {
            str(key): int(value)
            for key, value in subject_level.groupby("class")["subject_id"].nunique().items()
        },
        "age_statistics": age_summary.to_dict(orient="records"),
        "missing_values": missing_values.to_dict(orient="records"),
        "CDR_consistency": cdr_consistency.to_dict(orient="records"),
        "clinical_value_conflicts": int(len(clinical_conflicts)),
        "imagesoasis_metadata_conflicts": int(len(imagesoasis_conflicts)),
        "subject_level_clinical_conflicts": int(len(subject_conflicts)),
        "age_60plus_cohort_counts": age_60plus_summary[
            ["class", "original_subjects", "retained_subjects", "retained_percent"]
        ].to_dict(orient="records"),
        "key_statistical_comparison_results": age_comparison.to_dict(orient="records"),
        "age_60plus_statistical_comparison_results": age_60plus_comparison.to_dict(
            orient="records"
        ),
        "age_matching_feasibility": {
            "possible_exact_age_pairs": int(
                exact_age_feasibility["possible_exact_age_pairs"].sum()
            )
            if not exact_age_feasibility.empty
            else 0,
            "note": "This is only feasibility information. The final matched cohort belongs to E02.",
        },
        "interpretation_rules": [
            "Missing CDR is kept missing and is not interpreted as 0.",
            "The Mann-Whitney U p-value is not interpreted as clinical importance.",
            "E01 selects the Age >= 60 strategy for implementation in E02, but does not create or validate the E02 cohort dataset.",
            "The outputs describe a research prototype dataset, not a clinical diagnostic system.",
        ],
    }
    return to_plain_python(summary)


def print_human_summary(
    subject_level: pd.DataFrame,
    scan_level: pd.DataFrame,
    missing_values: pd.DataFrame,
    age_summary: pd.DataFrame,
    age_comparison: pd.DataFrame,
    cdr_consistency: pd.DataFrame,
    age_60plus_summary: pd.DataFrame,
    age_60plus_comparison: pd.DataFrame,
    clinical_conflicts: pd.DataFrame,
    subject_conflicts: pd.DataFrame,
) -> None:
    coverage = matching_summary(scan_level)
    subjects_per_class = subject_level.groupby("class")["subject_id"].nunique()

    print("\nE01 - Clinical metadata and confounder analysis")
    print("=" * 55)
    print(f"Binary subjects: {subject_level['subject_id'].nunique()}")
    print(f"Unique scans: {scan_level['scan_id'].nunique()}")
    print(
        "Scan-level metadata coverage: "
        f"{coverage['matched_scans']}/{coverage['total_scans']} "
        f"({coverage['scan_coverage_percent']}%)"
    )
    print(
        "Subjects with all scans matched: "
        f"{coverage['subjects_with_all_scans_matched']}/{coverage['total_subjects']} "
        f"({coverage['subjects_with_all_scans_matched_percent']}%)"
    )

    print("\nSubjects per class:")
    for class_name in CLASS_ORDER:
        print(f"- {class_name}: {int(subjects_per_class.get(class_name, 0))}")

    print("\nMissing values at subject level:")
    for row in missing_values.to_dict(orient="records"):
        print(
            f"- {row['variable']}: {row['missing_count']} missing "
            f"({row['missing_percent']}%), conflicts: {row['conflict_count']}"
        )

    print("\nAge summary by class:")
    print(age_summary.to_string(index=False))

    print("\nAge comparison:")
    print(age_comparison.to_string(index=False))

    print("\nCDR consistency:")
    print(cdr_consistency.to_string(index=False))

    print("\nExploratory age >= 60 cohort:")
    print(
        age_60plus_summary[
            ["class", "original_subjects", "retained_subjects", "retained_percent"]
        ].to_string(index=False)
    )

    print("\nAge comparison within exploratory age >= 60 cohort:")
    print(age_60plus_comparison.to_string(index=False))

    if len(clinical_conflicts) or len(subject_conflicts):
        print("\nClinical value conflicts were detected and saved in the results directory.")
    else:
        print("\nNo conflicting clinical values were detected in resolved scan/subject data.")

    print("\nE01 selects Age >= 60 as the primary E02 strategy but does not create E02 data.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binary_metadata = load_binary_imagesoasis_metadata()
    clinical_metadata = load_clinical_metadata()

    imagesoasis_scans, imagesoasis_conflicts = build_imagesoasis_scan_table(binary_metadata)
    clinical_scans, clinical_conflicts = build_clinical_scan_table(clinical_metadata)
    scan_level = create_scan_level_metadata(imagesoasis_scans, clinical_scans)
    subject_level, subject_conflicts = create_subject_level_metadata(scan_level)

    unmatched_scans = scan_level.loc[~scan_level["matched_clinical_metadata"]].copy()
    unmatched_subjects = (
        unmatched_scans[["subject_id"]].drop_duplicates().sort_values("subject_id")
        if not unmatched_scans.empty
        else pd.DataFrame(columns=["subject_id"])
    )

    missing_values = create_missing_values_summary(subject_level)
    age_summary = descriptive_summary(subject_level, "Age")
    age_comparison = compare_age_distributions(subject_level)
    sex_distribution = make_distribution_table(subject_level, "Sex")
    cdr_distribution = make_distribution_table(subject_level, "CDR")
    cdr_consistency = create_cdr_consistency(subject_level)
    mmse_summary = descriptive_summary(subject_level, "MMSE")
    age_60plus_summary = create_age_60plus_summary(subject_level)
    age_60plus_subjects = subject_level[
        pd.to_numeric(subject_level["Age"], errors="coerce") >= 60
    ].copy()
    age_60plus_comparison = compare_age_distributions(
        age_60plus_subjects, cohort_label="Age >= 60 subjects"
    )
    exact_age_feasibility, five_year_feasibility = create_age_matching_feasibility(subject_level)

    scan_level.to_csv(RESULTS_DIR / "scan_level_metadata.csv", index=False)
    subject_level.to_csv(RESULTS_DIR / "subject_level_metadata.csv", index=False)
    missing_values.to_csv(RESULTS_DIR / "missing_values.csv", index=False)
    age_summary.to_csv(RESULTS_DIR / "age_summary.csv", index=False)
    age_comparison.to_csv(RESULTS_DIR / "age_comparison.csv", index=False)
    age_60plus_comparison.to_csv(RESULTS_DIR / "age_60plus_comparison.csv", index=False)
    sex_distribution.to_csv(RESULTS_DIR / "sex_distribution.csv", index=False)
    cdr_distribution.to_csv(RESULTS_DIR / "cdr_distribution.csv", index=False)
    cdr_consistency.to_csv(RESULTS_DIR / "cdr_consistency.csv", index=False)
    mmse_summary.to_csv(RESULTS_DIR / "mmse_summary.csv", index=False)
    age_60plus_summary.to_csv(RESULTS_DIR / "age_60plus_summary.csv", index=False)
    exact_age_feasibility.to_csv(RESULTS_DIR / "age_matching_exact_age_counts.csv", index=False)
    five_year_feasibility.to_csv(RESULTS_DIR / "age_matching_5year_bins.csv", index=False)
    unmatched_scans.to_csv(RESULTS_DIR / "unmatched_scans.csv", index=False)
    unmatched_subjects.to_csv(RESULTS_DIR / "unmatched_subjects.csv", index=False)
    imagesoasis_conflicts.to_csv(RESULTS_DIR / "imagesoasis_metadata_conflicts.csv", index=False)
    clinical_conflicts.to_csv(RESULTS_DIR / "clinical_value_conflicts.csv", index=False)
    subject_conflicts.to_csv(RESULTS_DIR / "subject_level_clinical_conflicts.csv", index=False)

    plot_age_distribution(subject_level, RESULTS_DIR / "age_distribution.png")
    plot_sex_distribution(sex_distribution, RESULTS_DIR / "sex_distribution.png")

    summary = write_json_summary(
        subject_level=subject_level,
        scan_level=scan_level,
        missing_values=missing_values,
        age_summary=age_summary,
        age_comparison=age_comparison,
        cdr_consistency=cdr_consistency,
        age_60plus_summary=age_60plus_summary,
        age_60plus_comparison=age_60plus_comparison,
        clinical_conflicts=clinical_conflicts,
        imagesoasis_conflicts=imagesoasis_conflicts,
        subject_conflicts=subject_conflicts,
        exact_age_feasibility=exact_age_feasibility,
    )
    with (RESULTS_DIR / "e01_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print_human_summary(
        subject_level=subject_level,
        scan_level=scan_level,
        missing_values=missing_values,
        age_summary=age_summary,
        age_comparison=age_comparison,
        cdr_consistency=cdr_consistency,
        age_60plus_summary=age_60plus_summary,
        age_60plus_comparison=age_60plus_comparison,
        clinical_conflicts=clinical_conflicts,
        subject_conflicts=subject_conflicts,
    )


if __name__ == "__main__":
    main()
