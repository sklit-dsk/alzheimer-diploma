"""E05 - Subject-level threshold and sensitivity-specificity analysis.

This script uses the E04 subject-level mean probability score to describe
threshold behavior on the existing E03/E04 test set. It does not retrain the
Random Forest, tune the model, or select a final validated threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]
LABEL_MAP = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}

EXPECTED_SUBJECTS = 29
EXPECTED_CLASS_COUNTS = {NEGATIVE_CLASS: 17, POSITIVE_CLASS: 12}
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

EXPECTED_E04_DEFAULT = {
    "accuracy": 0.6206896552,
    "balanced_accuracy": 0.5906862745,
    "sensitivity": 0.4166666667,
    "specificity": 0.7647058824,
    "confusion_matrix": [[13, 4], [7, 5]],
}
EXPECTED_E04_ROC_AUC = 0.6568627451
EXPECTED_E04_PR_AUC = 0.5628292498
METRIC_TOLERANCE = 1e-9

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
E04_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E04_subject_level" / "results"
SUBJECT_PREDICTIONS_PATH = E04_RESULTS_DIR / "subject_level_predictions.csv"
E04_SUMMARY_PATH = E04_RESULTS_DIR / "e04_summary.json"
RESULTS_DIR = SCRIPT_DIR / "results"

REQUIRED_COLUMNS = [
    "subject_id",
    "true_class",
    "true_label",
    "mean_probability",
]

NO_FINAL_THRESHOLD_STATEMENT = (
    "E05 performs descriptive threshold analysis on the existing 29-subject test set. "
    "No optimal, final, clinically recommended, or independently validated threshold is selected."
)
TEST_SET_REUSE_LIMITATION = (
    "The same E03/E04 test labels are used to describe threshold behavior. "
    "A deployment threshold would require an independent validation cohort or nested cross-validation."
)


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


def ensure_required_columns(frame: pd.DataFrame, required_columns: list[str], source: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_subject_predictions() -> pd.DataFrame:
    subject_level = pd.read_csv(SUBJECT_PREDICTIONS_PATH)
    ensure_required_columns(subject_level, REQUIRED_COLUMNS, "E04 subject-level predictions")
    subject_level = subject_level.copy()
    subject_level["true_label"] = pd.to_numeric(
        subject_level["true_label"], errors="raise"
    ).astype(int)
    subject_level["mean_probability"] = pd.to_numeric(
        subject_level["mean_probability"], errors="raise"
    )
    validate_subject_predictions(subject_level)
    return subject_level.sort_values("subject_id").reset_index(drop=True)


def validate_subject_predictions(subject_level: pd.DataFrame) -> None:
    if len(subject_level) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_SUBJECTS} subject rows; observed {len(subject_level)}."
        )

    unique_subjects = int(subject_level["subject_id"].nunique())
    if unique_subjects != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} unique subject_id values; observed {unique_subjects}."
        )

    if subject_level["subject_id"].duplicated().any():
        duplicates = subject_level.loc[
            subject_level["subject_id"].duplicated(), "subject_id"
        ].tolist()
        raise ValueError(f"Duplicate subject_id rows found: {duplicates}")

    class_counts = subject_level["true_class"].value_counts().to_dict()
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Unexpected E04 test subject class counts. "
            f"Expected {EXPECTED_CLASS_COUNTS}; observed {class_counts}."
        )

    class_label_pairs = (
        subject_level[["true_class", "true_label"]].drop_duplicates().to_dict(orient="records")
    )
    for pair in class_label_pairs:
        expected_label = LABEL_MAP.get(pair["true_class"])
        if expected_label is None:
            raise ValueError(f"Unexpected true_class value: {pair['true_class']}")
        if int(pair["true_label"]) != expected_label:
            raise ValueError(
                "Class-to-label mapping mismatch. "
                f"Observed {pair['true_class']} -> {pair['true_label']}; "
                f"expected {expected_label}."
            )

    subject_label_counts = subject_level.groupby("subject_id")["true_label"].nunique()
    inconsistent_labels = subject_label_counts[subject_label_counts != 1]
    if not inconsistent_labels.empty:
        raise ValueError(
            f"Subjects with more than one true label: {inconsistent_labels.to_dict()}"
        )

    missing_scores = int(subject_level["mean_probability"].isna().sum())
    if missing_scores:
        raise ValueError(f"Missing mean_probability values found: {missing_scores}")

    out_of_range = subject_level[
        ~subject_level["mean_probability"].between(0.0, 1.0, inclusive="both")
    ]
    if not out_of_range.empty:
        raise ValueError(
            "mean_probability values outside [0, 1] found: "
            f"{out_of_range[['subject_id', 'mean_probability']].to_dict(orient='records')}"
        )


def validate_e04_summary_reference() -> None:
    if not E04_SUMMARY_PATH.exists():
        raise ValueError(f"E04 summary file is missing: {E04_SUMMARY_PATH}")

    with E04_SUMMARY_PATH.open("r", encoding="utf-8") as input_file:
        summary = json.load(input_file)

    mean_metrics = summary.get("metric_summaries", {}).get("mean_probability")
    if not mean_metrics:
        raise ValueError("E04 summary does not contain mean_probability metrics.")

    for metric_name, expected_value in EXPECTED_E04_DEFAULT.items():
        if metric_name == "confusion_matrix":
            observed_cm = mean_metrics.get("confusion_matrix")
            if observed_cm != expected_value:
                raise ValueError(
                    "E04 summary mean_probability confusion matrix does not match the "
                    f"confirmed reference. Observed {observed_cm}; expected {expected_value}."
                )
        else:
            observed_value = float(mean_metrics.get(metric_name))
            if not np.isclose(observed_value, expected_value, atol=METRIC_TOLERANCE):
                raise ValueError(
                    f"E04 summary mean_probability {metric_name}={observed_value} does not "
                    f"match confirmed reference {expected_value}."
                )

    observed_roc_auc = float(mean_metrics.get("roc_auc"))
    observed_pr_auc = float(mean_metrics.get("pr_auc"))
    if not np.isclose(observed_roc_auc, EXPECTED_E04_ROC_AUC, atol=METRIC_TOLERANCE):
        raise ValueError(
            f"E04 summary ROC-AUC={observed_roc_auc} does not match {EXPECTED_E04_ROC_AUC}."
        )
    if not np.isclose(observed_pr_auc, EXPECTED_E04_PR_AUC, atol=METRIC_TOLERANCE):
        raise ValueError(
            f"E04 summary PR-AUC={observed_pr_auc} does not match {EXPECTED_E04_PR_AUC}."
        )


def calculate_threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_threshold_metrics(subject_level: pd.DataFrame) -> pd.DataFrame:
    y_true = subject_level["true_label"].to_numpy(dtype=int)
    scores = subject_level["mean_probability"].to_numpy(dtype=float)
    rows = [calculate_threshold_metrics(y_true, scores, threshold) for threshold in THRESHOLD_GRID]
    metrics = pd.DataFrame(rows)
    validate_default_threshold(metrics)
    return metrics


def validate_default_threshold(threshold_metrics: pd.DataFrame) -> None:
    default_rows = threshold_metrics[np.isclose(threshold_metrics["threshold"], DEFAULT_THRESHOLD)]
    if len(default_rows) != 1:
        raise ValueError("Threshold grid must contain exactly one default threshold row at 0.50.")

    default_row = default_rows.iloc[0]
    observed_cm = [
        [int(default_row["TN"]), int(default_row["FP"])],
        [int(default_row["FN"]), int(default_row["TP"])],
    ]
    if observed_cm != EXPECTED_E04_DEFAULT["confusion_matrix"]:
        raise ValueError(
            "E05 threshold 0.50 confusion matrix does not reproduce E04. "
            f"Observed {observed_cm}; expected {EXPECTED_E04_DEFAULT['confusion_matrix']}."
        )

    for metric_name, expected_value in EXPECTED_E04_DEFAULT.items():
        if metric_name == "confusion_matrix":
            continue
        observed_value = float(default_row[metric_name])
        if not np.isclose(observed_value, expected_value, atol=METRIC_TOLERANCE):
            raise ValueError(
                f"E05 threshold 0.50 {metric_name}={observed_value} does not reproduce "
                f"E04 reference {expected_value}."
            )


def calculate_roc_outputs(subject_level: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    y_true = subject_level["true_label"].to_numpy(dtype=int)
    scores = subject_level["mean_probability"].to_numpy(dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = float(roc_auc_score(y_true, scores))
    if not np.isclose(roc_auc, EXPECTED_E04_ROC_AUC, atol=METRIC_TOLERANCE):
        raise ValueError(
            f"E05 ROC-AUC={roc_auc} does not reproduce E04 mean_probability ROC-AUC "
            f"{EXPECTED_E04_ROC_AUC}."
        )
    roc_points = pd.DataFrame(
        {
            "roc_threshold": thresholds,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "specificity": 1.0 - fpr,
        }
    )
    return roc_points, roc_auc


def calculate_pr_outputs(subject_level: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    y_true = subject_level["true_label"].to_numpy(dtype=int)
    scores = subject_level["mean_probability"].to_numpy(dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = float(average_precision_score(y_true, scores))
    if not np.isclose(pr_auc, EXPECTED_E04_PR_AUC, atol=METRIC_TOLERANCE):
        raise ValueError(
            f"E05 PR-AUC={pr_auc} does not reproduce E04 mean_probability PR-AUC "
            f"{EXPECTED_E04_PR_AUC}."
        )
    threshold_values = np.append(thresholds, np.nan)
    pr_points = pd.DataFrame(
        {
            "pr_threshold": threshold_values,
            "precision": precision,
            "recall": recall,
        }
    )
    return pr_points, pr_auc


def threshold_row_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "threshold": float(row["threshold"]),
        "sensitivity": float(row["sensitivity"]),
        "specificity": float(row["specificity"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "precision": float(row["precision"]),
        "f1_positive": float(row["f1_positive"]),
        "confusion_matrix": [
            [int(row["TN"]), int(row["FP"])],
            [int(row["FN"]), int(row["TP"])],
        ],
    }


def rows_matching_value(frame: pd.DataFrame, column: str, value: float) -> list[dict[str, Any]]:
    rows = frame[np.isclose(frame[column], value)]
    return [threshold_row_payload(row) for _, row in rows.iterrows()]


def lowest_threshold_at_sensitivity(
    threshold_metrics: pd.DataFrame, target_sensitivity: float
) -> dict[str, Any] | None:
    candidates = threshold_metrics[threshold_metrics["sensitivity"] >= target_sensitivity]
    if candidates.empty:
        return None
    row = candidates.sort_values("threshold").iloc[0]
    return threshold_row_payload(row)


def highest_threshold_at_sensitivity(
    threshold_metrics: pd.DataFrame, target_sensitivity: float
) -> dict[str, Any] | None:
    candidates = threshold_metrics[threshold_metrics["sensitivity"] >= target_sensitivity]
    if candidates.empty:
        return None
    row = candidates.sort_values("threshold", ascending=False).iloc[0]
    return threshold_row_payload(row)


def identify_operating_points(threshold_metrics: pd.DataFrame) -> dict[str, Any]:
    default_row = threshold_metrics[
        np.isclose(threshold_metrics["threshold"], DEFAULT_THRESHOLD)
    ].iloc[0]
    max_sensitivity = float(threshold_metrics["sensitivity"].max())
    max_balanced_accuracy = float(threshold_metrics["balanced_accuracy"].max())
    closeness = (threshold_metrics["sensitivity"] - threshold_metrics["specificity"]).abs()
    closest_difference = float(closeness.min())
    closest_rows = threshold_metrics[np.isclose(closeness, closest_difference)]

    return {
        "default_operating_point": threshold_row_payload(default_row),
        "highest_sensitivity_in_grid": {
            "value": max_sensitivity,
            "thresholds": rows_matching_value(threshold_metrics, "sensitivity", max_sensitivity),
        },
        "highest_balanced_accuracy_in_grid": {
            "value": max_balanced_accuracy,
            "thresholds": rows_matching_value(
                threshold_metrics, "balanced_accuracy", max_balanced_accuracy
            ),
        },
        "sensitivity_specificity_closest_in_grid": {
            "absolute_difference": closest_difference,
            "thresholds": [
                threshold_row_payload(row) for _, row in closest_rows.iterrows()
            ],
        },
        "lowest_threshold_with_sensitivity_at_least_0_70": lowest_threshold_at_sensitivity(
            threshold_metrics, 0.70
        ),
        "lowest_threshold_with_sensitivity_at_least_0_80": lowest_threshold_at_sensitivity(
            threshold_metrics, 0.80
        ),
        "highest_threshold_with_sensitivity_at_least_0_70": highest_threshold_at_sensitivity(
            threshold_metrics, 0.70
        ),
        "highest_threshold_with_sensitivity_at_least_0_80": highest_threshold_at_sensitivity(
            threshold_metrics, 0.80
        ),
        "interpretation_note": (
            "These are descriptive operating points on the existing test set, not optimal, "
            "validated, clinically recommended, or final thresholds."
        ),
    }


def flatten_operating_points(operating_points: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rows(point_name: str, payload: Any) -> None:
        if payload is None:
            rows.append({"operating_point": point_name, "available": False})
            return
        if isinstance(payload, dict) and "threshold" in payload:
            row = {"operating_point": point_name, "available": True}
            row.update(payload)
            row["confusion_matrix"] = json.dumps(payload["confusion_matrix"])
            rows.append(row)
            return
        if isinstance(payload, dict) and "thresholds" in payload:
            threshold_rows = payload["thresholds"]
            for item in threshold_rows:
                row = {"operating_point": point_name, "available": True}
                row.update(item)
                row["confusion_matrix"] = json.dumps(item["confusion_matrix"])
                rows.append(row)

    add_rows("default_operating_point", operating_points["default_operating_point"])
    add_rows("highest_sensitivity_in_grid", operating_points["highest_sensitivity_in_grid"])
    add_rows(
        "highest_balanced_accuracy_in_grid",
        operating_points["highest_balanced_accuracy_in_grid"],
    )
    add_rows(
        "sensitivity_specificity_closest_in_grid",
        operating_points["sensitivity_specificity_closest_in_grid"],
    )
    add_rows(
        "lowest_threshold_with_sensitivity_at_least_0_70",
        operating_points["lowest_threshold_with_sensitivity_at_least_0_70"],
    )
    add_rows(
        "lowest_threshold_with_sensitivity_at_least_0_80",
        operating_points["lowest_threshold_with_sensitivity_at_least_0_80"],
    )
    add_rows(
        "highest_threshold_with_sensitivity_at_least_0_70",
        operating_points["highest_threshold_with_sensitivity_at_least_0_70"],
    )
    add_rows(
        "highest_threshold_with_sensitivity_at_least_0_80",
        operating_points["highest_threshold_with_sensitivity_at_least_0_80"],
    )
    return pd.DataFrame(rows)


def save_roc_curve(roc_points: pd.DataFrame, roc_auc: float, output_path: Path) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=160)
    ax.plot(
        roc_points["false_positive_rate"],
        roc_points["true_positive_rate"],
        marker="o",
        linewidth=1.8,
        label=f"Mean probability ROC-AUC = {roc_auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sensitivity")
    ax.set_title("Subject-level ROC curve", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    pr_points: pd.DataFrame, pr_auc: float, output_path: Path
) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=160)
    ax.plot(
        pr_points["recall"],
        pr_points["precision"],
        marker="o",
        linewidth=1.8,
        label=f"Average precision = {pr_auc:.3f}",
    )
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Subject-level precision-recall curve", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_specificity_plot(
    threshold_metrics: pd.DataFrame, output_path: Path
) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(
        threshold_metrics["threshold"],
        threshold_metrics["sensitivity"],
        marker="o",
        linewidth=1.8,
        label="Sensitivity",
    )
    ax.plot(
        threshold_metrics["threshold"],
        threshold_metrics["specificity"],
        marker="s",
        linewidth=1.8,
        label="Specificity",
    )
    ax.axvline(DEFAULT_THRESHOLD, color="0.35", linestyle="--", linewidth=1.0, label="0.5 reference")
    ax.set_xlabel("Mean-probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Sensitivity-specificity tradeoff", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_subject_probability_distribution(
    subject_level: pd.DataFrame, output_path: Path
) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    rng = np.random.default_rng(42)
    colors = {NEGATIVE_CLASS: "#386cb0", POSITIVE_CLASS: "#d95f02"}

    for position, class_name in enumerate(CLASS_LABELS):
        class_scores = subject_level.loc[
            subject_level["true_class"] == class_name, "mean_probability"
        ].to_numpy(dtype=float)
        jitter = rng.uniform(-0.07, 0.07, size=len(class_scores))
        ax.scatter(
            np.full(len(class_scores), position) + jitter,
            class_scores,
            alpha=0.85,
            s=44,
            color=colors[class_name],
            edgecolor="white",
            linewidth=0.6,
            label=f"{class_name} subjects",
        )
        if len(class_scores):
            ax.boxplot(
                class_scores,
                positions=[position],
                widths=0.32,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": "white", "alpha": 0.35, "color": colors[class_name]},
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": colors[class_name]},
                capprops={"color": colors[class_name]},
            )

    ax.axhline(DEFAULT_THRESHOLD, color="0.25", linestyle="--", linewidth=1.0, label="0.5 reference")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_ylabel("Mean predicted score for Very mild Dementia")
    ax.set_title("Subject-level mean-probability distribution", loc="left", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    subject_level: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    roc_auc: float,
    pr_auc: float,
    operating_points: dict[str, Any],
) -> dict[str, Any]:
    default_row = threshold_metrics[
        np.isclose(threshold_metrics["threshold"], DEFAULT_THRESHOLD)
    ].iloc[0]
    class_counts = subject_level["true_class"].value_counts().to_dict()
    return {
        "experiment_id": "E05",
        "objective": "Subject-level threshold and sensitivity-specificity analysis",
        "number_of_subjects": int(len(subject_level)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "positive_class": POSITIVE_CLASS,
        "score_used": "mean_probability",
        "score_note": (
            "Mean probability is used as the primary working aggregation from E04 because it "
            "provides a direct continuous subject-level score."
        ),
        "threshold_grid": [float(value) for value in THRESHOLD_GRID],
        "default_threshold_results": threshold_row_payload(default_row),
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        "descriptive_operating_points": operating_points,
        "no_optimal_final_threshold_selected": True,
        "no_final_threshold_statement": NO_FINAL_THRESHOLD_STATEMENT,
        "methodological_limitation_test_set_reuse": TEST_SET_REUSE_LIMITATION,
        "validated_threshold_selection_requirement": (
            "Independent validation or nested cross-validation would be required for validated "
            "threshold selection."
        ),
        "interpretation_rules": [
            "All E05 metrics are calculated at subject level.",
            "The threshold grid is descriptive and uses the existing E03/E04 test set.",
            "Do not use these results to tune the Random Forest.",
            "Do not report any threshold from E05 as independently validated.",
            "Do not present mean_probability as a clinical diagnostic probability.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def print_summary(
    threshold_metrics: pd.DataFrame,
    roc_auc: float,
    pr_auc: float,
    operating_points: dict[str, Any],
) -> None:
    default = operating_points["default_operating_point"]
    highest_sensitivity = operating_points["highest_sensitivity_in_grid"]
    highest_balanced = operating_points["highest_balanced_accuracy_in_grid"]

    print("\nE05 - Subject-level threshold analysis")
    print("=" * 44)
    print(f"Subjects: {EXPECTED_SUBJECTS}")
    print(f"Score: mean_probability")
    print(f"Positive class: {POSITIVE_CLASS}")
    print(f"Thresholds evaluated: {len(threshold_metrics)}")
    print("\nDefault threshold 0.50:")
    print(
        "sensitivity={sensitivity:.4f}, specificity={specificity:.4f}, "
        "balanced_accuracy={balanced_accuracy:.4f}, confusion_matrix={confusion_matrix}".format(
            **default
        )
    )
    print(f"\nROC-AUC: {roc_auc:.10f}")
    print(f"PR-AUC / average precision: {pr_auc:.10f}")
    print(
        "\nHighest sensitivity in grid: "
        f"{highest_sensitivity['value']:.4f} at "
        f"{[item['threshold'] for item in highest_sensitivity['thresholds']]}"
    )
    print(
        "Highest balanced accuracy in grid: "
        f"{highest_balanced['value']:.4f} at "
        f"{[item['threshold'] for item in highest_balanced['thresholds']]}"
    )
    print("\nNo optimal or final threshold was selected in E05.")
    print(TEST_SET_REUSE_LIMITATION)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validate_e04_summary_reference()
    subject_level = load_subject_predictions()

    threshold_metrics = build_threshold_metrics(subject_level)
    threshold_metrics.to_csv(RESULTS_DIR / "threshold_metrics.csv", index=False)

    sensitivity_specificity_table = threshold_metrics[
        [
            "threshold",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "precision",
            "f1_positive",
        ]
    ].copy()
    sensitivity_specificity_table.to_csv(
        RESULTS_DIR / "sensitivity_specificity_table.csv", index=False
    )

    roc_points, roc_auc = calculate_roc_outputs(subject_level)
    roc_points.to_csv(RESULTS_DIR / "roc_curve_points.csv", index=False)
    save_roc_curve(roc_points, roc_auc, RESULTS_DIR / "roc_curve.png")

    pr_points, pr_auc = calculate_pr_outputs(subject_level)
    pr_points.to_csv(RESULTS_DIR / "pr_curve_points.csv", index=False)
    save_precision_recall_curve(pr_points, pr_auc, RESULTS_DIR / "precision_recall_curve.png")

    save_sensitivity_specificity_plot(
        threshold_metrics, RESULTS_DIR / "sensitivity_specificity_by_threshold.png"
    )
    save_subject_probability_distribution(
        subject_level, RESULTS_DIR / "subject_probability_distribution.png"
    )

    operating_points = identify_operating_points(threshold_metrics)
    flatten_operating_points(operating_points).to_csv(
        RESULTS_DIR / "descriptive_operating_points.csv", index=False
    )

    summary = build_summary(
        subject_level=subject_level,
        threshold_metrics=threshold_metrics,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        operating_points=operating_points,
    )
    write_json(RESULTS_DIR / "e05_summary.json", summary)
    print_summary(threshold_metrics, roc_auc, pr_auc, operating_points)


if __name__ == "__main__":
    main()
