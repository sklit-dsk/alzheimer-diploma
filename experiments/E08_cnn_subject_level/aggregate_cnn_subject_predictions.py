"""E08 - CNN subject-level aggregation of E07 test slice predictions.

This script evaluates the frozen E07 Attempt 2 CNN test predictions at subject
level. It does not retrain the CNN, change the checkpoint, optimize thresholds,
or use test labels for model selection.
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
    precision_score,
    roc_auc_score,
)


NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]
LABEL_MAP = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}

EXPECTED_TEST_SLICES = 6954
EXPECTED_TEST_SUBJECTS = 29
EXPECTED_TEST_SUBJECT_COUNTS = {NEGATIVE_CLASS: 17, POSITIVE_CLASS: 12}
DEFAULT_THRESHOLD = 0.5
TIE_HANDLING_RULE = (
    "Majority voting predicts Very mild Dementia only when positive_vote_fraction > 0.5. "
    "Exact ties are explicitly flagged and assigned to Non Demented by this strict rule."
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "E07_cnn"
    / "results"
    / "test_slice_predictions_attempt2.csv"
)
RESULTS_DIR = SCRIPT_DIR / "results"

REQUIRED_COLUMNS = [
    "subject_id",
    "scan_id",
    "true_class",
    "true_label",
    "probability_very_mild_dementia",
    "predicted_label_at_0_5",
]


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


def ensure_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"E07 CNN slice predictions are missing columns: {missing}")


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    ensure_required_columns(predictions)
    predictions = predictions.copy()
    predictions["true_label"] = pd.to_numeric(predictions["true_label"], errors="raise").astype(int)
    predictions["predicted_label_at_0_5"] = pd.to_numeric(
        predictions["predicted_label_at_0_5"], errors="raise"
    ).astype(int)
    predictions["probability_very_mild_dementia"] = pd.to_numeric(
        predictions["probability_very_mild_dementia"], errors="raise"
    )
    validate_slice_predictions(predictions)
    return predictions


def validate_slice_predictions(predictions: pd.DataFrame) -> None:
    row_count = int(len(predictions))
    if row_count != EXPECTED_TEST_SLICES:
        raise ValueError(f"Expected {EXPECTED_TEST_SLICES} test slice rows; observed {row_count}.")

    subject_count = int(predictions["subject_id"].nunique())
    if subject_count != EXPECTED_TEST_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SUBJECTS} unique frozen test subjects; observed {subject_count}."
        )

    subject_classes = predictions.groupby("subject_id")["true_class"].nunique()
    inconsistent_class_subjects = subject_classes[subject_classes != 1]
    if not inconsistent_class_subjects.empty:
        raise ValueError(
            "Some subjects have more than one true class: "
            f"{inconsistent_class_subjects.to_dict()}"
        )

    subject_labels = predictions.groupby("subject_id")["true_label"].nunique()
    inconsistent_label_subjects = subject_labels[subject_labels != 1]
    if not inconsistent_label_subjects.empty:
        raise ValueError(
            "Some subjects have more than one true label: "
            f"{inconsistent_label_subjects.to_dict()}"
        )

    label_mismatches = predictions[
        predictions["true_class"].map(LABEL_MAP) != predictions["true_label"]
    ]
    if not label_mismatches.empty:
        examples = label_mismatches[["subject_id", "true_class", "true_label"]].head(20)
        raise ValueError(
            "True class/label mapping mismatch found. Expected Non Demented = 0 and "
            f"Very mild Dementia = 1. Examples: {examples.to_dict(orient='records')}"
        )

    invalid_predicted_labels = sorted(set(predictions["predicted_label_at_0_5"]) - {0, 1})
    if invalid_predicted_labels:
        raise ValueError(f"Invalid predicted_label_at_0_5 values: {invalid_predicted_labels}")

    one_row_per_subject = predictions.drop_duplicates("subject_id")
    class_counts = one_row_per_subject["true_class"].value_counts().to_dict()
    if class_counts != EXPECTED_TEST_SUBJECT_COUNTS:
        raise ValueError(
            "Unexpected frozen test subject class counts. "
            f"Expected {EXPECTED_TEST_SUBJECT_COUNTS}; observed {class_counts}."
        )

    missing_probabilities = int(predictions["probability_very_mild_dementia"].isna().sum())
    if missing_probabilities:
        raise ValueError(f"Missing probabilities found: {missing_probabilities}")

    out_of_range = predictions[
        ~predictions["probability_very_mild_dementia"].between(0.0, 1.0, inclusive="both")
    ]
    if not out_of_range.empty:
        examples = out_of_range[["subject_id", "probability_very_mild_dementia"]].head(20)
        raise ValueError(
            "Probabilities outside [0, 1] found: "
            f"{examples.to_dict(orient='records')}"
        )


def aggregate_subject_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, subject_rows in predictions.groupby("subject_id", sort=True):
        true_classes = subject_rows["true_class"].dropna().unique().tolist()
        true_labels = subject_rows["true_label"].dropna().unique().tolist()
        if len(true_classes) != 1 or len(true_labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels/classes.")

        probabilities = subject_rows["probability_very_mild_dementia"]
        positive_vote_count = int((subject_rows["predicted_label_at_0_5"] == 1).sum())
        negative_vote_count = int((subject_rows["predicted_label_at_0_5"] == 0).sum())
        total_votes = positive_vote_count + negative_vote_count
        if total_votes != len(subject_rows):
            raise ValueError(f"Subject {subject_id} has invalid majority-vote counts.")

        positive_vote_fraction = positive_vote_count / total_votes
        majority_tie = bool(np.isclose(positive_vote_fraction, DEFAULT_THRESHOLD))

        mean_probability = float(probabilities.mean())
        median_probability = float(probabilities.median())
        mean_predicted_label = int(mean_probability >= DEFAULT_THRESHOLD)
        median_predicted_label = int(median_probability >= DEFAULT_THRESHOLD)
        majority_vote_predicted_label = int(positive_vote_fraction > DEFAULT_THRESHOLD)

        rows.append(
            {
                "subject_id": subject_id,
                "true_class": true_classes[0],
                "true_label": int(true_labels[0]),
                "number_of_test_images": int(len(subject_rows)),
                "mean_probability": mean_probability,
                "median_probability": median_probability,
                "positive_vote_count": positive_vote_count,
                "negative_vote_count": negative_vote_count,
                "positive_vote_fraction": positive_vote_fraction,
                "majority_vote_tie": majority_tie,
                "mean_predicted_label": mean_predicted_label,
                "median_predicted_label": median_predicted_label,
                "majority_vote_predicted_label": majority_vote_predicted_label,
                "mean_predicted_class": INVERSE_LABEL_MAP[mean_predicted_label],
                "median_predicted_class": INVERSE_LABEL_MAP[median_predicted_label],
                "majority_vote_predicted_class": INVERSE_LABEL_MAP[
                    majority_vote_predicted_label
                ],
            }
        )

    subject_level = pd.DataFrame(rows).sort_values(["true_class", "subject_id"]).reset_index(drop=True)
    validate_subject_level_table(subject_level)
    return subject_level


def validate_subject_level_table(subject_level: pd.DataFrame) -> None:
    if len(subject_level) != EXPECTED_TEST_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SUBJECTS} subject-level rows; observed {len(subject_level)}."
        )
    if subject_level["subject_id"].duplicated().any():
        duplicates = subject_level.loc[
            subject_level["subject_id"].duplicated(), "subject_id"
        ].tolist()
        raise ValueError(f"Duplicate subject rows found: {duplicates}")

    class_counts = subject_level["true_class"].value_counts().to_dict()
    if class_counts != EXPECTED_TEST_SUBJECT_COUNTS:
        raise ValueError(
            f"Unexpected subject-level class counts: {class_counts}; "
            f"expected {EXPECTED_TEST_SUBJECT_COUNTS}."
        )


def calculate_metrics(
    subject_level: pd.DataFrame,
    *,
    method_name: str,
    prediction_column: str,
    score_column: str,
) -> tuple[dict[str, Any], np.ndarray]:
    y_true = subject_level["true_label"].to_numpy(dtype=int)
    y_pred = subject_level[prediction_column].to_numpy(dtype=int)
    scores = subject_level[score_column].to_numpy(dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "experiment_id": "E08",
        "source_experiment": "E07 Attempt 2",
        "model": "ResNet18",
        "method": method_name,
        "evaluation_level": "subject",
        "number_of_subjects": int(len(subject_level)),
        "positive_class": POSITIVE_CLASS,
        "default_threshold": DEFAULT_THRESHOLD,
        "score_column": score_column,
        "prediction_column": prediction_column,
        "tie_handling_rule": TIE_HANDLING_RULE if method_name == "majority_vote" else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "recall_positive": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_LABELS,
        "metric_note": "All E08 metrics are calculated from exactly one row per subject.",
    }
    return metrics, cm


def save_confusion_matrix(cm: np.ndarray, output_csv: Path, output_png: Path, title: str) -> None:
    pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(output_csv)

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def build_comparison_table(metrics_by_method: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method_name, metrics in metrics_by_method.items():
        rows.append(
            {
                "method": method_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1_positive": metrics["f1_positive"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def best_by_descriptive_metric(comparison: pd.DataFrame) -> dict[str, Any]:
    selected_metrics = [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_positive",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]
    best = {}
    for metric in selected_metrics:
        max_value = comparison[metric].max()
        methods = comparison.loc[np.isclose(comparison[metric], max_value), "method"].tolist()
        best[metric] = {"value": float(max_value), "methods": methods}
    return best


def build_summary(
    subject_level: pd.DataFrame,
    slice_predictions: pd.DataFrame,
    metrics_by_method: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    class_counts = subject_level["true_class"].value_counts().to_dict()
    return {
        "experiment_id": "E08",
        "source_experiment": "E07 Attempt 2",
        "model": "ResNet18",
        "objective": "CNN subject-level aggregation of frozen E07 Attempt 2 test predictions",
        "number_of_test_subjects": int(len(subject_level)),
        "number_of_test_slices": int(len(slice_predictions)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "aggregation_methods": [
            "mean_probability",
            "median_probability",
            "majority_vote",
        ],
        "default_threshold": DEFAULT_THRESHOLD,
        "tie_handling_rule": TIE_HANDLING_RULE,
        "metric_summaries": metrics_by_method,
        "best_performing_method_by_selected_descriptive_metrics": best_by_descriptive_metric(
            comparison
        ),
        "methodological_recommendation_note": (
            "Do not declare one aggregation method universally superior from this small "
            "test set. Interpret sensitivity, specificity, continuous-score metrics and "
            "stability together."
        ),
        "threshold_optimization_performed": False,
        "threshold_note": (
            "E08 uses only the default 0.5 threshold. No threshold optimization or test-set "
            "threshold search was performed."
        ),
        "important_limitations": [
            "Metrics are subject-level and use exactly one row per subject.",
            "The frozen test set contains only 29 subjects.",
            "E08 does not retrain the CNN, modify the checkpoint, tune thresholds or choose a new checkpoint.",
        ],
    }


def print_summary(comparison: pd.DataFrame, subject_level: pd.DataFrame, slice_count: int) -> None:
    print("\nE08 - CNN subject-level evaluation")
    print("=" * 43)
    print(f"Test slices: {slice_count}")
    print(f"Test subjects: {len(subject_level)}")
    print(subject_level["true_class"].value_counts().to_string())
    print("\nAggregation comparison:")
    print(comparison.to_string(index=False))
    tie_count = int(subject_level["majority_vote_tie"].sum())
    print(f"\nMajority-vote ties: {tie_count}")
    print("All E08 metrics are subject-level metrics; no threshold optimization was performed.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    subject_level = aggregate_subject_predictions(predictions)
    subject_level.to_csv(RESULTS_DIR / "subject_level_predictions.csv", index=False)

    methods = {
        "mean_probability": {
            "prediction_column": "mean_predicted_label",
            "score_column": "mean_probability",
            "metrics_path": RESULTS_DIR / "mean_probability_metrics.json",
            "confusion_csv": RESULTS_DIR / "mean_probability_confusion_matrix.csv",
            "confusion_png": RESULTS_DIR / "mean_probability_confusion_matrix.png",
            "title": "CNN subject-level confusion matrix: mean probability",
        },
        "median_probability": {
            "prediction_column": "median_predicted_label",
            "score_column": "median_probability",
            "metrics_path": RESULTS_DIR / "median_probability_metrics.json",
            "confusion_csv": RESULTS_DIR / "median_probability_confusion_matrix.csv",
            "confusion_png": RESULTS_DIR / "median_probability_confusion_matrix.png",
            "title": "CNN subject-level confusion matrix: median probability",
        },
        "majority_vote": {
            "prediction_column": "majority_vote_predicted_label",
            "score_column": "positive_vote_fraction",
            "metrics_path": RESULTS_DIR / "majority_vote_metrics.json",
            "confusion_csv": RESULTS_DIR / "majority_vote_confusion_matrix.csv",
            "confusion_png": RESULTS_DIR / "majority_vote_confusion_matrix.png",
            "title": "CNN subject-level confusion matrix: majority vote",
        },
    }

    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method_name, config in methods.items():
        metrics, cm = calculate_metrics(
            subject_level,
            method_name=method_name,
            prediction_column=config["prediction_column"],
            score_column=config["score_column"],
        )
        metrics_by_method[method_name] = metrics
        write_json(config["metrics_path"], metrics)
        save_confusion_matrix(cm, config["confusion_csv"], config["confusion_png"], config["title"])

    comparison = build_comparison_table(metrics_by_method)
    comparison.to_csv(RESULTS_DIR / "aggregation_comparison.csv", index=False)

    summary = build_summary(subject_level, predictions, metrics_by_method, comparison)
    write_json(RESULTS_DIR / "e08_summary.json", summary)
    print_summary(comparison, subject_level, len(predictions))


if __name__ == "__main__":
    main()
