"""E09 - Random Forest vs CNN subject-level comparison.

This experiment reconstructs the already-defined E03 Random Forest, verifies
that it reproduces the saved E03 image-level result, evaluates it on the full
frozen E03 test slice population, and compares it with the saved E08 CNN
subject-level predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


SEED = 42
NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
BINARY_CLASSES = [NEGATIVE_CLASS, POSITIVE_CLASS]
LABEL_MAP = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
DEFAULT_THRESHOLD = 0.5

EXPECTED_FEATURE_COUNT = 89
EXPECTED_TRAIN_ROWS = 21838
EXPECTED_TRAIN_SUBJECTS = 114
EXPECTED_TRAIN_CLASS_ROWS = {NEGATIVE_CLASS: 10919, POSITIVE_CLASS: 10919}
EXPECTED_TEST_BALANCED_ROWS = 5612
EXPECTED_TEST_BALANCED_SUBJECTS = 29
EXPECTED_TEST_BALANCED_CLASS_ROWS = {NEGATIVE_CLASS: 2806, POSITIVE_CLASS: 2806}
EXPECTED_FULL_ROWS = 34404
EXPECTED_FULL_SUBJECTS = 143
EXPECTED_FULL_CLASS_ROWS = {NEGATIVE_CLASS: 20679, POSITIVE_CLASS: 13725}
EXPECTED_FROZEN_TEST_SUBJECTS = 29
EXPECTED_FROZEN_TEST_CLASS_SUBJECTS = {NEGATIVE_CLASS: 17, POSITIVE_CLASS: 12}
EXPECTED_FULL_TEST_SLICES = 6954

EXPECTED_E03_REPRODUCTION = {
    "accuracy": 0.5787598004276551,
    "balanced_accuracy": 0.5787598004276551,
    "roc_auc": 0.6254816707300159,
    "confusion_matrix": [[1943, 863], [1501, 1305]],
}
REPRODUCTION_ATOL = 1e-12

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"

E03_ARTIFACTS_DIR = PROJECT_ROOT / "experiments" / "E03_clean_rf" / "artifacts"
E03_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E03_clean_rf" / "results"
E04_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E04_subject_level" / "results"
E08_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E08_cnn_subject_level" / "results"

BINARY_TRAIN_FEATURES_PATH = E03_ARTIFACTS_DIR / "binary_train_features.pkl"
BINARY_TEST_FEATURES_PATH = E03_ARTIFACTS_DIR / "binary_test_features.pkl"
FULL_FEATURES_PATH = E03_ARTIFACTS_DIR / "e02_full_features.pkl"
E03_TEST_SUBJECTS_PATH = E03_RESULTS_DIR / "test_subjects.csv"
E03_IMAGE_METRICS_PATH = E03_RESULTS_DIR / "image_level_metrics.json"
E04_SUMMARY_PATH = E04_RESULTS_DIR / "e04_summary.json"
E08_SUMMARY_PATH = E08_RESULTS_DIR / "e08_summary.json"
E08_SUBJECT_PREDICTIONS_PATH = E08_RESULTS_DIR / "subject_level_predictions.csv"

GLOBAL_FEATURES = [
    "global_mean",
    "global_std",
    "global_p10",
    "global_p50",
    "global_p90",
    "global_iqr",
]
INTENSITY_MASK_FEATURES = [
    "dark_ratio",
    "bright_ratio",
    "brain_area_ratio",
    "bbox_area_ratio",
    "centroid_x",
    "centroid_y",
]
EDGE_ENTROPY_FEATURES = ["edge_mean", "edge_p90", "entropy"]
REGIONAL_FEATURES = [
    f"roi_r{row}_c{col}_{stat}"
    for row in range(1, 5)
    for col in range(1, 9)
    for stat in ["mean", "std"]
]
TEXTURE_FEATURES = [f"lbp_uniform_bin_{idx}" for idx in range(10)]
FEATURE_COLUMNS = (
    GLOBAL_FEATURES
    + INTENSITY_MASK_FEATURES
    + EDGE_ENTROPY_FEATURES
    + REGIONAL_FEATURES
    + TEXTURE_FEATURES
)
METADATA_COLUMNS = ["path", "filename", "class", "subject_id", "scan_id"]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def ensure_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def validate_feature_table(
    frame: pd.DataFrame,
    *,
    source_name: str,
    expected_rows: int,
    expected_subjects: int,
    expected_class_rows: dict[str, int],
) -> None:
    if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Internal feature list has {len(FEATURE_COLUMNS)} features.")
    ensure_columns(frame, FEATURE_COLUMNS + METADATA_COLUMNS, source_name)

    if len(frame) != expected_rows:
        raise ValueError(f"{source_name} has {len(frame)} rows; expected {expected_rows}.")

    observed_subjects = int(frame["subject_id"].nunique())
    if observed_subjects != expected_subjects:
        raise ValueError(
            f"{source_name} has {observed_subjects} subjects; expected {expected_subjects}."
        )

    observed_class_rows = frame["class"].value_counts().reindex(BINARY_CLASSES).fillna(0).astype(int)
    observed = {class_name: int(observed_class_rows.loc[class_name]) for class_name in BINARY_CLASSES}
    if observed != expected_class_rows:
        raise ValueError(
            f"{source_name} class row counts differ. Expected {expected_class_rows}; observed {observed}."
        )

    unknown_classes = sorted(set(frame["class"]) - set(BINARY_CLASSES))
    if unknown_classes:
        raise ValueError(f"{source_name} contains unexpected classes: {unknown_classes}")


def prepare_feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features; observed {len(FEATURE_COLUMNS)}.")
    return feature_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)


def train_reconstructed_rf(train_features: pd.DataFrame) -> RandomForestClassifier:
    x_train = prepare_feature_matrix(train_features)
    y_train = train_features["class"].map(LABEL_MAP).to_numpy(dtype=int)
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf


def evaluate_slice_level_rf(
    clf: RandomForestClassifier, features: pd.DataFrame
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    x_test = prepare_feature_matrix(features)
    y_true = features["class"].map(LABEL_MAP).to_numpy(dtype=int)
    y_pred = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)[:, LABEL_MAP[POSITIVE_CLASS]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, cm, y_pred, probabilities


def compare_reproduction(reproduced: dict[str, Any], cm: np.ndarray) -> dict[str, Any]:
    expected_cm = np.array(EXPECTED_E03_REPRODUCTION["confusion_matrix"])
    metric_diffs = {}
    metric_pass = {}
    for metric in ["accuracy", "balanced_accuracy", "roc_auc"]:
        diff = abs(float(reproduced[metric]) - float(EXPECTED_E03_REPRODUCTION[metric]))
        metric_diffs[metric] = diff
        metric_pass[metric] = bool(
            np.isclose(
                reproduced[metric],
                EXPECTED_E03_REPRODUCTION[metric],
                atol=REPRODUCTION_ATOL,
                rtol=0.0,
            )
        )
    cm_equal = bool(np.array_equal(cm, expected_cm))
    passed = bool(all(metric_pass.values()) and cm_equal)
    return {
        "expected_values": EXPECTED_E03_REPRODUCTION,
        "reproduced_values": reproduced,
        "absolute_differences": metric_diffs,
        "metric_pass": metric_pass,
        "confusion_matrix_equality": cm_equal,
        "tolerance": REPRODUCTION_ATOL,
        "passed": passed,
    }


def validate_saved_e03_metrics() -> None:
    saved = read_json(E03_IMAGE_METRICS_PATH)
    for metric in ["accuracy", "balanced_accuracy", "roc_auc"]:
        if not np.isclose(
            float(saved[metric]),
            EXPECTED_E03_REPRODUCTION[metric],
            atol=REPRODUCTION_ATOL,
            rtol=0.0,
        ):
            raise ValueError(
                f"Saved E03 {metric} differs from expected value: "
                f"{saved[metric]} vs {EXPECTED_E03_REPRODUCTION[metric]}"
            )
    if saved["confusion_matrix"] != EXPECTED_E03_REPRODUCTION["confusion_matrix"]:
        raise ValueError(
            "Saved E03 confusion matrix differs from expected value: "
            f"{saved['confusion_matrix']}"
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = pd.read_pickle(BINARY_TRAIN_FEATURES_PATH)
    balanced_test_features = pd.read_pickle(BINARY_TEST_FEATURES_PATH)
    full_features = pd.read_pickle(FULL_FEATURES_PATH)
    test_subjects = pd.read_csv(E03_TEST_SUBJECTS_PATH)
    cnn_subjects = pd.read_csv(E08_SUBJECT_PREDICTIONS_PATH)

    validate_feature_table(
        train_features,
        source_name="binary_train_features.pkl",
        expected_rows=EXPECTED_TRAIN_ROWS,
        expected_subjects=EXPECTED_TRAIN_SUBJECTS,
        expected_class_rows=EXPECTED_TRAIN_CLASS_ROWS,
    )
    validate_feature_table(
        balanced_test_features,
        source_name="binary_test_features.pkl",
        expected_rows=EXPECTED_TEST_BALANCED_ROWS,
        expected_subjects=EXPECTED_TEST_BALANCED_SUBJECTS,
        expected_class_rows=EXPECTED_TEST_BALANCED_CLASS_ROWS,
    )
    validate_feature_table(
        full_features,
        source_name="e02_full_features.pkl",
        expected_rows=EXPECTED_FULL_ROWS,
        expected_subjects=EXPECTED_FULL_SUBJECTS,
        expected_class_rows=EXPECTED_FULL_CLASS_ROWS,
    )
    ensure_columns(
        test_subjects,
        ["subject_id", "class", "images_before_balancing", "images_after_balancing"],
        "E03 test_subjects.csv",
    )
    ensure_columns(
        cnn_subjects,
        ["subject_id", "true_class", "true_label", "mean_probability", "mean_predicted_label"],
        "E08 subject_level_predictions.csv",
    )
    return train_features, balanced_test_features, full_features, test_subjects, cnn_subjects


def validate_frozen_test_subject_table(test_subjects: pd.DataFrame) -> None:
    if test_subjects["subject_id"].duplicated().any():
        duplicates = test_subjects.loc[test_subjects["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"E03 test_subjects.csv contains duplicate subjects: {duplicates}")
    if len(test_subjects) != EXPECTED_FROZEN_TEST_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_FROZEN_TEST_SUBJECTS} E03 test subjects; observed {len(test_subjects)}."
        )
    counts = test_subjects["class"].value_counts().reindex(BINARY_CLASSES).fillna(0).astype(int)
    observed = {class_name: int(counts.loc[class_name]) for class_name in BINARY_CLASSES}
    if observed != EXPECTED_FROZEN_TEST_CLASS_SUBJECTS:
        raise ValueError(
            f"Unexpected E03 test subject counts. Expected {EXPECTED_FROZEN_TEST_CLASS_SUBJECTS}; "
            f"observed {observed}."
        )


def build_full_rf_test_set(full_features: pd.DataFrame, test_subjects: pd.DataFrame) -> pd.DataFrame:
    validate_frozen_test_subject_table(test_subjects)
    frozen_subject_ids = set(test_subjects["subject_id"])
    full_test = full_features[full_features["subject_id"].isin(frozen_subject_ids)].copy()

    if len(full_test) != EXPECTED_FULL_TEST_SLICES:
        raise ValueError(
            f"Full frozen RF test set has {len(full_test)} slices; expected {EXPECTED_FULL_TEST_SLICES}."
        )
    if int(full_test["subject_id"].nunique()) != EXPECTED_FROZEN_TEST_SUBJECTS:
        raise ValueError(
            f"Full frozen RF test set has {full_test['subject_id'].nunique()} subjects; "
            f"expected {EXPECTED_FROZEN_TEST_SUBJECTS}."
        )
    extra_subjects = sorted(set(full_test["subject_id"]) - frozen_subject_ids)
    if extra_subjects:
        raise ValueError(f"Full RF test set contains non-frozen subjects: {extra_subjects}")

    row_class_counts = full_test["class"].value_counts().reindex(BINARY_CLASSES).fillna(0).astype(int)
    if int(row_class_counts.sum()) != EXPECTED_FULL_TEST_SLICES:
        raise ValueError("Full RF test set class row counts do not sum to expected slice count.")

    subject_class_counts = (
        full_test.drop_duplicates("subject_id")["class"].value_counts().reindex(BINARY_CLASSES).fillna(0).astype(int)
    )
    observed_subject_counts = {
        class_name: int(subject_class_counts.loc[class_name]) for class_name in BINARY_CLASSES
    }
    if observed_subject_counts != EXPECTED_FROZEN_TEST_CLASS_SUBJECTS:
        raise ValueError(
            "Full RF test subject class counts differ. "
            f"Expected {EXPECTED_FROZEN_TEST_CLASS_SUBJECTS}; observed {observed_subject_counts}."
        )

    subject_class_nunique = full_test.groupby("subject_id")["class"].nunique()
    inconsistent = subject_class_nunique[subject_class_nunique != 1]
    if not inconsistent.empty:
        raise ValueError(f"Subjects with inconsistent class labels: {inconsistent.to_dict()}")

    return full_test.sort_values(["class", "subject_id", "scan_id", "filename"]).reset_index(drop=True)


def save_rf_full_test_slice_predictions(
    clf: RandomForestClassifier, full_test_features: pd.DataFrame
) -> pd.DataFrame:
    probabilities = clf.predict_proba(prepare_feature_matrix(full_test_features))[
        :, LABEL_MAP[POSITIVE_CLASS]
    ]
    predictions = full_test_features[["path", "subject_id", "scan_id", "class"]].copy()
    if "filename" in full_test_features.columns:
        predictions.insert(1, "filename", full_test_features["filename"].to_numpy())
    predictions = predictions.rename(columns={"class": "true_class"})
    predictions["true_label"] = predictions["true_class"].map(LABEL_MAP).astype(int)
    predictions["probability_very_mild_dementia"] = probabilities
    predictions["predicted_label_at_0_5"] = (
        predictions["probability_very_mild_dementia"] >= DEFAULT_THRESHOLD
    ).astype(int)
    predictions["predicted_class_at_0_5"] = predictions["predicted_label_at_0_5"].map(INVERSE_LABEL_MAP)
    predictions.to_csv(RESULTS_DIR / "rf_full_test_slice_predictions.csv", index=False)
    return predictions


def aggregate_rf_subjects(rf_slice_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, subject_rows in rf_slice_predictions.groupby("subject_id", sort=True):
        true_classes = subject_rows["true_class"].unique().tolist()
        true_labels = subject_rows["true_label"].unique().tolist()
        if len(true_classes) != 1 or len(true_labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent RF true class/label values.")

        mean_probability = float(subject_rows["probability_very_mild_dementia"].mean())
        predicted_label = int(mean_probability >= DEFAULT_THRESHOLD)
        rows.append(
            {
                "subject_id": subject_id,
                "true_class": true_classes[0],
                "true_label": int(true_labels[0]),
                "number_of_test_images": int(len(subject_rows)),
                "mean_probability": mean_probability,
                "predicted_label_at_0_5": predicted_label,
                "predicted_class": INVERSE_LABEL_MAP[predicted_label],
            }
        )

    subjects = pd.DataFrame(rows).sort_values(["true_class", "subject_id"]).reset_index(drop=True)
    subjects.to_csv(RESULTS_DIR / "rf_full_test_subject_predictions.csv", index=False)
    return subjects


def calculate_subject_metrics(
    subject_predictions: pd.DataFrame,
    *,
    model_name: str,
    score_column: str,
    prediction_column: str,
) -> tuple[dict[str, Any], np.ndarray]:
    y_true = subject_predictions["true_label"].to_numpy(dtype=int)
    y_pred = subject_predictions[prediction_column].to_numpy(dtype=int)
    scores = subject_predictions[score_column].to_numpy(dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return (
        {
            "model": model_name,
            "evaluation_level": "subject",
            "number_of_subjects": int(len(subject_predictions)),
            "positive_class": POSITIVE_CLASS,
            "threshold": DEFAULT_THRESHOLD,
            "score_column": score_column,
            "prediction_column": prediction_column,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": BINARY_CLASSES,
        },
        cm,
    )


def save_confusion_matrix(cm: np.ndarray, output_csv: Path, output_png: Path, title: str) -> None:
    pd.DataFrame(cm, index=BINARY_CLASSES, columns=BINARY_CLASSES).to_csv(output_csv)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(BINARY_CLASSES)), labels=BINARY_CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(BINARY_CLASSES)), labels=BINARY_CLASSES)
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


def validate_cnn_subjects(cnn_subjects: pd.DataFrame, rf_subjects: pd.DataFrame) -> pd.DataFrame:
    cnn = cnn_subjects[
        ["subject_id", "true_class", "true_label", "mean_probability", "mean_predicted_label"]
    ].copy()
    cnn["true_label"] = pd.to_numeric(cnn["true_label"], errors="raise").astype(int)
    cnn["mean_predicted_label"] = pd.to_numeric(
        cnn["mean_predicted_label"], errors="raise"
    ).astype(int)
    cnn["mean_probability"] = pd.to_numeric(cnn["mean_probability"], errors="raise")

    if len(cnn) != EXPECTED_FROZEN_TEST_SUBJECTS:
        raise ValueError(f"CNN subject table has {len(cnn)} rows; expected 29.")
    if cnn["subject_id"].duplicated().any():
        duplicates = cnn.loc[cnn["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"CNN subject table contains duplicate subjects: {duplicates}")

    rf_subject_ids = set(rf_subjects["subject_id"])
    cnn_subject_ids = set(cnn["subject_id"])
    if cnn_subject_ids != rf_subject_ids:
        raise ValueError(
            "CNN and RF subject sets differ. "
            f"Only CNN: {sorted(cnn_subject_ids - rf_subject_ids)}; "
            f"only RF: {sorted(rf_subject_ids - cnn_subject_ids)}"
        )

    merged = rf_subjects[["subject_id", "true_class", "true_label"]].merge(
        cnn[["subject_id", "true_class", "true_label"]],
        on="subject_id",
        how="inner",
        suffixes=("_rf", "_cnn"),
        validate="one_to_one",
    )
    disagree = merged[
        (merged["true_class_rf"] != merged["true_class_cnn"])
        | (merged["true_label_rf"] != merged["true_label_cnn"])
    ]
    if not disagree.empty:
        raise ValueError(
            "CNN and RF true labels/classes disagree for subjects: "
            f"{disagree.to_dict(orient='records')}"
        )

    if cnn["mean_probability"].isna().any() or not cnn["mean_probability"].between(0, 1).all():
        raise ValueError("CNN mean probabilities contain missing or out-of-range values.")
    return cnn


def build_paired_subject_predictions(
    rf_subjects: pd.DataFrame, cnn_subjects: pd.DataFrame
) -> pd.DataFrame:
    cnn = validate_cnn_subjects(cnn_subjects, rf_subjects)
    paired = rf_subjects[
        ["subject_id", "true_class", "true_label", "mean_probability", "predicted_label_at_0_5"]
    ].rename(
        columns={
            "mean_probability": "rf_mean_probability",
            "predicted_label_at_0_5": "rf_predicted_label",
        }
    )
    paired = paired.merge(
        cnn[["subject_id", "mean_probability", "mean_predicted_label"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "mean_probability": "cnn_mean_probability",
            "mean_predicted_label": "cnn_predicted_label",
        }
    )
    paired["rf_correct"] = paired["rf_predicted_label"] == paired["true_label"]
    paired["cnn_correct"] = paired["cnn_predicted_label"] == paired["true_label"]
    paired = paired.sort_values(["true_class", "subject_id"]).reset_index(drop=True)
    paired.to_csv(RESULTS_DIR / "paired_subject_predictions.csv", index=False)
    return paired


def build_model_comparison(rf_metrics: dict[str, Any], cnn_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in [("Random Forest", rf_metrics), ("ResNet18 CNN", cnn_metrics)]:
        rows.append(
            {
                "model": model_name,
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
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comparison


def metric_differences_cnn_minus_rf(comparison: pd.DataFrame) -> dict[str, float]:
    rf = comparison.set_index("model").loc["Random Forest"]
    cnn = comparison.set_index("model").loc["ResNet18 CNN"]
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1_positive",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]
    return {metric: float(cnn[metric] - rf[metric]) for metric in metrics}


def paired_correctness_summary(paired: pd.DataFrame) -> dict[str, Any]:
    both_correct = int((paired["rf_correct"] & paired["cnn_correct"]).sum())
    both_wrong = int((~paired["rf_correct"] & ~paired["cnn_correct"]).sum())
    rf_only_correct = int((paired["rf_correct"] & ~paired["cnn_correct"]).sum())
    cnn_only_correct = int((~paired["rf_correct"] & paired["cnn_correct"]).sum())

    mcnemar: dict[str, Any] = {
        "calculated": False,
        "test": "exact McNemar via scipy.stats.binomtest on discordant paired correctness counts",
        "p_value": None,
        "reason": None,
    }
    discordant = rf_only_correct + cnn_only_correct
    try:
        from scipy.stats import binomtest

        if discordant > 0:
            result = binomtest(min(rf_only_correct, cnn_only_correct), n=discordant, p=0.5)
            mcnemar.update(
                {
                    "calculated": True,
                    "p_value": float(result.pvalue),
                    "discordant_total": int(discordant),
                }
            )
        else:
            mcnemar.update(
                {
                    "calculated": True,
                    "p_value": 1.0,
                    "discordant_total": 0,
                    "reason": "No discordant paired correctness outcomes.",
                }
            )
    except ImportError:
        mcnemar["reason"] = "SciPy is not available in the current environment."

    summary = {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "rf_only_correct": rf_only_correct,
        "cnn_only_correct": cnn_only_correct,
        "discordant_counts": {
            "rf_correct_cnn_wrong": rf_only_correct,
            "rf_wrong_cnn_correct": cnn_only_correct,
        },
        "mcnemar_exact": mcnemar,
        "interpretation_limitation": (
            "This is a supplementary paired analysis on only 29 test subjects. "
            "A non-significant result does not prove equivalence, and a significant result "
            "would not establish clinical superiority."
        ),
    }
    write_json(RESULTS_DIR / "paired_correctness_summary.json", summary)
    return summary


def build_e04_vs_e09_rf_comparison(rf_full_metrics: dict[str, Any]) -> pd.DataFrame:
    e04_summary = read_json(E04_SUMMARY_PATH)
    e04_rf = e04_summary["metric_summaries"]["mean_probability"]
    rows = [
        {
            "evaluation": "Historical E04 RF",
            "slice_population": "5612 image-balanced test slices",
            "aggregation": "mean_probability",
            "threshold": DEFAULT_THRESHOLD,
            "accuracy": e04_rf["accuracy"],
            "balanced_accuracy": e04_rf["balanced_accuracy"],
            "sensitivity": e04_rf["sensitivity"],
            "specificity": e04_rf["specificity"],
            "precision": e04_rf["precision"],
            "f1_positive": e04_rf["f1_positive"],
            "f1_macro": e04_rf["f1_macro"],
            "roc_auc": e04_rf["roc_auc"],
            "pr_auc": e04_rf["pr_auc"],
        },
        {
            "evaluation": "E09 full-test RF",
            "slice_population": "6954 full frozen test slices",
            "aggregation": "mean_probability",
            "threshold": DEFAULT_THRESHOLD,
            "accuracy": rf_full_metrics["accuracy"],
            "balanced_accuracy": rf_full_metrics["balanced_accuracy"],
            "sensitivity": rf_full_metrics["sensitivity"],
            "specificity": rf_full_metrics["specificity"],
            "precision": rf_full_metrics["precision"],
            "f1_positive": rf_full_metrics["f1_positive"],
            "f1_macro": rf_full_metrics["f1_macro"],
            "roc_auc": rf_full_metrics["roc_auc"],
            "pr_auc": rf_full_metrics["pr_auc"],
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "e04_vs_e09_rf_comparison.csv", index=False)
    return comparison


def plot_metric_comparison(comparison: pd.DataFrame) -> None:
    selected_metrics = [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_positive",
        "roc_auc",
        "pr_auc",
    ]
    plot_frame = comparison.set_index("model")[selected_metrics].T
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    x = np.arange(len(selected_metrics))
    width = 0.36
    ax.bar(x - width / 2, plot_frame["Random Forest"], width, label="Random Forest", color="#2A6F97")
    ax.bar(x + width / 2, plot_frame["ResNet18 CNN"], width, label="ResNet18 CNN", color="#B23A48")
    ax.set_xticks(x, labels=[metric.replace("_", " ") for metric in selected_metrics], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("Subject-level RF vs CNN metrics", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_vs_cnn_subject_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_curves(paired: pd.DataFrame) -> None:
    y_true = paired["true_label"].to_numpy(dtype=int)
    curve_specs = [
        ("Random Forest", paired["rf_mean_probability"].to_numpy(dtype=float), "#2A6F97"),
        ("ResNet18 CNN", paired["cnn_mean_probability"].to_numpy(dtype=float), "#B23A48"),
    ]

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for model_name, scores, color in curve_specs:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Subject-level ROC curves", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_vs_cnn_roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    baseline = float((y_true == 1).mean())
    for model_name, scores, color in curve_specs:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})", color=color, linewidth=2)
    ax.axhline(baseline, color="0.6", linestyle="--", linewidth=1, label=f"Positive rate={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Subject-level precision-recall curves", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_vs_cnn_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def build_summary(
    *,
    reproduction_check: dict[str, Any],
    rf_metrics: dict[str, Any],
    cnn_metrics: dict[str, Any],
    comparison: pd.DataFrame,
    paired_summary: dict[str, Any],
    e04_vs_e09: pd.DataFrame,
) -> dict[str, Any]:
    e08_summary = read_json(E08_SUMMARY_PATH)
    return {
        "experiment_id": "E09",
        "objective": "Direct subject-level Random Forest vs ResNet18 CNN comparison",
        "cohort": "E02 Age >= 60 cohort",
        "frozen_test_subject_count": EXPECTED_FROZEN_TEST_SUBJECTS,
        "class_counts": EXPECTED_FROZEN_TEST_CLASS_SUBJECTS,
        "full_test_slice_count": EXPECTED_FULL_TEST_SLICES,
        "rf_reconstruction_reproduction_status": reproduction_check,
        "primary_comparison_definition": {
            "same_subjects": True,
            "same_full_test_slice_population": True,
            "evaluation_level": "subject",
            "aggregation": "mean_probability",
            "threshold": DEFAULT_THRESHOLD,
            "positive_class": POSITIVE_CLASS,
        },
        "rf_subject_metrics": rf_metrics,
        "cnn_subject_metrics": cnn_metrics,
        "cnn_source_summary_mean_probability_metrics": e08_summary["metric_summaries"][
            "mean_probability"
        ],
        "metric_differences_cnn_minus_rf": metric_differences_cnn_minus_rf(comparison),
        "paired_correctness_counts": paired_summary,
        "historical_e04_comparison_note": (
            "Historical E04 RF used 5612 image-balanced test slices. E09 reconstructs the "
            "E03 RF and evaluates it on all 6954 frozen test slices for a more symmetric "
            "comparison with E08 CNN."
        ),
        "historical_e04_vs_symmetric_e09_rf": e04_vs_e09.to_dict(orient="records"),
        "threshold": DEFAULT_THRESHOLD,
        "threshold_optimization_performed": False,
        "limitations": [
            "The frozen test set contains only 29 subjects.",
            "Metric differences are descriptive observations on this frozen test set.",
            "No threshold optimization, CNN retraining, RF hyperparameter tuning or new feature extraction was performed.",
            "The comparison does not establish clinical diagnostic validity or clinical superiority.",
        ],
    }


def print_summary(comparison: pd.DataFrame, paired_summary: dict[str, Any]) -> None:
    print("\nE09 - Random Forest vs CNN comparison")
    print("=" * 44)
    print("Primary comparison: subject-level, same 29 frozen test subjects, mean probability, threshold 0.5")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nPaired correctness:")
    print(json.dumps(to_plain_python(paired_summary), indent=2))
    print("\nE09 implemented comparison outputs. Do not treat results as clinical validation.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validate_saved_e03_metrics()

    (
        train_features,
        balanced_test_features,
        full_features,
        test_subjects,
        cnn_subjects,
    ) = load_inputs()

    print("Training reconstructed E03 Random Forest on saved binary_train_features.pkl", flush=True)
    clf = train_reconstructed_rf(train_features)

    print("Running mandatory E03 reproduction check on binary_test_features.pkl", flush=True)
    reproduced_metrics, reproduced_cm, _, _ = evaluate_slice_level_rf(clf, balanced_test_features)
    reproduction_check = compare_reproduction(reproduced_metrics, reproduced_cm)
    write_json(RESULTS_DIR / "e03_rf_reproduction_check.json", reproduction_check)
    if not reproduction_check["passed"]:
        raise RuntimeError(
            "Reconstructed RF did not reproduce E03 exactly enough. "
            "See results/e03_rf_reproduction_check.json."
        )

    print("Evaluating reconstructed RF on all frozen test slices", flush=True)
    full_test_features = build_full_rf_test_set(full_features, test_subjects)
    rf_slice_predictions = save_rf_full_test_slice_predictions(clf, full_test_features)
    rf_subjects = aggregate_rf_subjects(rf_slice_predictions)
    rf_metrics, rf_cm = calculate_subject_metrics(
        rf_subjects,
        model_name="Random Forest",
        score_column="mean_probability",
        prediction_column="predicted_label_at_0_5",
    )
    write_json(RESULTS_DIR / "rf_full_test_subject_metrics.json", rf_metrics)
    save_confusion_matrix(
        rf_cm,
        RESULTS_DIR / "rf_full_test_subject_confusion_matrix.csv",
        RESULTS_DIR / "rf_full_test_subject_confusion_matrix.png",
        "Full-test RF subject-level confusion matrix",
    )

    paired = build_paired_subject_predictions(rf_subjects, cnn_subjects)
    cnn_for_metrics = paired[
        ["subject_id", "true_class", "true_label", "cnn_mean_probability", "cnn_predicted_label"]
    ].rename(
        columns={
            "cnn_mean_probability": "mean_probability",
            "cnn_predicted_label": "mean_predicted_label",
        }
    )
    cnn_metrics, _ = calculate_subject_metrics(
        cnn_for_metrics,
        model_name="ResNet18 CNN",
        score_column="mean_probability",
        prediction_column="mean_predicted_label",
    )

    comparison = build_model_comparison(rf_metrics, cnn_metrics)
    paired_summary = paired_correctness_summary(paired)
    e04_vs_e09 = build_e04_vs_e09_rf_comparison(rf_metrics)
    plot_metric_comparison(comparison)
    plot_curves(paired)

    summary = build_summary(
        reproduction_check=reproduction_check,
        rf_metrics=rf_metrics,
        cnn_metrics=cnn_metrics,
        comparison=comparison,
        paired_summary=paired_summary,
        e04_vs_e09=e04_vs_e09,
    )
    write_json(RESULTS_DIR / "e09_summary.json", summary)
    print_summary(comparison, paired_summary)


if __name__ == "__main__":
    main()
