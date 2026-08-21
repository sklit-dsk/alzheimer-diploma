"""E03 - Clean Random Forest baseline on the E02 Age >= 60 cohort.

This experiment intentionally reproduces the E00 handcrafted-feature Random
Forest baseline as closely as possible while using the E02 cohort as the
authoritative subject/scan definition.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.filters import sobel, threshold_otsu
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


SEED = 42
TRAIN_FRACTION = 0.8
CLASS_ORDER = ["Non Demented", "Very mild Dementia"]
BINARY_CLASSES = ["Non Demented", "Very mild Dementia"]
LABEL_MAP = {"Non Demented": 0, "Very mild Dementia": 1}
INVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
POSITIVE_CLASS = "Very mild Dementia"

EXPECTED_E02_SUBJECT_COUNTS = {"Non Demented": 85, "Very mild Dementia": 58}
EXPECTED_E02_TOTAL_SUBJECTS = 143
EXPECTED_E02_SCANS = 143
EXPECTED_FEATURE_COUNT = 89

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "imagesoasis" / "Data"
E00_METADATA_PATH = PROJECT_ROOT / "experiments" / "E00_baseline_rf" / "dataset_metadata.csv"
E02_SUBJECT_COHORT_PATH = (
    PROJECT_ROOT / "experiments" / "E02_age_control" / "results" / "e02_subject_level_cohort.csv"
)
E02_SCAN_COHORT_PATH = (
    PROJECT_ROOT / "experiments" / "E02_age_control" / "results" / "e02_scan_level_cohort.csv"
)
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
RESULTS_DIR = SCRIPT_DIR / "results"

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


def ensure_required_columns(
    frame: pd.DataFrame, required_columns: list[str], source_name: str
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def parse_subject_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else Path(filename).stem


def parse_scan_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else parse_subject_id(filename)


def relative_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_e02_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_cohort = pd.read_csv(E02_SUBJECT_COHORT_PATH)
    scan_cohort = pd.read_csv(E02_SCAN_COHORT_PATH)

    ensure_required_columns(
        subject_cohort,
        ["subject_id", "class", "Age", "Sex", "CDR", "MMSE", "number_of_scans", "number_of_images"],
        "E02 subject-level cohort",
    )
    ensure_required_columns(
        scan_cohort,
        ["scan_id", "subject_id", "class", "number_of_images"],
        "E02 scan-level cohort",
    )

    validate_e02_cohort(subject_cohort, scan_cohort)
    return subject_cohort, scan_cohort


def validate_e02_cohort(subject_cohort: pd.DataFrame, scan_cohort: pd.DataFrame) -> None:
    if subject_cohort["subject_id"].duplicated().any():
        duplicates = subject_cohort.loc[
            subject_cohort["subject_id"].duplicated(), "subject_id"
        ].tolist()
        raise ValueError(f"E02 subject-level cohort contains duplicate subject_id rows: {duplicates}")

    subject_cohort["Age"] = pd.to_numeric(subject_cohort["Age"], errors="coerce")
    if subject_cohort["Age"].isna().any():
        missing = subject_cohort.loc[subject_cohort["Age"].isna(), "subject_id"].tolist()
        raise ValueError(f"E02 cohort contains missing Age values: {missing}")

    under_60 = subject_cohort.loc[subject_cohort["Age"] < 60, "subject_id"].tolist()
    if under_60:
        raise ValueError(f"E02 cohort contains subjects younger than 60: {under_60}")

    subject_counts = subject_cohort.groupby("class")["subject_id"].nunique().to_dict()
    if (
        subject_counts != EXPECTED_E02_SUBJECT_COUNTS
        or subject_cohort["subject_id"].nunique() != EXPECTED_E02_TOTAL_SUBJECTS
    ):
        raise ValueError(
            "Unexpected E02 subject counts. "
            f"Expected {EXPECTED_E02_SUBJECT_COUNTS} and total {EXPECTED_E02_TOTAL_SUBJECTS}; "
            f"observed {subject_counts} and total {subject_cohort['subject_id'].nunique()}."
        )

    if scan_cohort["scan_id"].nunique() != EXPECTED_E02_SCANS:
        raise ValueError(
            f"Expected {EXPECTED_E02_SCANS} E02 scans; observed {scan_cohort['scan_id'].nunique()}."
        )

    subject_ids = set(subject_cohort["subject_id"])
    scan_subject_ids = set(scan_cohort["subject_id"])
    excluded_in_scans = sorted(scan_subject_ids - subject_ids)
    if excluded_in_scans:
        raise ValueError(f"E02 scan cohort contains subjects not in subject cohort: {excluded_in_scans}")

    subjects_without_scans = sorted(subject_ids - scan_subject_ids)
    if subjects_without_scans:
        raise ValueError(f"E02 subjects without scans: {subjects_without_scans}")


def load_e00_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(E00_METADATA_PATH)
    ensure_required_columns(
        metadata,
        ["path", "class", "subject_id", "scan_id", "filename"],
        "E00 dataset metadata",
    )
    return metadata


def reconstruct_e02_image_metadata(
    e00_metadata: pd.DataFrame,
    subject_cohort: pd.DataFrame,
    scan_cohort: pd.DataFrame,
) -> pd.DataFrame:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw ImagesOASIS root was not found: {relative_display_path(RAW_ROOT)}"
        )

    expected_scans = set(scan_cohort["scan_id"])
    expected_subjects = set(subject_cohort["subject_id"])
    selected = e00_metadata[e00_metadata["scan_id"].isin(expected_scans)].copy()

    if selected.empty:
        raise ValueError("No E00 metadata rows matched E02 scan IDs.")

    selected["parsed_subject_id"] = selected["filename"].map(parse_subject_id)
    selected["parsed_scan_id"] = selected["filename"].map(parse_scan_id)

    inconsistent_filename_rows = selected[
        (selected["parsed_subject_id"] != selected["subject_id"])
        | (selected["parsed_scan_id"] != selected["scan_id"])
    ]
    if not inconsistent_filename_rows.empty:
        raise ValueError(
            "Image filenames do not map consistently to E00 subject_id/scan_id: "
            f"{inconsistent_filename_rows[['filename', 'subject_id', 'scan_id', 'parsed_subject_id', 'parsed_scan_id']].head(20).to_dict(orient='records')}"
        )

    observed_scans = set(selected["scan_id"])
    missing_scans = sorted(expected_scans - observed_scans)
    extra_scans = sorted(observed_scans - expected_scans)
    if missing_scans or extra_scans:
        raise ValueError(
            f"Could not reconstruct E02 scans exactly. Missing: {missing_scans}; extra: {extra_scans}"
        )

    excluded_subjects = sorted(set(selected["subject_id"]) - expected_subjects)
    if excluded_subjects:
        raise ValueError(f"Selected image metadata contains excluded subjects: {excluded_subjects}")

    subject_classes = subject_cohort[["subject_id", "class"]].rename(
        columns={"class": "e02_subject_class"}
    )
    selected = selected.merge(subject_classes, on="subject_id", how="left", validate="many_to_one")
    class_mismatch = selected[selected["class"] != selected["e02_subject_class"]]
    if not class_mismatch.empty:
        raise ValueError(
            "E00 image class does not match E02 subject class: "
            f"{class_mismatch[['filename', 'class', 'e02_subject_class']].head(20).to_dict(orient='records')}"
        )
    selected = selected.drop(columns=["e02_subject_class", "parsed_subject_id", "parsed_scan_id"])

    selected["raw_path"] = selected.apply(
        lambda row: RAW_ROOT / str(row["class"]) / str(row["filename"]), axis=1
    )
    missing_files = selected.loc[~selected["raw_path"].map(lambda path: Path(path).exists())]
    if not missing_files.empty:
        raise FileNotFoundError(
            "Some selected E02 image files are missing under the local raw root. "
            f"Examples: {missing_files[['class', 'filename', 'raw_path']].head(20).to_dict(orient='records')}"
        )

    selected["path"] = selected["raw_path"].map(lambda path: str(Path(path)))
    selected = selected.drop(columns=["raw_path"])

    scan_counts = selected.groupby("scan_id").size().rename("selected_images").reset_index()
    expected_scan_counts = scan_cohort[["scan_id", "number_of_images"]].copy()
    scan_check = expected_scan_counts.merge(scan_counts, on="scan_id", how="left")
    scan_check["selected_images"] = scan_check["selected_images"].fillna(0).astype(int)
    mismatched_counts = scan_check[scan_check["number_of_images"] != scan_check["selected_images"]]
    if not mismatched_counts.empty:
        raise ValueError(
            "Selected image counts do not match E02 scan-level number_of_images: "
            f"{mismatched_counts.head(20).to_dict(orient='records')}"
        )

    selected = selected.sort_values(["class", "subject_id", "scan_id", "filename"]).reset_index(drop=True)
    return selected[["path", "filename", "class", "subject_id", "scan_id"]]


def stratified_group_split(df: pd.DataFrame, classes: list[str], train_frac: float = TRAIN_FRACTION):
    train_subjects = set()
    test_subjects = set()
    rng = random.Random(SEED)
    for class_name in classes:
        subjects = sorted(df.loc[df["class"] == class_name, "subject_id"].unique())
        rng.shuffle(subjects)
        n_train = max(1, int(round(len(subjects) * train_frac)))
        if len(subjects) > 1:
            n_train = min(n_train, len(subjects) - 1)
        train_subjects.update(subjects[:n_train])
        test_subjects.update(subjects[n_train:])
    train = df[df["subject_id"].isin(train_subjects)].copy()
    test = df[df["subject_id"].isin(test_subjects)].copy()
    return train, test


def balanced_sample_by_class(df: pd.DataFrame, classes: list[str], max_per_class=None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    counts = df["class"].value_counts()
    target = min(int(counts.loc[class_name]) for class_name in classes)
    if max_per_class is not None:
        target = min(target, max_per_class)
    parts = []
    for class_name in classes:
        idx = df.index[df["class"] == class_name].to_numpy()
        chosen = rng.choice(idx, size=target, replace=False)
        parts.append(df.loc[chosen])
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


def load_grayscale(path: str, size=(128, 64)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def safe_otsu_mask(arr: np.ndarray) -> np.ndarray:
    try:
        positive = arr[arr > 0.01]
        thr = threshold_otsu(positive) if positive.size > 50 else 0.05
        mask = arr > max(0.03, float(thr) * 0.35)
    except Exception:
        mask = arr > 0.05
    if mask.mean() < 0.03:
        mask = arr > 0.03
    return mask


def extract_features(path: str) -> dict[str, float]:
    arr = load_grayscale(path)
    h, w = arr.shape
    mask = safe_otsu_mask(arr)
    vals = arr[mask]
    if vals.size < 10:
        vals = arr.reshape(-1)

    features: dict[str, float] = {}
    features["global_mean"] = float(vals.mean())
    features["global_std"] = float(vals.std())
    features["global_p10"] = float(np.percentile(vals, 10))
    features["global_p50"] = float(np.percentile(vals, 50))
    features["global_p90"] = float(np.percentile(vals, 90))
    features["global_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    features["dark_ratio"] = float((vals < 0.15).mean())
    features["bright_ratio"] = float((vals > 0.75).mean())
    features["brain_area_ratio"] = float(mask.mean())

    ys, xs = np.where(mask)
    if len(xs):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        features["bbox_area_ratio"] = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / (h * w))
        features["centroid_x"] = float(xs.mean() / w)
        features["centroid_y"] = float(ys.mean() / h)
    else:
        features["bbox_area_ratio"] = 0.0
        features["centroid_x"] = 0.5
        features["centroid_y"] = 0.5

    edges = sobel(arr)
    edge_vals = edges[mask] if mask.any() else edges.reshape(-1)
    features["edge_mean"] = float(edge_vals.mean())
    features["edge_p90"] = float(np.percentile(edge_vals, 90))
    features["entropy"] = float(shannon_entropy((arr * 255).astype(np.uint8)))

    grid_rows, grid_cols = 4, 8
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(round(row * h / grid_rows))
            y1 = int(round((row + 1) * h / grid_rows))
            x0 = int(round(col * w / grid_cols))
            x1 = int(round((col + 1) * w / grid_cols))
            cell = arr[y0:y1, x0:x1]
            cell_mask = mask[y0:y1, x0:x1]
            cell_vals = cell[cell_mask] if cell_mask.sum() > 5 else cell.reshape(-1)
            features[f"roi_r{row + 1}_c{col + 1}_mean"] = float(cell_vals.mean())
            features[f"roi_r{row + 1}_c{col + 1}_std"] = float(cell_vals.std())

    lbp = local_binary_pattern((arr * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_vals = lbp[mask] if mask.any() else lbp.reshape(-1)
    hist, _ = np.histogram(lbp_vals, bins=np.arange(0, 11), density=True)
    hist = np.nan_to_num(hist)
    for idx, value in enumerate(hist):
        features[f"lbp_uniform_bin_{idx}"] = float(value)

    if list(features.keys()) != FEATURE_COLUMNS:
        raise ValueError(
            "Extracted feature order/count differs from E00 definition. "
            f"Expected {len(FEATURE_COLUMNS)}; observed {len(features)}."
        )
    return features


def extract_feature_matrix(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        validate_feature_table(cached, expected_rows=len(df), source_name=str(cache_path))
        return cached

    rows = []
    total = len(df)
    for idx, row in enumerate(df.to_dict("records"), start=1):
        features = extract_features(row["path"])
        features.update(
            {
                "path": row["path"],
                "filename": row["filename"],
                "class": row["class"],
                "subject_id": row["subject_id"],
                "scan_id": row["scan_id"],
            }
        )
        rows.append(features)
        if idx == 1 or idx % 1000 == 0 or idx == total:
            print(f"features {idx}/{total}", flush=True)

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df[FEATURE_COLUMNS + METADATA_COLUMNS]
    validate_feature_table(feature_df, expected_rows=total, source_name="newly extracted E03 features")
    feature_df.to_pickle(cache_path)
    return feature_df


def validate_feature_table(frame: pd.DataFrame, expected_rows: int, source_name: str) -> None:
    missing_features = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    missing_metadata = [column for column in METADATA_COLUMNS if column not in frame.columns]
    if missing_features or missing_metadata:
        raise ValueError(
            f"{source_name} is missing features {missing_features} or metadata {missing_metadata}."
        )
    if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Internal feature list does not contain {EXPECTED_FEATURE_COUNT} features.")
    if len(frame) != expected_rows:
        raise ValueError(f"{source_name} has {len(frame)} rows; expected {expected_rows}.")


def prepare_feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [column for column in FEATURE_COLUMNS if column in feature_df.columns]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} model features; observed {len(feature_cols)}.")
    X = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def validate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    train_subjects = set(train_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected between train and test: {overlap}")

    all_subjects = set(all_df["subject_id"])
    assigned_subjects = train_subjects | test_subjects
    missing_subjects = sorted(all_subjects - assigned_subjects)
    extra_subjects = sorted(assigned_subjects - all_subjects)
    if missing_subjects or extra_subjects:
        raise ValueError(
            f"Split does not assign each subject exactly once. Missing={missing_subjects}; extra={extra_subjects}"
        )

    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        classes = set(split_df["class"])
        missing_classes = sorted(set(BINARY_CLASSES) - classes)
        if missing_classes:
            raise ValueError(f"{split_name} split is missing classes: {missing_classes}")


def split_subject_table(split_df: pd.DataFrame, balanced_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    before = (
        split_df.groupby(["subject_id", "class"])
        .size()
        .rename("images_before_balancing")
        .reset_index()
    )
    after = (
        balanced_df.groupby("subject_id")
        .size()
        .rename("images_after_balancing")
        .reset_index()
    )
    table = before.merge(after, on="subject_id", how="left")
    table["images_after_balancing"] = table["images_after_balancing"].fillna(0).astype(int)
    table["represented_after_balancing"] = table["images_after_balancing"] > 0
    table.insert(0, "split", split_name)
    return table.sort_values(["class", "subject_id"])


def create_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_balanced: pd.DataFrame,
    test_balanced: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split_name, before_df, after_df in [
        ("train", train_df, train_balanced),
        ("test", test_df, test_balanced),
    ]:
        for class_name in BINARY_CLASSES:
            before_class = before_df[before_df["class"] == class_name]
            after_class = after_df[after_df["class"] == class_name]
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "subjects_before_balancing": int(before_class["subject_id"].nunique()),
                    "images_before_balancing": int(len(before_class)),
                    "subjects_after_balancing": int(after_class["subject_id"].nunique()),
                    "images_after_balancing": int(len(after_class)),
                }
            )
    return pd.DataFrame(rows)


def feature_category(feature: str) -> str:
    if feature.startswith("roi_"):
        return "Regional intensity / contrast"
    if feature.startswith("lbp_"):
        return "Texture: Local Binary Pattern"
    if feature.startswith("edge_"):
        return "Edges / structural boundaries"
    if feature in {"brain_area_ratio", "bbox_area_ratio", "centroid_x", "centroid_y"}:
        return "Shape / brain mask geometry"
    if feature in {"entropy", "dark_ratio", "bright_ratio"}:
        return "Intensity distribution"
    if feature.startswith("global_"):
        return "Global intensity / contrast"
    return "Other"


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(BINARY_CLASSES)), labels=BINARY_CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(BINARY_CLASSES)), labels=BINARY_CLASSES)
    ax.set_title("Image-level confusion matrix: E03 Random Forest", loc="left", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    top = importance_df.head(20).iloc[::-1]
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    ax.barh(
        top["feature"],
        top["permutation_importance_mean"],
        xerr=top["permutation_importance_std"],
        color="#20808D",
    )
    ax.set_title("Top 20 features by image-level permutation importance", loc="left", fontweight="bold")
    ax.set_xlabel("Mean decrease in balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(report: dict[str, Any], output_path: Path) -> None:
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"label": label}
            row.update(metrics)
            rows.append(row)
        else:
            rows.append({"label": label, "value": metrics})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def train_random_forest(train_features: pd.DataFrame) -> RandomForestClassifier:
    X_train = prepare_feature_matrix(train_features)
    y_train = train_features["class"].map(LABEL_MAP).to_numpy()

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_image_level(
    clf: RandomForestClassifier, test_features: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    X_test = prepare_feature_matrix(test_features)
    y_test = test_features["class"].map(LABEL_MAP).to_numpy()
    predicted_labels = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)[:, LABEL_MAP[POSITIVE_CLASS]]
    cm = confusion_matrix(y_test, predicted_labels, labels=[0, 1])
    report = classification_report(
        y_test,
        predicted_labels,
        labels=[0, 1],
        target_names=BINARY_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment_id": "E03",
        "task": "Non Demented vs Very mild Dementia",
        "evaluation_level": "image/slice",
        "important_limitation": "These are image/slice-level metrics and are not the final diploma evaluation.",
        "positive_class": POSITIVE_CLASS,
        "decision_threshold_note": "Class predictions use RandomForestClassifier.predict, matching E00 default behavior; no threshold optimization is performed in E03.",
        "split": "subject-level split stratified by class, followed by E00-style image-level undersampling",
        "seed": SEED,
        "train_size": None,
        "test_size": int(len(test_features)),
        "test_class_counts": {
            str(key): int(value) for key, value in test_features["class"].value_counts().items()
        },
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_labels)),
        "f1_macro": float(f1_score(y_test, predicted_labels, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    predictions = test_features[["path", "filename", "subject_id", "scan_id", "class"]].copy()
    predictions = predictions.rename(columns={"class": "true_class"})
    predictions["true_label"] = y_test
    predictions["predicted_label"] = predicted_labels
    predictions["predicted_class"] = predictions["predicted_label"].map(INVERSE_LABEL_MAP)
    predictions["probability_very_mild_dementia"] = probabilities
    predictions = predictions[
        [
            "path",
            "filename",
            "subject_id",
            "scan_id",
            "true_class",
            "true_label",
            "predicted_class",
            "predicted_label",
            "probability_very_mild_dementia",
        ]
    ]
    return metrics, predictions, cm


def save_feature_importance(
    clf: RandomForestClassifier, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test = prepare_feature_matrix(test_features)
    y_test = test_features["class"].map(LABEL_MAP).to_numpy()
    perm = permutation_importance(
        clf,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=8,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "rf_impurity_importance": clf.feature_importances_,
        }
    )
    importance_df["category"] = importance_df["feature"].map(feature_category)
    importance_df = importance_df.sort_values(
        "permutation_importance_mean", ascending=False
    ).reset_index(drop=True)

    legacy_category = (
        importance_df.assign(
            positive=lambda frame: frame["permutation_importance_mean"].clip(lower=0)
        )
        .groupby("category", as_index=False)["positive"]
        .sum()
        .rename(columns={"positive": "legacy_unadjusted_positive_permutation_importance"})
        .sort_values("legacy_unadjusted_positive_permutation_importance", ascending=False)
    )
    legacy_category["interpretation_note"] = (
        "Legacy unadjusted summed category importance for E00 comparability only; "
        "corrected grouped Feature Importance belongs to E06."
    )
    return importance_df, legacy_category


def build_validation_summary(
    subject_cohort: pd.DataFrame,
    scan_cohort: pd.DataFrame,
    selected_images: pd.DataFrame,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    train_balanced: pd.DataFrame,
    test_balanced: pd.DataFrame,
    train_subject_table: pd.DataFrame,
    test_subject_table: pd.DataFrame,
) -> dict[str, Any]:
    train_subjects = set(train_meta["subject_id"])
    test_subjects = set(test_meta["subject_id"])
    selected_counts = selected_images["class"].value_counts().reindex(BINARY_CLASSES).fillna(0).astype(int)

    eliminated = pd.concat([train_subject_table, test_subject_table], ignore_index=True)
    eliminated = eliminated[~eliminated["represented_after_balancing"]]

    validation = {
        "experiment_id": "E03",
        "E02_subject_count": int(subject_cohort["subject_id"].nunique()),
        "E02_scan_count": int(scan_cohort["scan_id"].nunique()),
        "selected_image_count": int(len(selected_images)),
        "selected_images_by_class": {
            class_name: int(selected_counts.loc[class_name]) for class_name in BINARY_CLASSES
        },
        "number_of_features": len(FEATURE_COLUMNS),
        "train_test_subject_overlap": sorted(train_subjects & test_subjects),
        "train_test_subject_overlap_count": int(len(train_subjects & test_subjects)),
        "train_test_subject_counts_by_class": split_subject_counts(train_meta, test_meta),
        "image_counts_before_balancing": split_image_counts(train_meta, test_meta),
        "image_counts_after_balancing": split_image_counts(train_balanced, test_balanced),
        "subjects_represented_after_balancing": {
            "train": int(train_balanced["subject_id"].nunique()),
            "test": int(test_balanced["subject_id"].nunique()),
            "by_split_and_class": split_subject_counts(train_balanced, test_balanced),
        },
        "subjects_eliminated_by_image_level_undersampling": eliminated[
            ["split", "class", "subject_id", "images_before_balancing", "images_after_balancing"]
        ].to_dict(orient="records"),
        "random_seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "rf_hyperparameters": {
            "n_estimators": 350,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": SEED,
            "n_jobs": -1,
        },
        "local_raw_data_root_used": relative_display_path(RAW_ROOT),
        "image_level_metric_warning": "E03 metrics are image/slice-level only. Final subject-level evaluation belongs to E04.",
    }
    return to_plain_python(validation)


def split_subject_counts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        counts = split_df.groupby("class")["subject_id"].nunique().reindex(BINARY_CLASSES).fillna(0)
        result[split_name] = {class_name: int(counts.loc[class_name]) for class_name in BINARY_CLASSES}
    return result


def split_image_counts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        counts = split_df["class"].value_counts().reindex(BINARY_CLASSES).fillna(0)
        result[split_name] = {class_name: int(counts.loc[class_name]) for class_name in BINARY_CLASSES}
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("E03 - loading E02 cohort and E00 metadata", flush=True)
    subject_cohort, scan_cohort = load_e02_cohort()
    e00_metadata = load_e00_metadata()
    selected_images = reconstruct_e02_image_metadata(e00_metadata, subject_cohort, scan_cohort)
    selected_images.to_csv(RESULTS_DIR / "selected_image_metadata.csv", index=False)

    print(
        "Selected E02 images: "
        f"{len(selected_images):,} "
        f"({selected_images['class'].value_counts().to_dict()})",
        flush=True,
    )

    print("Extracting or loading full E02 feature cache", flush=True)
    full_features = extract_feature_matrix(selected_images, ARTIFACTS_DIR / "e02_full_features.pkl")
    validate_feature_table(full_features, expected_rows=len(selected_images), source_name="full E02 features")

    train_meta, test_meta = stratified_group_split(selected_images, BINARY_CLASSES, train_frac=TRAIN_FRACTION)
    validate_split(train_meta, test_meta, selected_images)

    train_paths = set(train_meta["path"])
    test_paths = set(test_meta["path"])
    train_features_unbalanced = full_features[full_features["path"].isin(train_paths)].copy()
    test_features_unbalanced = full_features[full_features["path"].isin(test_paths)].copy()

    print("Applying E00-style image-level undersampling after subject split", flush=True)
    train_features = balanced_sample_by_class(train_features_unbalanced, BINARY_CLASSES)
    test_features = balanced_sample_by_class(test_features_unbalanced, BINARY_CLASSES)
    train_features.to_pickle(ARTIFACTS_DIR / "binary_train_features.pkl")
    test_features.to_pickle(ARTIFACTS_DIR / "binary_test_features.pkl")

    train_subject_table = split_subject_table(train_meta, train_features, "train")
    test_subject_table = split_subject_table(test_meta, test_features, "test")
    train_subject_table.to_csv(RESULTS_DIR / "train_subjects.csv", index=False)
    test_subject_table.to_csv(RESULTS_DIR / "test_subjects.csv", index=False)

    eliminated = pd.concat([train_subject_table, test_subject_table], ignore_index=True)
    eliminated = eliminated[~eliminated["represented_after_balancing"]]
    eliminated.to_csv(RESULTS_DIR / "subjects_eliminated_by_balancing.csv", index=False)
    if not eliminated.empty:
        print(
            "Warning: some subjects lost all images during image-level undersampling. "
            "See results/subjects_eliminated_by_balancing.csv",
            flush=True,
        )

    split_summary = create_split_summary(train_meta, test_meta, train_features, test_features)
    split_summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

    print("Training E00-compatible Random Forest", flush=True)
    clf = train_random_forest(train_features)

    metrics, predictions, cm = evaluate_image_level(clf, test_features)
    metrics["train_size"] = int(len(train_features))
    metrics["train_class_counts"] = {
        str(key): int(value) for key, value in train_features["class"].value_counts().items()
    }
    write_json(RESULTS_DIR / "image_level_metrics.json", metrics)
    predictions.to_csv(RESULTS_DIR / "image_level_predictions.csv", index=False)
    save_classification_report(
        metrics["classification_report"], RESULTS_DIR / "image_level_classification_report.csv"
    )
    pd.DataFrame(cm, index=BINARY_CLASSES, columns=BINARY_CLASSES).to_csv(
        RESULTS_DIR / "image_level_confusion_matrix.csv"
    )
    plot_confusion_matrix(cm, RESULTS_DIR / "image_level_confusion_matrix.png")

    print("Computing E00-compatible feature importance", flush=True)
    importance_df, legacy_category = save_feature_importance(clf, test_features)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    legacy_category.to_csv(RESULTS_DIR / "legacy_unadjusted_category_importance.csv", index=False)
    plot_top_importance(importance_df, RESULTS_DIR / "top20_permutation_importance.png")

    validation = build_validation_summary(
        subject_cohort=subject_cohort,
        scan_cohort=scan_cohort,
        selected_images=selected_images,
        train_meta=train_meta,
        test_meta=test_meta,
        train_balanced=train_features,
        test_balanced=test_features,
        train_subject_table=train_subject_table,
        test_subject_table=test_subject_table,
    )
    write_json(RESULTS_DIR / "validation_summary.json", validation)

    print("\nE03 image-level Random Forest baseline complete")
    print("=" * 50)
    print(f"Selected images: {len(selected_images):,}")
    print(f"Train images after balancing: {len(train_features):,}")
    print(f"Test images after balancing: {len(test_features):,}")
    print(f"Image-level accuracy: {metrics['accuracy']:.4f}")
    print(f"Image-level balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Image-level macro F1: {metrics['f1_macro']:.4f}")
    print(f"Image-level ROC-AUC: {metrics['roc_auc']:.4f}")
    print("These are image/slice-level metrics only; subject-level aggregation belongs to E04.")


if __name__ == "__main__":
    main()
