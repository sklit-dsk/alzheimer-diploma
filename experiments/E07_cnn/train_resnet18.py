"""E07 - Transfer-learning CNN with ResNet18.

Attempt 2 uses staged transfer learning after an incomplete Attempt 1 showed
rapid overfitting / unstable validation behavior. The E03 frozen test subjects
are used only once for final slice-level inference after validation checkpoint
selection. Final CNN subject-level test evaluation belongs to E08.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode


NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
CLASS_ORDER = [NEGATIVE_CLASS, POSITIVE_CLASS]
LABEL_MAP = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INVERSE_LABEL_MAP = {0: NEGATIVE_CLASS, 1: POSITIVE_CLASS}

ATTEMPT_ID = "attempt2"
SEED = 42
INPUT_SIZE = 224
BATCH_SIZE = 16
VALIDATION_FRACTION = 0.20
PHASE_1_EPOCHS = 4
PHASE_1_LEARNING_RATE = 1e-3
PHASE_2_MAX_EPOCHS = 8
PHASE_2_LEARNING_RATE = 1e-5
PHASE_2_EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 1e-4
DEFAULT_THRESHOLD = 0.5
NUM_WORKERS = 4

EXPECTED_E03_TRAIN_COUNTS = {NEGATIVE_CLASS: 68, POSITIVE_CLASS: 46}
EXPECTED_E03_TEST_COUNTS = {NEGATIVE_CLASS: 17, POSITIVE_CLASS: 12}
EXPECTED_CNN_TRAIN_COUNTS = {NEGATIVE_CLASS: 54, POSITIVE_CLASS: 37}
EXPECTED_CNN_VALIDATION_COUNTS = {NEGATIVE_CLASS: 14, POSITIVE_CLASS: 9}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
E03_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E03_clean_rf" / "results"
E03_TRAIN_SUBJECTS_PATH = E03_RESULTS_DIR / "train_subjects.csv"
E03_TEST_SUBJECTS_PATH = E03_RESULTS_DIR / "test_subjects.csv"
E03_SELECTED_IMAGE_METADATA_PATH = E03_RESULTS_DIR / "selected_image_metadata.csv"
RAW_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "imagesoasis" / "Data"
RESULTS_DIR = SCRIPT_DIR / "results"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
VALIDATION_EPOCH_PREDICTIONS_DIR = RESULTS_DIR / "validation_epoch_predictions_attempt2"
BEST_CHECKPOINT_PATH = ARTIFACTS_DIR / "best_resnet18_attempt2.pt"
TRAINING_CONFIG_PATH = RESULTS_DIR / "training_config_attempt2.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history_attempt2.csv"
TRAIN_SUBJECTS_OUTPUT_PATH = RESULTS_DIR / "train_subjects_attempt2.csv"
VALIDATION_SUBJECTS_OUTPUT_PATH = RESULTS_DIR / "validation_subjects_attempt2.csv"
TEST_SUBJECTS_OUTPUT_PATH = RESULTS_DIR / "test_subjects_attempt2.csv"
SPLIT_SUMMARY_OUTPUT_PATH = RESULTS_DIR / "split_summary_attempt2.csv"
TRAIN_IMAGE_METADATA_OUTPUT_PATH = RESULTS_DIR / "train_image_metadata_attempt2.csv"
VALIDATION_IMAGE_METADATA_OUTPUT_PATH = RESULTS_DIR / "validation_image_metadata_attempt2.csv"
TEST_IMAGE_METADATA_OUTPUT_PATH = RESULTS_DIR / "test_image_metadata_attempt2.csv"
VALIDATION_SLICE_PREDICTIONS_OUTPUT_PATH = RESULTS_DIR / "validation_slice_predictions_attempt2.csv"
VALIDATION_SUBJECT_PREDICTIONS_OUTPUT_PATH = RESULTS_DIR / "validation_subject_predictions_attempt2.csv"
TEST_SLICE_PREDICTIONS_OUTPUT_PATH = RESULTS_DIR / "test_slice_predictions_attempt2.csv"
E07_SUMMARY_OUTPUT_PATH = RESULTS_DIR / "e07_summary_attempt2.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRETRAINED_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
PRETRAINED_WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str = "E07"
    attempt_id: str = ATTEMPT_ID
    attempt_strategy: str = "staged transfer learning"
    model_architecture: str = "torchvision.models.resnet18"
    pretrained_weights: str = PRETRAINED_WEIGHTS_ID
    seed: int = SEED
    input_size: int = INPUT_SIZE
    batch_size: int = BATCH_SIZE
    phase_1_epochs: int = PHASE_1_EPOCHS
    phase_1_trainable_layers: str = "fc only"
    phase_1_optimizer: str = "AdamW"
    phase_1_learning_rate: float = PHASE_1_LEARNING_RATE
    phase_1_weight_decay: float = WEIGHT_DECAY
    phase_2_max_epochs: int = PHASE_2_MAX_EPOCHS
    phase_2_trainable_layers: str = "layer4 and fc"
    phase_2_optimizer: str = "AdamW"
    phase_2_learning_rate: float = PHASE_2_LEARNING_RATE
    phase_2_weight_decay: float = WEIGHT_DECAY
    phase_2_early_stopping_patience: int = PHASE_2_EARLY_STOPPING_PATIENCE
    weight_decay: float = WEIGHT_DECAY
    loss: str = "CrossEntropyLoss with class weights from CNN train subject counts"
    validation_fraction: float = VALIDATION_FRACTION
    checkpoint_selection_metric: str = "validation_subject_roc_auc"
    checkpoint_tie_breaker: str = "validation_loss"
    default_threshold: float = DEFAULT_THRESHOLD
    frozen_test_policy: str = (
        "E03 test subjects are not used for training, validation, early stopping, "
        "model selection, hyperparameter tuning, augmentation decisions, or threshold selection."
    )
    reproducibility_note: str = (
        "Seeds and deterministic flags are configured where practical, with "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch import. Absolute bit-for-bit "
        "reproducibility is not promised across all CUDA environments."
    )
    image_root: str = "data/raw/imagesoasis/Data"
    train_augmentation: str = (
        "Resize 224x224; random affine rotation +/-5 degrees, translation 2%, "
        "scale 0.98-1.02; grayscale repeated to 3 channels; ImageNet normalization."
    )
    validation_test_transform: str = (
        "Resize 224x224; grayscale repeated to 3 channels; ImageNet normalization."
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def ensure_required_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_subject_table(
    subjects: pd.DataFrame, expected_counts: dict[str, int], source: str
) -> None:
    if subjects["subject_id"].duplicated().any():
        duplicates = subjects.loc[subjects["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"{source} has duplicate subjects: {duplicates}")
    counts = subjects.groupby("class")["subject_id"].nunique().to_dict()
    if counts != expected_counts:
        raise ValueError(f"{source} counts {counts} do not match expected {expected_counts}.")
    invalid_classes = sorted(set(subjects["class"]) - set(CLASS_ORDER))
    if invalid_classes:
        raise ValueError(f"{source} has unexpected classes: {invalid_classes}")


def load_subjects(path: Path, expected_counts: dict[str, int], expected_split: str) -> pd.DataFrame:
    subjects = pd.read_csv(path)
    ensure_required_columns(subjects, ["split", "subject_id", "class"], str(path))
    subjects = subjects[["split", "subject_id", "class"]].copy()
    observed_splits = set(subjects["split"])
    if observed_splits != {expected_split}:
        raise ValueError(f"{path} has split values {observed_splits}, expected {expected_split}.")
    validate_subject_table(subjects, expected_counts, f"E03 {expected_split} subjects")
    return subjects.sort_values(["class", "subject_id"]).reset_index(drop=True)


def split_train_validation_subjects(e03_train_subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for class_name in CLASS_ORDER:
        class_rows = e03_train_subjects[e03_train_subjects["class"] == class_name].copy()
        subject_ids = sorted(class_rows["subject_id"].tolist())
        rng.shuffle(subject_ids)
        validation_count = int(round(len(subject_ids) * VALIDATION_FRACTION))
        validation_ids = set(subject_ids[:validation_count])
        validation_parts.append(class_rows[class_rows["subject_id"].isin(validation_ids)])
        train_parts.append(class_rows[~class_rows["subject_id"].isin(validation_ids)])

    train_subjects = (
        pd.concat(train_parts, ignore_index=True)
        .assign(split="train")
        .sort_values(["class", "subject_id"])
        .reset_index(drop=True)
    )
    validation_subjects = (
        pd.concat(validation_parts, ignore_index=True)
        .assign(split="validation")
        .sort_values(["class", "subject_id"])
        .reset_index(drop=True)
    )
    validate_subject_table(train_subjects, EXPECTED_CNN_TRAIN_COUNTS, "CNN train subjects")
    validate_subject_table(
        validation_subjects, EXPECTED_CNN_VALIDATION_COUNTS, "CNN validation subjects"
    )
    return train_subjects, validation_subjects


def validate_no_overlap(*subject_tables: pd.DataFrame) -> None:
    names = ["train", "validation", "test"]
    sets = [set(table["subject_id"]) for table in subject_tables]
    overlaps = {}
    for i, left in enumerate(sets):
        for j, right in enumerate(sets):
            if i < j:
                overlap = sorted(left & right)
                if overlap:
                    overlaps[f"{names[i]}_vs_{names[j]}"] = overlap
    if overlaps:
        raise ValueError(f"Subject overlap detected: {overlaps}")


def load_selected_image_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(E03_SELECTED_IMAGE_METADATA_PATH)
    ensure_required_columns(
        metadata,
        ["path", "filename", "class", "subject_id", "scan_id"],
        "E03 selected_image_metadata.csv",
    )
    metadata = metadata.copy()
    metadata["true_label"] = metadata["class"].map(LABEL_MAP)
    if metadata["true_label"].isna().any():
        invalid = sorted(metadata.loc[metadata["true_label"].isna(), "class"].unique())
        raise ValueError(f"Unexpected image classes: {invalid}")
    metadata["true_label"] = metadata["true_label"].astype(int)
    metadata["path"] = metadata.apply(
        lambda row: str(RAW_IMAGE_ROOT / str(row["class"]) / str(row["filename"])),
        axis=1,
    )
    metadata = metadata.rename(columns={"class": "true_class"})
    return metadata


def attach_and_validate_splits(
    metadata: pd.DataFrame,
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
) -> pd.DataFrame:
    subjects = pd.concat(
        [
            train_subjects[["split", "subject_id", "class"]],
            validation_subjects[["split", "subject_id", "class"]],
            test_subjects[["split", "subject_id", "class"]],
        ],
        ignore_index=True,
    ).rename(columns={"class": "subject_class"})
    merged = metadata.merge(subjects, on="subject_id", how="left", validate="many_to_one")

    if merged["split"].isna().any():
        missing = sorted(merged.loc[merged["split"].isna(), "subject_id"].unique())
        raise ValueError(f"Images found for subjects outside the E07 split: {missing}")
    if (merged["true_class"] != merged["subject_class"]).any():
        sample = merged.loc[
            merged["true_class"] != merged["subject_class"],
            ["subject_id", "filename", "true_class", "subject_class"],
        ].head(20)
        raise ValueError(f"Class mismatch between images and subjects: {sample.to_dict('records')}")
    represented_test = set(merged.loc[merged["split"] == "test", "subject_id"])
    missing_test = sorted(set(test_subjects["subject_id"]) - represented_test)
    if missing_test:
        raise ValueError(f"Frozen test subjects missing from image metadata: {missing_test}")

    paths = merged["path"].map(Path)
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing raw image files, first 20: {missing_paths[:20]}")
    return merged


def image_metadata_for_split(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    columns = ["path", "filename", "subject_id", "scan_id", "true_class", "true_label", "split"]
    return (
        metadata[metadata["split"] == split][columns]
        .sort_values(["true_class", "subject_id", "scan_id", "filename"])
        .reset_index(drop=True)
    )


def save_split_outputs(
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    test_images: pd.DataFrame,
) -> pd.DataFrame:
    train_subjects.to_csv(TRAIN_SUBJECTS_OUTPUT_PATH, index=False)
    validation_subjects.to_csv(VALIDATION_SUBJECTS_OUTPUT_PATH, index=False)
    test_subjects.to_csv(TEST_SUBJECTS_OUTPUT_PATH, index=False)
    train_images.to_csv(TRAIN_IMAGE_METADATA_OUTPUT_PATH, index=False)
    validation_images.to_csv(VALIDATION_IMAGE_METADATA_OUTPUT_PATH, index=False)
    test_images.to_csv(TEST_IMAGE_METADATA_OUTPUT_PATH, index=False)

    rows = []
    for split, subject_frame, image_frame in [
        ("train", train_subjects, train_images),
        ("validation", validation_subjects, validation_images),
        ("test", test_subjects, test_images),
    ]:
        for class_name in CLASS_ORDER:
            rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "subjects": int((subject_frame["class"] == class_name).sum()),
                    "images": int((image_frame["true_class"] == class_name).sum()),
                }
            )
        rows.append(
            {
                "split": split,
                "class": "Total",
                "subjects": int(subject_frame["subject_id"].nunique()),
                "images": int(len(image_frame)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SPLIT_SUMMARY_OUTPUT_PATH, index=False)
    return summary


def prepare_e07_state(*, write_split_outputs: bool) -> dict[str, Any]:
    e03_train_subjects = load_subjects(E03_TRAIN_SUBJECTS_PATH, EXPECTED_E03_TRAIN_COUNTS, "train")
    test_subjects = load_subjects(E03_TEST_SUBJECTS_PATH, EXPECTED_E03_TEST_COUNTS, "test")
    validate_frozen_test_set(test_subjects)
    train_subjects, validation_subjects = split_train_validation_subjects(e03_train_subjects)
    validate_no_overlap(train_subjects, validation_subjects, test_subjects)

    metadata = load_selected_image_metadata()
    split_metadata = attach_and_validate_splits(
        metadata, train_subjects, validation_subjects, test_subjects
    )
    train_images = image_metadata_for_split(split_metadata, "train")
    validation_images = image_metadata_for_split(split_metadata, "validation")
    test_images = image_metadata_for_split(split_metadata, "test")
    split_summary = build_split_summary(
        train_subjects,
        validation_subjects,
        test_subjects,
        train_images,
        validation_images,
        test_images,
    )
    if write_split_outputs:
        write_split_outputs_to_disk(
            train_subjects,
            validation_subjects,
            test_subjects,
            train_images,
            validation_images,
            test_images,
            split_summary,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms()
    train_loader = build_loader(train_images, train_transform, shuffle=True, device=device)
    validation_loader = build_loader(validation_images, eval_transform, shuffle=False, device=device)
    test_loader = build_loader(test_images, eval_transform, shuffle=False, device=device)
    model = build_model(device)
    class_weights = class_weights_from_train_subjects(train_subjects, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    config = build_training_config_payload(train_subjects, class_weights, device)

    return {
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "train_images": train_images,
        "validation_images": validation_images,
        "test_images": test_images,
        "split_summary": split_summary,
        "device": device,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "model": model,
        "class_weights": class_weights,
        "criterion": criterion,
        "config": config,
    }


def build_split_summary(
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    test_images: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, subject_frame, image_frame in [
        ("train", train_subjects, train_images),
        ("validation", validation_subjects, validation_images),
        ("test", test_subjects, test_images),
    ]:
        for class_name in CLASS_ORDER:
            rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "subjects": int((subject_frame["class"] == class_name).sum()),
                    "images": int((image_frame["true_class"] == class_name).sum()),
                }
            )
        rows.append(
            {
                "split": split,
                "class": "Total",
                "subjects": int(subject_frame["subject_id"].nunique()),
                "images": int(len(image_frame)),
            }
        )
    return pd.DataFrame(rows)


def write_split_outputs_to_disk(
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    test_images: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    train_subjects.to_csv(TRAIN_SUBJECTS_OUTPUT_PATH, index=False)
    validation_subjects.to_csv(VALIDATION_SUBJECTS_OUTPUT_PATH, index=False)
    test_subjects.to_csv(TEST_SUBJECTS_OUTPUT_PATH, index=False)
    train_images.to_csv(TRAIN_IMAGE_METADATA_OUTPUT_PATH, index=False)
    validation_images.to_csv(VALIDATION_IMAGE_METADATA_OUTPUT_PATH, index=False)
    test_images.to_csv(TEST_IMAGE_METADATA_OUTPUT_PATH, index=False)
    split_summary.to_csv(SPLIT_SUMMARY_OUTPUT_PATH, index=False)


class MRISliceDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.metadata.iloc[index]
        image = Image.open(row["path"]).convert("L")
        label = int(row["true_label"])
        meta = {
            "path": row["path"],
            "filename": row["filename"],
            "subject_id": row["subject_id"],
            "scan_id": row["scan_id"],
            "true_class": row["true_class"],
            "true_label": label,
        }
        return self.transform(image), label, meta


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_loader(
    metadata: pd.DataFrame, transform: transforms.Compose, *, shuffle: bool, device: torch.device
) -> DataLoader:
    workers = NUM_WORKERS if len(metadata) >= BATCH_SIZE * 4 else 0
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        MRISliceDataset(metadata, transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if workers else None,
        generator=generator,
        persistent_workers=workers > 0,
    )


def build_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=PRETRAINED_WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    return model.to(device)


def set_trainable_layers(model: nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if phase == "phase_1_classifier_head":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if phase == "phase_2_layer4_fc":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unknown training phase: {phase}")


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters are enabled.")
    return parameters


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def class_weights_from_train_subjects(train_subjects: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_subjects.groupby("class")["subject_id"].nunique().to_dict()
    total = sum(counts.values())
    weights = [total / (len(CLASS_ORDER) * counts[class_name]) for class_name in CLASS_ORDER]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def hardware_info(device: torch.device) -> dict[str, Any]:
    import torchvision

    return {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mixed_precision_enabled": device.type == "cuda",
    }


def build_training_config_payload(
    train_subjects: pd.DataFrame, class_weights: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    config = asdict(TrainingConfig())
    config["class_label_mapping"] = LABEL_MAP
    config["class_weights_by_train_subject_counts"] = {
        class_name: float(class_weights[index].detach().cpu().item())
        for index, class_name in enumerate(CLASS_ORDER)
    }
    config["train_subject_counts_for_class_weights"] = {
        class_name: int((train_subjects["class"] == class_name).sum()) for class_name in CLASS_ORDER
    }
    config["hardware"] = hardware_info(device)
    return config


def save_training_config(
    train_subjects: pd.DataFrame, class_weights: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    config = build_training_config_payload(train_subjects, class_weights, device)
    write_json(TRAINING_CONFIG_PATH, config)
    return config


def meta_rows(
    meta: dict[str, Any], probabilities: np.ndarray, predicted_labels: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(predicted_labels)):
        label = int(meta["true_label"][index])
        predicted_label = int(predicted_labels[index])
        rows.append(
            {
                "path": str(meta["path"][index]),
                "filename": str(meta["filename"][index]),
                "subject_id": str(meta["subject_id"][index]),
                "scan_id": str(meta["scan_id"][index]),
                "true_class": str(meta["true_class"][index]),
                "true_label": label,
                "probability_very_mild_dementia": float(probabilities[index]),
                "predicted_label_at_0_5": predicted_label,
                "predicted_class_at_0_5": INVERSE_LABEL_MAP[predicted_label],
            }
        )
    return rows


def run_inference(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, labels, meta in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            predicted = (probabilities >= DEFAULT_THRESHOLD).astype(int)
            rows.extend(meta_rows(meta, probabilities, predicted))
            batch_size = int(inputs.size(0))
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Inference loader is empty.")
    return total_loss / total_examples, pd.DataFrame(rows)


def aggregate_validation_subjects(slice_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, subject_rows in slice_predictions.groupby("subject_id", sort=True):
        true_classes = subject_rows["true_class"].unique().tolist()
        true_labels = subject_rows["true_label"].unique().tolist()
        if len(true_classes) != 1 or len(true_labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent validation labels.")
        mean_probability = float(subject_rows["probability_very_mild_dementia"].mean())
        predicted_label = int(mean_probability >= DEFAULT_THRESHOLD)
        rows.append(
            {
                "subject_id": subject_id,
                "true_class": true_classes[0],
                "true_label": int(true_labels[0]),
                "number_of_validation_images": int(len(subject_rows)),
                "mean_probability": mean_probability,
                "predicted_label_at_0_5": predicted_label,
                "predicted_class_at_0_5": INVERSE_LABEL_MAP[predicted_label],
            }
        )
    return pd.DataFrame(rows).sort_values(["true_class", "subject_id"]).reset_index(drop=True)


def validation_metrics(subject_predictions: pd.DataFrame) -> dict[str, float]:
    y_true = subject_predictions["true_label"].to_numpy(dtype=int)
    scores = subject_predictions["mean_probability"].to_numpy(dtype=float)
    y_pred = subject_predictions["predicted_label_at_0_5"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    return {
        "validation_subject_roc_auc": float(roc_auc_score(y_true, scores)),
        "validation_subject_pr_auc": float(average_precision_score(y_true, scores)),
        "validation_subject_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_subject_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "validation_subject_sensitivity": float(sensitivity),
        "validation_subject_specificity": float(specificity),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, labels, _meta in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(inputs.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / total_examples


def should_replace_best(
    current_metrics: dict[str, float], validation_loss: float, best: dict[str, Any] | None
) -> bool:
    if best is None:
        return True
    current_auc = current_metrics["validation_subject_roc_auc"]
    best_auc = float(best["validation_subject_roc_auc"])
    if current_auc > best_auc and not np.isclose(current_auc, best_auc):
        return True
    return bool(np.isclose(current_auc, best_auc) and validation_loss < float(best["validation_loss"]))


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_record: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "global_epoch": int(epoch_record["global_epoch"]),
            "phase": epoch_record["phase"],
            "epoch_in_phase": int(epoch_record["epoch_in_phase"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": float(epoch_record["validation_loss"]),
            "validation_metrics": {
                key: epoch_record[key]
                for key in epoch_record
                if key.startswith("validation_subject_")
            },
            "config": config,
            "class_order": CLASS_ORDER,
            "label_map": LABEL_MAP,
        },
        BEST_CHECKPOINT_PATH,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history_rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    phases = [
        {
            "phase": "phase_1_classifier_head",
            "epochs": PHASE_1_EPOCHS,
            "learning_rate": PHASE_1_LEARNING_RATE,
            "early_stopping_patience": None,
        },
        {
            "phase": "phase_2_layer4_fc",
            "epochs": PHASE_2_MAX_EPOCHS,
            "learning_rate": PHASE_2_LEARNING_RATE,
            "early_stopping_patience": PHASE_2_EARLY_STOPPING_PATIENCE,
        },
    ]

    global_epoch = 0
    for phase_config in phases:
        phase = str(phase_config["phase"])
        set_trainable_layers(model, phase)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=float(phase_config["learning_rate"]),
            weight_decay=WEIGHT_DECAY,
        )
        epochs_without_improvement = 0

        print(
            f"\nStarting {phase}: trainable parameters={trainable_parameter_count(model)}",
            flush=True,
        )

        for epoch_in_phase in range(1, int(phase_config["epochs"]) + 1):
            global_epoch += 1
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp
            )
            validation_loss, validation_slice_predictions = run_inference(
                model, validation_loader, criterion, device
            )
            validation_slice_predictions.to_csv(
                VALIDATION_EPOCH_PREDICTIONS_DIR
                / f"attempt2_global_epoch_{global_epoch:03d}_slice_predictions.csv",
                index=False,
            )
            subject_predictions = aggregate_validation_subjects(validation_slice_predictions)
            metrics = validation_metrics(subject_predictions)
            epoch_record = {
                "phase": phase,
                "epoch_in_phase": epoch_in_phase,
                "global_epoch": global_epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "trainable_parameter_count": trainable_parameter_count(model),
                "epoch_seconds": time.time() - start_time,
            }
            history_rows.append(epoch_record)
            print(
                "Global epoch {global_epoch:02d} ({phase} {epoch_in_phase:02d}): "
                "train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, "
                "validation_subject_roc_auc={validation_subject_roc_auc:.4f}, "
                "validation_subject_pr_auc={validation_subject_pr_auc:.4f}".format(
                    **epoch_record
                ),
                flush=True,
            )
            if should_replace_best(metrics, validation_loss, best_record):
                best_record = dict(epoch_record)
                save_checkpoint(model, optimizer, epoch_record, config)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            patience = phase_config["early_stopping_patience"]
            if patience is not None and epochs_without_improvement >= int(patience):
                print(
                    f"Early stopping {phase} at global epoch {global_epoch}; "
                    f"best global epoch {best_record['global_epoch']}.",
                    flush=True,
                )
                break

    if best_record is None:
        raise RuntimeError("No validation checkpoint was selected.")
    history = pd.DataFrame(history_rows)
    history.to_csv(TRAINING_HISTORY_PATH, index=False)
    return history, best_record


def load_best_checkpoint(model: nn.Module, device: torch.device) -> dict[str, Any]:
    # This checkpoint is created locally by this project's own training code and
    # stores optimizer/config metadata in addition to tensors. PyTorch 2.6+
    # defaults to weights_only=True, so local metadata checkpoints must opt out.
    # Do not use weights_only=False for arbitrary external checkpoints.
    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def require_postprocess_inputs() -> None:
    missing = [
        str(path)
        for path in [BEST_CHECKPOINT_PATH, TRAINING_HISTORY_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Postprocess-only mode requires existing Attempt 2 training outputs: "
            f"{missing}"
        )


def read_training_history() -> pd.DataFrame:
    history = pd.read_csv(TRAINING_HISTORY_PATH)
    ensure_required_columns(
        history,
        [
            "phase",
            "epoch_in_phase",
            "global_epoch",
            "train_loss",
            "validation_loss",
            "validation_subject_roc_auc",
            "validation_subject_pr_auc",
            "validation_subject_accuracy",
            "validation_subject_balanced_accuracy",
            "validation_subject_sensitivity",
            "validation_subject_specificity",
            "learning_rate",
            "trainable_parameter_count",
        ],
        str(TRAINING_HISTORY_PATH),
    )
    history["global_epoch"] = pd.to_numeric(history["global_epoch"], errors="raise").astype(int)
    history["epoch_in_phase"] = pd.to_numeric(
        history["epoch_in_phase"], errors="raise"
    ).astype(int)
    return history


def validate_checkpoint_against_history(
    checkpoint: dict[str, Any], history: pd.DataFrame
) -> dict[str, Any]:
    required_checkpoint_keys = [
        "global_epoch",
        "phase",
        "epoch_in_phase",
        "validation_loss",
        "validation_metrics",
        "model_state_dict",
    ]
    missing = [key for key in required_checkpoint_keys if key not in checkpoint]
    if missing:
        raise ValueError(f"Attempt 2 checkpoint is missing keys: {missing}")

    checkpoint_global_epoch = int(checkpoint["global_epoch"])
    checkpoint_phase = str(checkpoint["phase"])
    checkpoint_rows = history[history["global_epoch"] == checkpoint_global_epoch]
    if len(checkpoint_rows) != 1:
        raise ValueError(
            "Checkpoint global epoch must occur exactly once in training history. "
            f"global_epoch={checkpoint_global_epoch}, matches={len(checkpoint_rows)}"
        )

    history_row = checkpoint_rows.iloc[0].to_dict()
    if str(history_row["phase"]) != checkpoint_phase:
        raise ValueError(
            "Checkpoint phase does not match training history. "
            f"checkpoint={checkpoint_phase}, history={history_row['phase']}"
        )

    checkpoint_roc_auc = float(
        checkpoint["validation_metrics"]["validation_subject_roc_auc"]
    )
    history_roc_auc = float(history_row["validation_subject_roc_auc"])
    if not np.isclose(checkpoint_roc_auc, history_roc_auc, atol=1e-9):
        raise ValueError(
            "Checkpoint validation ROC-AUC does not match training history. "
            f"checkpoint={checkpoint_roc_auc}, history={history_roc_auc}"
        )

    checkpoint_validation_loss = float(checkpoint["validation_loss"])
    history_validation_loss = float(history_row["validation_loss"])
    if not np.isclose(checkpoint_validation_loss, history_validation_loss, atol=1e-9):
        raise ValueError(
            "Checkpoint validation loss does not match training history. "
            f"checkpoint={checkpoint_validation_loss}, history={history_validation_loss}"
        )

    return {
        "phase": str(history_row["phase"]),
        "epoch_in_phase": int(history_row["epoch_in_phase"]),
        "global_epoch": int(history_row["global_epoch"]),
        "train_loss": float(history_row["train_loss"]),
        "validation_loss": float(history_row["validation_loss"]),
        "validation_subject_roc_auc": float(history_row["validation_subject_roc_auc"]),
        "validation_subject_pr_auc": float(history_row["validation_subject_pr_auc"]),
        "validation_subject_accuracy": float(history_row["validation_subject_accuracy"]),
        "validation_subject_balanced_accuracy": float(
            history_row["validation_subject_balanced_accuracy"]
        ),
        "validation_subject_sensitivity": float(
            history_row["validation_subject_sensitivity"]
        ),
        "validation_subject_specificity": float(
            history_row["validation_subject_specificity"]
        ),
        "learning_rate": float(history_row["learning_rate"]),
        "trainable_parameter_count": int(history_row["trainable_parameter_count"]),
    }


def validate_frozen_test_set(test_subjects: pd.DataFrame) -> None:
    counts = test_subjects.groupby("class")["subject_id"].nunique().to_dict()
    total = int(test_subjects["subject_id"].nunique())
    if total != 29 or counts != EXPECTED_E03_TEST_COUNTS:
        raise ValueError(
            "Frozen E03 test set changed. "
            f"Expected total=29 and {EXPECTED_E03_TEST_COUNTS}; observed total={total}, {counts}."
        )


def save_training_plots(history: pd.DataFrame) -> None:
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(history["global_epoch"], history["train_loss"], marker="o", label="Train")
    ax.plot(history["global_epoch"], history["validation_loss"], marker="o", label="Validation")
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Loss")
    ax.set_title("E07 Attempt 2 train and validation loss", loc="left", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_validation_loss_attempt2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(history["global_epoch"], history["validation_subject_roc_auc"], marker="o")
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Validation subject ROC-AUC")
    ax.set_title("E07 Attempt 2 validation subject ROC-AUC", loc="left", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_subject_roc_auc_by_epoch_attempt2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(
        history["global_epoch"],
        history["validation_subject_sensitivity"],
        marker="o",
        label="Sensitivity",
    )
    ax.plot(
        history["global_epoch"],
        history["validation_subject_specificity"],
        marker="s",
        label="Specificity",
    )
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Metric value")
    ax.set_title(
        "E07 Attempt 2 validation sensitivity and specificity",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "validation_subject_sensitivity_specificity_by_epoch_attempt2.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def build_summary(
    config: dict[str, Any],
    split_summary: pd.DataFrame,
    history: pd.DataFrame,
    best_record: dict[str, Any],
    validation_subject_predictions: pd.DataFrame,
    test_slice_predictions: pd.DataFrame,
) -> dict[str, Any]:
    total_rows = split_summary[split_summary["class"] == "Total"]
    image_counts = {
        row["split"]: int(row["images"]) for _, row in total_rows.iterrows()
    }
    subject_counts = {
        row["split"]: int(row["subjects"]) for _, row in total_rows.iterrows()
    }
    return {
        "experiment_id": "E07",
        "attempt_id": ATTEMPT_ID,
        "attempt_strategy": "staged transfer learning",
        "model_architecture": "ResNet18",
        "pretrained_weights": PRETRAINED_WEIGHTS_ID,
        "cohort": "E02 Age >= 60 cohort with frozen E03 test split",
        "train_validation_test_subject_counts": subject_counts,
        "image_counts": image_counts,
        "split_summary": split_summary.to_dict(orient="records"),
        "best_global_epoch": int(best_record["global_epoch"]),
        "best_phase": best_record["phase"],
        "best_epoch_in_phase": int(best_record["epoch_in_phase"]),
        "checkpoint_selection_metric": "validation_subject_roc_auc",
        "checkpoint_tie_breaker": "validation_loss",
        "best_validation_metrics": {
            key: best_record[key]
            for key in [
                "validation_loss",
                "validation_subject_roc_auc",
                "validation_subject_pr_auc",
                "validation_subject_accuracy",
                "validation_subject_balanced_accuracy",
                "validation_subject_sensitivity",
                "validation_subject_specificity",
            ]
        },
        "class_weights": config["class_weights_by_train_subject_counts"],
        "training_configuration": config,
        "hardware": config["hardware"],
        "global_epochs_completed": int(history["global_epoch"].max()),
        "phase_1_epochs_configured": PHASE_1_EPOCHS,
        "phase_2_max_epochs_configured": PHASE_2_MAX_EPOCHS,
        "phase_2_early_stopping_patience": PHASE_2_EARLY_STOPPING_PATIENCE,
        "test_inference_status": {
            "completed": True,
            "test_slice_predictions": int(len(test_slice_predictions)),
            "test_subjects": int(test_slice_predictions["subject_id"].nunique()),
            "note": "E07 saves slice-level CNN test predictions only.",
        },
        "validation_subject_predictions_saved": int(len(validation_subject_predictions)),
        "final_cnn_subject_level_test_evaluation": "Belongs to E08.",
        "interpretation_rules": [
            "Do not claim CNN superiority over RF in E07.",
            "Do not treat E07 test slice predictions as final subject-level CNN performance.",
            "The frozen E03 test subjects are not used for training decisions.",
            "This is a research classifier, not a clinically validated diagnostic system.",
        ],
    }


def run_postprocess_only() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    require_postprocess_inputs()
    set_reproducibility(SEED)

    history = read_training_history()
    state = prepare_e07_state(write_split_outputs=False)
    device = state["device"]
    model = state["model"]
    criterion = state["criterion"]
    validation_loader = state["validation_loader"]
    test_loader = state["test_loader"]

    checkpoint = load_best_checkpoint(model, device)
    best_record = validate_checkpoint_against_history(checkpoint, history)
    validate_frozen_test_set(state["test_subjects"])
    save_training_plots(history)

    print("\nE07 Attempt 2 postprocess-only mode")
    print("=" * 48)
    print("No training will be run.")
    print(
        "Using checkpoint metadata as authoritative: "
        f"global_epoch={best_record['global_epoch']}, phase={best_record['phase']}"
    )

    validation_loss, validation_slice_predictions = run_inference(
        model, validation_loader, criterion, device
    )
    validation_slice_predictions.to_csv(VALIDATION_SLICE_PREDICTIONS_OUTPUT_PATH, index=False)
    validation_subject_predictions = aggregate_validation_subjects(validation_slice_predictions)
    validation_subject_predictions.to_csv(VALIDATION_SUBJECT_PREDICTIONS_OUTPUT_PATH, index=False)

    _test_loss, test_slice_predictions = run_inference(model, test_loader, criterion, device)
    test_slice_predictions.to_csv(TEST_SLICE_PREDICTIONS_OUTPUT_PATH, index=False)

    summary = build_summary(
        state["config"],
        state["split_summary"],
        history,
        best_record,
        validation_subject_predictions,
        test_slice_predictions,
    )
    summary["best_checkpoint_path"] = str(BEST_CHECKPOINT_PATH.relative_to(PROJECT_ROOT))
    summary["checkpoint_global_epoch_in_file"] = int(checkpoint["global_epoch"])
    summary["checkpoint_phase_in_file"] = checkpoint["phase"]
    summary["checkpoint_epoch_in_phase_in_file"] = int(checkpoint["epoch_in_phase"])
    summary["validation_loss_from_best_checkpoint_inference"] = float(validation_loss)
    summary["postprocess_only"] = True
    summary["training_rerun"] = False
    summary["postprocess_recovery_note"] = (
        "Attempt 2 training had already completed. This mode loaded the existing local "
        "checkpoint with weights_only=False because it contains optimizer/config metadata, "
        "verified it against training_history_attempt2.csv, then generated validation and "
        "frozen-test slice predictions without retraining."
    )
    write_json(E07_SUMMARY_OUTPUT_PATH, summary)

    print("Validation inference succeeded.")
    print("Frozen test slice inference completed once and was saved for E08.")
    print("No final CNN subject-level test metrics are calculated in E07.")


def run_training() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_EPOCH_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducibility(SEED)

    state = prepare_e07_state(write_split_outputs=True)
    train_subjects = state["train_subjects"]
    train_images = state["train_images"]
    validation_images = state["validation_images"]
    test_images = state["test_images"]
    split_summary = state["split_summary"]
    device = state["device"]
    train_loader = state["train_loader"]
    validation_loader = state["validation_loader"]
    test_loader = state["test_loader"]
    model = state["model"]
    class_weights = state["class_weights"]
    criterion = state["criterion"]
    config = save_training_config(train_subjects, class_weights, device)

    print("\nE07 - Transfer-learning CNN using ResNet18")
    print("=" * 48)
    print("Attempt 2: staged transfer learning")
    print(f"Device: {device}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(validation_images)}")
    print(f"Frozen test images: {len(test_images)}")
    print("Frozen test subjects are not used during epoch-level training decisions.")

    history, best_record = train_model(
        model, train_loader, validation_loader, criterion, device, config
    )
    save_training_plots(history)

    checkpoint = load_best_checkpoint(model, device)
    validation_loss, validation_slice_predictions = run_inference(
        model, validation_loader, criterion, device
    )
    validation_slice_predictions.to_csv(VALIDATION_SLICE_PREDICTIONS_OUTPUT_PATH, index=False)
    validation_subject_predictions = aggregate_validation_subjects(validation_slice_predictions)
    validation_subject_predictions.to_csv(VALIDATION_SUBJECT_PREDICTIONS_OUTPUT_PATH, index=False)

    _test_loss, test_slice_predictions = run_inference(model, test_loader, criterion, device)
    test_slice_predictions.to_csv(TEST_SLICE_PREDICTIONS_OUTPUT_PATH, index=False)

    summary = build_summary(
        config,
        split_summary,
        history,
        best_record,
        validation_subject_predictions,
        test_slice_predictions,
    )
    summary["best_checkpoint_path"] = str(BEST_CHECKPOINT_PATH.relative_to(PROJECT_ROOT))
    summary["checkpoint_global_epoch_in_file"] = int(checkpoint["global_epoch"])
    summary["checkpoint_phase_in_file"] = checkpoint["phase"]
    summary["checkpoint_epoch_in_phase_in_file"] = int(checkpoint["epoch_in_phase"])
    summary["validation_loss_from_best_checkpoint_inference"] = float(validation_loss)
    write_json(E07_SUMMARY_OUTPUT_PATH, summary)

    print("\nE07 Attempt 2 training complete.")
    print(f"Best global epoch: {best_record['global_epoch']}")
    print(f"Best validation subject ROC-AUC: {best_record['validation_subject_roc_auc']:.4f}")
    print("Frozen test slice predictions saved for E08 subject-level aggregation.")
    print("No final CNN subject-level test metrics are calculated in E07.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E07 ResNet18 training and postprocessing.")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Do not retrain. Load the existing Attempt 2 checkpoint/history, verify them, "
            "then generate validation and frozen-test slice predictions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.postprocess_only:
        run_postprocess_only()
    else:
        run_training()


if __name__ == "__main__":
    main()
