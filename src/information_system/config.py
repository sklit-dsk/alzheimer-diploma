"""Configuration for the information-system inference core."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "E07_cnn" / "artifacts" / "best_resnet18_attempt2.pt"
)

MODEL_NAME = "ResNet18 CNN, E07 Attempt 2"

NEGATIVE_CLASS = "Non Demented"
POSITIVE_CLASS = "Very mild Dementia"
LABEL_TO_CLASS = {
    0: NEGATIVE_CLASS,
    1: POSITIVE_CLASS,
}
CLASS_TO_LABEL = {class_name: label for label, class_name in LABEL_TO_CLASS.items()}

THRESHOLD = 0.5
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ACCEPTED_EXTENSIONS = {".jpg", ".jpeg"}

RESEARCH_WARNING = (
    "Research prototype only. This output is not clinically validated and does not "
    "diagnose Alzheimer's disease or replace medical assessment."
)

