"""Model construction and trusted local checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18

from .config import CHECKPOINT_PATH, LABEL_TO_CLASS, MODEL_NAME


class InformationSystemModelError(RuntimeError):
    """Raised when the local research model cannot be prepared for inference."""


class CheckpointNotFoundError(InformationSystemModelError):
    """Raised when the trusted local E07 checkpoint is missing."""


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    device: torch.device
    metadata: dict[str, Any]


def build_resnet18_model(device: torch.device) -> nn.Module:
    """Build the E07 ResNet18 architecture with a two-class final layer."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_TO_CLASS))
    return model.to(device)


def load_e07_resnet18_checkpoint(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> ModelBundle:
    """Load the trusted local E07 Attempt 2 checkpoint for inference.

    The E07 checkpoint was created by this project and contains optimizer/config
    metadata in addition to tensors. PyTorch 2.6+ defaults to weights_only=True,
    so this trusted local checkpoint must be loaded with weights_only=False.
    Do not use weights_only=False for arbitrary external checkpoints.
    """
    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(
            f"E07 Attempt 2 checkpoint was not found: {checkpoint_path}"
        )

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_model(resolved_device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise InformationSystemModelError(
            "E07 Attempt 2 checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    metadata = {
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint_path),
        "device": str(resolved_device),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "checkpoint_phase": checkpoint.get("phase"),
        "checkpoint_epoch_in_phase": checkpoint.get("epoch_in_phase"),
        "label_mapping": LABEL_TO_CLASS,
    }
    return ModelBundle(model=model, device=resolved_device, metadata=metadata)

