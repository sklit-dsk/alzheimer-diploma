"""Input validation and E07-compatible MRI slice preprocessing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import ACCEPTED_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be used for inference."""


def validate_filename_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        accepted = ", ".join(sorted(ACCEPTED_EXTENSIONS))
        raise ImageValidationError(
            f"Unsupported file extension for '{filename}'. Accepted extensions: {accepted}."
        )


def build_eval_transform() -> transforms.Compose:
    """Return the deterministic E07 validation/test preprocessing transform."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_mri_jpg_bytes(
    filename: str,
    image_bytes: bytes,
    transform: transforms.Compose | None = None,
) -> torch.Tensor:
    """Validate and preprocess one uploaded JPG/JPEG MRI slice.

    The output tensor shape is [3, 224, 224], matching E07 evaluation.
    Uploaded bytes are processed in memory and are not saved to disk.
    """
    validate_filename_extension(filename)
    if not image_bytes:
        raise ImageValidationError(f"File '{filename}' is empty.")

    eval_transform = transform or build_eval_transform()
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            grayscale_image = image.convert("L")
            tensor = eval_transform(grayscale_image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(
            f"File '{filename}' could not be decoded as a JPG/JPEG image."
        ) from exc

    if tuple(tensor.shape) != (3, INPUT_SIZE, INPUT_SIZE):
        raise ImageValidationError(
            f"File '{filename}' produced tensor shape {tuple(tensor.shape)}, "
            f"expected (3, {INPUT_SIZE}, {INPUT_SIZE})."
        )
    return tensor

