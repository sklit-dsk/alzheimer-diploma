from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from information_system.preprocessing import ImageValidationError, preprocess_mri_jpg_bytes


def _valid_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="Unsupported file extension"):
        preprocess_mri_jpg_bytes("slice.png", _valid_jpeg_bytes())


def test_empty_bytes_are_rejected() -> None:
    with pytest.raises(ImageValidationError, match="is empty"):
        preprocess_mri_jpg_bytes("slice.jpg", b"")


def test_corrupt_jpg_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        preprocess_mri_jpg_bytes("slice.jpg", b"not a decodable image")


def test_valid_jpg_produces_e07_tensor_shape() -> None:
    tensor = preprocess_mri_jpg_bytes("slice.jpg", _valid_jpeg_bytes())

    assert tuple(tensor.shape) == (3, 224, 224)
