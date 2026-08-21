from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from information_system.config import MODEL_NAME
from information_system.model import ModelBundle


class FakeModel(nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([0.0, 1.0], dtype=batch.dtype, device=batch.device)
        return logits.repeat(batch.shape[0], 1)


def _valid_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from information_system import main

    fake_bundle = ModelBundle(
        model=FakeModel(),
        device=torch.device("cpu"),
        metadata={"model_name": MODEL_NAME},
    )
    monkeypatch.setattr(main, "load_e07_resnet18_checkpoint", lambda: fake_bundle)
    with TestClient(main.app) as test_client:
        yield test_client


def test_index_returns_upload_form_and_warning(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "MRI Research Classification Prototype" in response.text
    assert 'type="file"' in response.text
    assert "Research prototype only" in response.text
    assert "Does not diagnose Alzheimer" in response.text


def test_predict_without_files_returns_clean_error(client: TestClient) -> None:
    response = client.post("/predict")

    assert response.status_code == 400
    assert "No files were uploaded." in response.text
    assert "Back to upload form" in response.text
    assert "Traceback" not in response.text


def test_predict_with_invalid_extension_returns_clean_error(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files=[("files", ("slice.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 400
    assert "Only JPG/JPEG MRI slice images are accepted." in response.text
    assert "Traceback" not in response.text


def test_predict_with_corrupt_jpg_returns_clean_error(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files=[("files", ("slice.jpg", b"not a decodable image", "image/jpeg"))],
    )

    assert response.status_code == 400
    assert "One or more files could not be decoded as images." in response.text
    assert "Traceback" not in response.text


def test_predict_with_valid_image_returns_subject_result(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files=[("files", ("slice.jpg", _valid_jpeg_bytes(), "image/jpeg"))],
    )

    assert response.status_code == 200
    assert "Analyzed slice count" in response.text
    assert ">1<" in response.text
    assert "Mean model score for Very mild Dementia" in response.text
    assert "Decision threshold" in response.text
    assert "0.50" in response.text
    assert "Research classification" in response.text
    assert "Very mild Dementia" in response.text
    assert "not clinically validated" in response.text


def test_predict_inference_failure_returns_generic_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from information_system import main

    def fail_inference(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("internal checkpoint path and stack detail")

    monkeypatch.setattr(main, "predict_subject_from_tensors", fail_inference)
    response = client.post(
        "/predict",
        files=[("files", ("slice.jpg", _valid_jpeg_bytes(), "image/jpeg"))],
    )

    assert response.status_code == 500
    assert "Inference failed; check the local model setup." in response.text
    assert "internal checkpoint path and stack detail" not in response.text
    assert "Traceback" not in response.text
