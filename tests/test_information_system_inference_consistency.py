from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from information_system.inference import predict_slices_from_tensors
from information_system.model import load_e07_resnet18_checkpoint
from information_system.preprocessing import preprocess_mri_jpg_bytes


E07_TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "E07_cnn"
    / "results"
    / "test_slice_predictions_attempt2.csv"
)
RAW_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "imagesoasis" / "Data"


def test_first_batch_probabilities_match_saved_e07_predictions() -> None:
    predictions = pd.read_csv(E07_TEST_PREDICTIONS_PATH)
    required_columns = {
        "filename",
        "true_class",
        "probability_very_mild_dementia",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        pytest.fail(f"E07 test predictions are missing columns: {sorted(missing)}")

    first_batch = predictions.head(16)
    if len(first_batch) != 16:
        pytest.fail(f"Expected at least 16 saved E07 test predictions, got {len(first_batch)}.")

    filenames: list[str] = []
    tensors = []
    expected_probabilities: list[float] = []
    for row in first_batch.itertuples(index=False):
        filename = str(row.filename)
        raw_path = RAW_IMAGE_ROOT / str(row.true_class) / filename
        if not raw_path.exists():
            pytest.fail(f"Raw E07 test image is missing: {raw_path}")

        filenames.append(filename)
        expected_probabilities.append(float(row.probability_very_mild_dementia))
        tensors.append(preprocess_mri_jpg_bytes(filename, raw_path.read_bytes()))

    model_bundle = load_e07_resnet18_checkpoint()
    # E07 test inference used batch size 16; reproducing the same batch shape
    # avoids CUDA batch-shape numerical differences in this regression test.
    slice_predictions = predict_slices_from_tensors(
        model_bundle.model,
        model_bundle.device,
        filenames,
        tensors,
    )

    assert len(slice_predictions) == 16
    assert [prediction.filename for prediction in slice_predictions] == filenames
    for prediction, expected_probability in zip(
        slice_predictions,
        expected_probabilities,
        strict=True,
    ):
        observed_probability = prediction.probability_very_mild_dementia
        assert 0.0 <= observed_probability <= 1.0
        assert observed_probability == pytest.approx(expected_probability, abs=1e-10)
