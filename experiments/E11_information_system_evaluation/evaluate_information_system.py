"""E11 - Final information-system evaluation.

This script evaluates the local research information-system prototype. It does
not retrain models, optimize thresholds, or modify historical experiment
outputs.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn


EXPERIMENT_ID = "E11"
SUBJECT_ID = "OAS1_0016"
CONSISTENCY_BATCH_SIZE = 16
CONSISTENCY_TOLERANCE = 1e-10

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from information_system.config import (  # noqa: E402
    ACCEPTED_EXTENSIONS,
    CHECKPOINT_PATH,
    INPUT_SIZE,
    MODEL_NAME,
    THRESHOLD,
)
from information_system.inference import (  # noqa: E402
    aggregate_subject_mean_probability,
    predict_slices_from_tensors,
)
from information_system.model import ModelBundle, load_e07_resnet18_checkpoint  # noqa: E402
from information_system.preprocessing import preprocess_mri_jpg_bytes  # noqa: E402


E07_TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT / "experiments" / "E07_cnn" / "results" / "test_slice_predictions_attempt2.csv"
)
E08_SUBJECT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "E08_cnn_subject_level"
    / "results"
    / "subject_level_predictions.csv"
)
RAW_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "imagesoasis" / "Data"


class FakeModel(nn.Module):
    """Small deterministic model for route-level checks."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([0.0, 1.0], dtype=batch.dtype, device=batch.device)
        return logits.repeat(batch.shape[0], 1)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_raw_image_path(row: pd.Series) -> Path:
    saved_path = Path(str(row["path"]))
    if saved_path.exists():
        return saved_path

    fallback_path = RAW_IMAGE_ROOT / str(row["true_class"]) / str(row["filename"])
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"Raw image was not found for {row['filename']}.")


def run_pytest_suite() -> dict[str, Any]:
    start_time = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execution_time = time.perf_counter() - start_time

    raw_output = "\n".join(
        part for part in [completed.stdout, completed.stderr] if part
    )
    (RESULTS_DIR / "pytest_output.txt").write_text(raw_output, encoding="utf-8")

    summary_line = raw_output.strip().splitlines()[-1] if raw_output.strip() else ""
    parsed: dict[str, int | None] = {
        "total_tests_collected": None,
        "passed": None,
        "failed": None,
        "errors": None,
    }
    collected_match = re.search(r"collected\s+(\d+)\s+items?", raw_output)
    if collected_match:
        parsed["total_tests_collected"] = int(collected_match.group(1))

    for key, pattern in {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "errors": r"(\d+)\s+errors?",
    }.items():
        match = re.search(pattern, summary_line)
        if match:
            parsed[key] = int(match.group(1))

    result = {
        **parsed,
        "return_code": completed.returncode,
        "execution_time_seconds": execution_time,
        "overall_passed": completed.returncode == 0,
        "parsing_note": (
            "Detailed counts were parsed from pytest output where available; "
            "raw stdout/stderr are saved for verification."
        ),
    }
    write_json(RESULTS_DIR / "test_suite_summary.json", result)
    return result


def evaluate_e07_inference_consistency(model_bundle: ModelBundle) -> dict[str, Any]:
    predictions = pd.read_csv(E07_TEST_PREDICTIONS_PATH).head(CONSISTENCY_BATCH_SIZE)
    if len(predictions) != CONSISTENCY_BATCH_SIZE:
        raise ValueError("Expected at least 16 E07 test predictions.")

    filenames: list[str] = []
    tensors = []
    expected_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []

    for row in predictions.to_dict(orient="records"):
        raw_path = resolve_raw_image_path(pd.Series(row))
        filename = str(row["filename"])
        filenames.append(filename)
        expected_probabilities.append(float(row["probability_very_mild_dementia"]))
        tensors.append(preprocess_mri_jpg_bytes(filename, raw_path.read_bytes()))

    slice_predictions = predict_slices_from_tensors(
        model_bundle.model,
        model_bundle.device,
        filenames,
        tensors,
    )

    absolute_differences: list[float] = []
    for expected, observed in zip(
        expected_probabilities,
        slice_predictions,
        strict=True,
    ):
        difference = abs(observed.probability_very_mild_dementia - expected)
        absolute_differences.append(difference)
        rows.append(
            {
                "filename": observed.filename,
                "saved_e07_probability": expected,
                "system_probability": observed.probability_very_mild_dementia,
                "absolute_difference": difference,
                "passed": difference <= CONSISTENCY_TOLERANCE,
            }
        )

    write_csv(
        RESULTS_DIR / "e07_inference_consistency.csv",
        rows,
        [
            "filename",
            "saved_e07_probability",
            "system_probability",
            "absolute_difference",
            "passed",
        ],
    )

    summary = {
        "number_of_slices_checked": len(rows),
        "tolerance": CONSISTENCY_TOLERANCE,
        "max_absolute_difference": max(absolute_differences),
        "mean_absolute_difference": sum(absolute_differences) / len(absolute_differences),
        "passed": all(row["passed"] for row in rows),
    }
    write_json(RESULTS_DIR / "e07_inference_consistency_summary.json", summary)
    return summary


def evaluate_subject_end_to_end(model_bundle: ModelBundle) -> dict[str, Any]:
    e08_predictions = pd.read_csv(E08_SUBJECT_PREDICTIONS_PATH)
    subject_rows = e08_predictions[e08_predictions["subject_id"] == SUBJECT_ID]
    if len(subject_rows) != 1:
        raise ValueError(f"Expected exactly one E08 row for subject {SUBJECT_ID}.")
    expected_subject = subject_rows.iloc[0]

    e07_predictions = pd.read_csv(E07_TEST_PREDICTIONS_PATH)
    if len(e07_predictions) != 6954:
        raise ValueError(f"Expected 6,954 E07 frozen test rows, got {len(e07_predictions)}.")

    full_sequence_predictions = []
    for start in range(0, len(e07_predictions), CONSISTENCY_BATCH_SIZE):
        batch = e07_predictions.iloc[start : start + CONSISTENCY_BATCH_SIZE]
        filenames: list[str] = []
        tensors = []
        for row in batch.to_dict(orient="records"):
            raw_path = resolve_raw_image_path(pd.Series(row))
            filename = str(row["filename"])
            filenames.append(filename)
            tensors.append(preprocess_mri_jpg_bytes(filename, raw_path.read_bytes()))

        full_sequence_predictions.extend(
            predict_slices_from_tensors(
                model_bundle.model,
                model_bundle.device,
                filenames,
                tensors,
            )
        )

    if len(full_sequence_predictions) != len(e07_predictions):
        raise ValueError(
            "Full-test prediction count does not match the saved E07 test sequence."
        )

    observed_filenames = [prediction.filename for prediction in full_sequence_predictions]
    saved_filenames = e07_predictions["filename"].astype(str).tolist()
    filename_order_preserved = observed_filenames == saved_filenames
    if not filename_order_preserved:
        raise ValueError("Full-test filename order was not preserved.")

    subject_ids = e07_predictions["subject_id"].astype(str).tolist()
    subject_id_alignment_preserved = len(subject_ids) == len(full_sequence_predictions)
    probabilities_in_range = all(
        0.0 <= prediction.probability_very_mild_dementia <= 1.0
        for prediction in full_sequence_predictions
    )
    if not subject_id_alignment_preserved:
        raise ValueError("Subject IDs do not align with the full-test prediction sequence.")
    if not probabilities_in_range:
        raise ValueError("One or more full-test probabilities are outside [0, 1].")

    full_test_absolute_differences = [
        abs(
            prediction.probability_very_mild_dementia
            - float(saved_probability)
        )
        for prediction, saved_probability in zip(
            full_sequence_predictions,
            e07_predictions["probability_very_mild_dementia"],
            strict=True,
        )
    ]

    subject_slice_predictions = [
        prediction
        for prediction, subject_id in zip(
            full_sequence_predictions,
            subject_ids,
            strict=True,
        )
        if subject_id == SUBJECT_ID
    ]
    if not subject_slice_predictions:
        raise ValueError(f"No E07 test slices found for subject {SUBJECT_ID}.")

    subject_prediction = aggregate_subject_mean_probability(
        subject_slice_predictions,
        model_name=MODEL_NAME,
    )
    saved_mean_probability = float(expected_subject["mean_probability"])
    absolute_difference = abs(subject_prediction.mean_model_score - saved_mean_probability)

    result = {
        "subject_id": SUBJECT_ID,
        "true_class": str(expected_subject["true_class"]),
        "number_of_slices": len(subject_slice_predictions),
        "expected_number_of_slices": int(expected_subject["number_of_test_images"]),
        "full_test_input_rows": len(e07_predictions),
        "full_test_current_predictions": len(full_sequence_predictions),
        "full_test_filename_order_preserved": filename_order_preserved,
        "full_test_subject_id_alignment_preserved": subject_id_alignment_preserved,
        "full_test_probabilities_in_range": probabilities_in_range,
        "full_test_max_absolute_slice_probability_difference": max(
            full_test_absolute_differences
        ),
        "full_test_mean_absolute_slice_probability_difference": sum(
            full_test_absolute_differences
        )
        / len(full_test_absolute_differences),
        "saved_e08_mean_probability": saved_mean_probability,
        "system_mean_model_score": subject_prediction.mean_model_score,
        "absolute_difference": absolute_difference,
        "saved_e08_predicted_class": str(expected_subject["mean_predicted_class"]),
        "system_predicted_class": subject_prediction.predicted_class,
        "threshold": THRESHOLD,
        "tolerance": CONSISTENCY_TOLERANCE,
        "batch_size": CONSISTENCY_BATCH_SIZE,
        "batching_note": (
            "The full E07 frozen test sequence is re-inferred in saved CSV row order "
            "with batch size 16 before selecting the subject. This reproduces the "
            "original test batch boundaries because CUDA floating-point results can "
            "depend slightly on batch shape and boundaries."
        ),
        "probability_consistency_passed": absolute_difference <= CONSISTENCY_TOLERANCE,
        "classification_consistency_passed": (
            subject_prediction.predicted_class == str(expected_subject["mean_predicted_class"])
        ),
        "slice_count_consistency_passed": len(subject_slice_predictions)
        == int(expected_subject["number_of_test_images"]),
    }
    write_json(RESULTS_DIR / "end_to_end_subject_consistency.json", result)
    return result


def make_valid_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="JPEG")
    return buffer.getvalue()


def build_route_test_client():
    from fastapi.testclient import TestClient
    from information_system import main

    fake_bundle = ModelBundle(
        model=FakeModel(),
        device=torch.device("cpu"),
        metadata={"model_name": MODEL_NAME},
    )
    original_loader = main.load_e07_resnet18_checkpoint
    main.load_e07_resnet18_checkpoint = lambda: fake_bundle
    return TestClient(main.app), main, original_loader


def evaluate_functional_scenarios() -> tuple[dict[str, Any], str]:
    from information_system import main

    client, main_module, original_loader = build_route_test_client()
    successful_result_html = ""
    scenario_rows: list[dict[str, Any]] = []

    def record(
        scenario: str,
        expected_behavior: str,
        response,
        passed: bool,
        observation: str,
    ) -> None:
        scenario_rows.append(
            {
                "scenario": scenario,
                "expected_behavior": expected_behavior,
                "observed_status_code": response.status_code,
                "passed": passed,
                "short_observation": observation,
            }
        )

    try:
        with client:
            response = client.get("/")
            record(
                "GET /",
                "HTTP 200, upload form visible, research warning visible",
                response,
                response.status_code == 200
                and 'type="file"' in response.text
                and "Research prototype only" in response.text,
                "Upload page rendered with research warning.",
            )

            response = client.post("/predict")
            record(
                "POST /predict with no files",
                "Clean validation response, no traceback displayed",
                response,
                response.status_code == 400
                and "No files were uploaded." in response.text
                and "Traceback" not in response.text,
                "No-file submission returned a clean error page.",
            )

            response = client.post(
                "/predict",
                files=[("files", ("slice.txt", b"not an image", "text/plain"))],
            )
            record(
                "POST /predict unsupported extension",
                "Unsupported extension rejected",
                response,
                response.status_code == 400
                and "Only JPG/JPEG MRI slice images are accepted." in response.text,
                "Unsupported extension rejected before inference.",
            )

            response = client.post(
                "/predict",
                files=[("files", ("slice.jpg", b"not a decodable image", "image/jpeg"))],
            )
            record(
                "POST /predict corrupt JPG bytes",
                "Corrupt JPG rejected",
                response,
                response.status_code == 400
                and "could not be decoded" in response.text
                and "Traceback" not in response.text,
                "Corrupt image bytes returned a clean error page.",
            )

            response = client.post(
                "/predict",
                files=[("files", ("slice.jpg", make_valid_jpeg_bytes(), "image/jpeg"))],
            )
            successful_result_html = response.text
            record(
                "POST /predict valid JPG",
                "Successful result page",
                response,
                response.status_code == 200
                and "Research classification" in response.text
                and "Mean model score for Very mild Dementia" in response.text,
                "Valid JPG returned a subject-level result page.",
            )

            too_many_files = [
                ("files", (f"slice_{index}.jpg", b"x", "image/jpeg"))
                for index in range(main.MAX_FILES_PER_REQUEST + 1)
            ]
            response = client.post("/predict", files=too_many_files)
            record(
                "POST /predict too many files",
                "Rejected when file count exceeds MAX_FILES_PER_REQUEST",
                response,
                response.status_code == 400
                and "Too many files were uploaded." in response.text,
                "File-count safety limit was enforced.",
            )

            oversized_bytes = b"0" * (main.MAX_FILE_SIZE_BYTES + 1)
            response = client.post(
                "/predict",
                files=[("files", ("large.jpg", oversized_bytes, "image/jpeg"))],
            )
            record(
                "POST /predict oversized file",
                "Rejected when a file exceeds MAX_FILE_SIZE_BYTES",
                response,
                response.status_code == 400
                and "File exceeds the maximum allowed size." in response.text,
                "Per-file size limit was enforced.",
            )
    finally:
        main_module.load_e07_resnet18_checkpoint = original_loader

    write_csv(
        RESULTS_DIR / "functional_scenarios.csv",
        scenario_rows,
        [
            "scenario",
            "expected_behavior",
            "observed_status_code",
            "passed",
            "short_observation",
        ],
    )

    summary = {
        "total_scenarios": len(scenario_rows),
        "passed_scenarios": sum(1 for row in scenario_rows if row["passed"]),
        "failed_scenarios": sum(1 for row in scenario_rows if not row["passed"]),
        "all_passed": all(row["passed"] for row in scenario_rows),
    }
    return summary, successful_result_html


def evaluate_result_presentation(result_html: str) -> dict[str, Any]:
    lower_html = result_html.lower()
    required_checks = {
        "analyzed_slice_count": "Analyzed slice count" in result_html,
        "model_name": MODEL_NAME in result_html,
        "mean_model_score_wording": "Mean model score for Very mild Dementia" in result_html,
        "decision_threshold": "Decision threshold" in result_html,
        "research_classification": "Research classification" in result_html,
        "research_only_warning": "Research prototype only" in result_html,
        "not_clinically_validated": "not clinically validated" in lower_html,
        "does_not_diagnose_alzheimers": "does not diagnose alzheimer" in lower_html,
    }
    prohibited_phrases = [
        "diagnosis:",
        "confirmed alzheimer's",
        "clinical diagnosis",
        "disease risk",
    ]
    prohibited_checks = {
        phrase: phrase not in lower_html for phrase in prohibited_phrases
    }
    result = {
        "required_checks": required_checks,
        "prohibited_phrase_checks": prohibited_checks,
        "passed": all(required_checks.values()) and all(prohibited_checks.values()),
    }
    write_json(RESULTS_DIR / "result_presentation_check.json", result)
    return result


def evaluate_privacy_storage() -> dict[str, Any]:
    main_source = (SRC_DIR / "information_system" / "main.py").read_text(encoding="utf-8")
    source_lower = main_source.lower()
    suspicious_write_terms = [
        ".write(",
        "open(",
        "shutil",
        "save(",
        "database",
        "sqlite",
        "sqlalchemy",
    ]
    static_checks = {
        "no_upload_write_calls_detected": not any(
            term in source_lower for term in suspicious_write_terms
        ),
        "uses_upload_read": "await upload.read()" in main_source,
        "no_database_import_detected": "database" not in source_lower
        and "sqlite" not in source_lower
        and "sqlalchemy" not in source_lower,
        "no_patient_identity_storage_detected": "patient" not in source_lower,
    }
    result = {
        "uploaded_file_bytes_processed_in_memory": static_checks["uses_upload_read"],
        "application_code_does_not_intentionally_write_uploads_to_disk": static_checks[
            "no_upload_write_calls_detected"
        ],
        "no_database_used": static_checks["no_database_import_detected"],
        "no_patient_identity_storage_implemented": static_checks[
            "no_patient_identity_storage_detected"
        ],
        "static_checks": static_checks,
        "limitations": [
            "no authentication",
            "no encryption layer added by the application",
            "no audit log",
            "no production hardening",
            "local demo only",
        ],
        "security_note": (
            "This is a code-level and behavior-scope check for the local prototype, "
            "not a strong security assessment."
        ),
        "passed": all(static_checks.values()),
    }
    write_json(RESULTS_DIR / "privacy_storage_check.json", result)
    return result


def collect_system_configuration(model_bundle: ModelBundle) -> dict[str, Any]:
    fastapi = importlib.import_module("fastapi")
    checkpoint_global_epoch = model_bundle.metadata.get("checkpoint_global_epoch")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    from information_system import main

    configuration = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "fastapi_version": fastapi.__version__,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_global_epoch": checkpoint_global_epoch,
        "device_used": str(model_bundle.device),
        "gpu_name": gpu_name,
        "input_size": INPUT_SIZE,
        "threshold": THRESHOLD,
        "aggregation": "mean probability",
        "maximum_file_count": main.MAX_FILES_PER_REQUEST,
        "maximum_per_file_size_bytes": main.MAX_FILE_SIZE_BYTES,
        "accepted_extensions": sorted(ACCEPTED_EXTENSIONS),
    }
    write_json(RESULTS_DIR / "system_configuration.json", configuration)
    return configuration


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("E11 - Final information-system evaluation")
    print("Running automated pytest suite...")
    test_suite_summary = run_pytest_suite()

    print("Loading E07 model once for real-model consistency checks...")
    model_bundle = load_e07_resnet18_checkpoint()

    print("Checking E07 first-batch inference consistency...")
    e07_consistency = evaluate_e07_inference_consistency(model_bundle)

    print(f"Checking subject-level end-to-end consistency for {SUBJECT_ID}...")
    subject_consistency = evaluate_subject_end_to_end(model_bundle)

    print("Evaluating FastAPI route scenarios with a fake route-level model...")
    functional_summary, result_html = evaluate_functional_scenarios()

    print("Checking result-presentation wording...")
    result_presentation = evaluate_result_presentation(result_html)

    print("Checking privacy/storage scope...")
    privacy_storage = evaluate_privacy_storage()

    print("Recording system configuration...")
    system_configuration = collect_system_configuration(model_bundle)

    mandatory_results = [
        test_suite_summary["overall_passed"],
        e07_consistency["passed"],
        subject_consistency["probability_consistency_passed"],
        subject_consistency["classification_consistency_passed"],
        subject_consistency["slice_count_consistency_passed"],
        functional_summary["all_passed"],
        result_presentation["passed"],
        privacy_storage["passed"],
    ]
    overall_status = "passed" if all(mandatory_results) else "failed"

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "objective": (
            "Evaluate the local research information-system prototype for "
            "pipeline consistency, validation behavior, presentation wording, "
            "privacy/storage scope and reproducible local execution."
        ),
        "automated_test_suite_result": test_suite_summary,
        "e07_inference_consistency_result": e07_consistency,
        "end_to_end_subject_consistency_result": subject_consistency,
        "functional_scenario_pass_counts": functional_summary,
        "result_presentation_check": result_presentation,
        "privacy_storage_findings": privacy_storage,
        "system_configuration": system_configuration,
        "overall_prototype_evaluation_status": overall_status,
        "interpretation": (
            "E11 evaluates implementation behavior only. It does not establish "
            "clinical validity, diagnostic safety, hospital readiness, regulatory "
            "compliance or real-world clinical effectiveness."
        ),
    }
    write_json(RESULTS_DIR / "e11_summary.json", summary)

    print(f"E11 overall prototype evaluation status: {overall_status}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
