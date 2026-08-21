# E11 - Final information-system evaluation

## Status

Implemented, not yet confirmed by saved execution results.

E11 must not be marked completed until `evaluate_information_system.py` has been executed successfully and its outputs have been reviewed.

## Objective

E11 evaluates the final local research information-system prototype.

The experiment verifies implementation behavior, not clinical validity.

The evaluation covers:

- model/checkpoint loading;
- consistency with saved E07 CNN inference;
- preprocessing behavior;
- subject-level mean-probability aggregation;
- threshold classification at `0.5`;
- upload validation and error handling;
- HTTP route behavior;
- non-diagnostic result presentation;
- absence of intentional permanent upload storage;
- reproducible local execution.

## Evaluated Prototype Architecture

The prototype is a local FastAPI research application.

The web layer accepts multiple JPG/JPEG MRI slices belonging to one subject, processes uploads in memory, applies the E07 evaluation preprocessing, runs the saved E07 ResNet18 Attempt 2 checkpoint, and aggregates slice probabilities using mean probability.

The subject-level result is displayed as a research classification only.

The system must not be described as a diagnostic or clinically validated tool.

## Inputs

E11 uses:

- `src/information_system/`;
- `tests/`;
- `experiments/E07_cnn/results/test_slice_predictions_attempt2.csv`;
- `experiments/E08_cnn_subject_level/results/subject_level_predictions.csv`;
- local raw ImagesOASIS JPG files under `data/raw/imagesoasis/Data`;
- the saved E07 Attempt 2 checkpoint.

## Methodology

The evaluation script runs the existing pytest suite using the current Python interpreter:

```bash
python -m pytest tests -q
```

It then performs real-model consistency checks:

- the first 16 saved E07 test slices are preprocessed and inferred as a single batch;
- current probabilities are compared with saved E07 probabilities using tolerance `1e-10`;
- subject `OAS1_0016` is evaluated end to end in deterministic batches of 16;
- the resulting mean subject score and threshold-0.5 classification are compared with saved E08 subject-level results.

Route-level validation and result-presentation checks use FastAPI `TestClient`. A lightweight fake model is used where route behavior is being evaluated, because the separate E07 consistency checks already verify the real model.

## Generated Outputs

When executed, E11 writes:

- `results/pytest_output.txt`;
- `results/test_suite_summary.json`;
- `results/e07_inference_consistency.csv`;
- `results/e07_inference_consistency_summary.json`;
- `results/end_to_end_subject_consistency.json`;
- `results/functional_scenarios.csv`;
- `results/result_presentation_check.json`;
- `results/privacy_storage_check.json`;
- `results/system_configuration.json`;
- `results/e11_summary.json`.

## Functional/Error Scenarios

The script evaluates:

- `GET /`;
- `POST /predict` with no files;
- unsupported extension;
- corrupt JPG bytes;
- valid JPG;
- too many files;
- oversized file.

The expected behavior is clean user-facing output without tracebacks.

## Result-Presentation Checks

The successful result page is checked for:

- analyzed slice count;
- model name;
- `Mean model score for Very mild Dementia`;
- decision threshold;
- research classification;
- research-only warning;
- wording that the prototype is not clinically validated;
- wording that it does not diagnose Alzheimer's disease.

The page is also checked for absence of diagnostic wording such as confirmed Alzheimer's, clinical diagnosis and disease risk.

## Privacy/Storage Scope

E11 includes code-level checks and documented limitations.

The prototype is expected to:

- process uploaded file bytes in memory;
- avoid intentional permanent upload storage;
- avoid a database;
- avoid patient identity storage.

Limitations:

- no authentication;
- no encryption layer added by the application;
- no audit log;
- no production hardening;
- local demo only.

These checks do not represent a full security assessment.

## Reproducibility

Run E11 from the repository root:

```bash
python experiments/E11_information_system_evaluation/evaluate_information_system.py
```

The script records Python, PyTorch, torchvision and FastAPI versions, device information, model/checkpoint metadata, accepted extensions and upload limits.

## Interpretation Rules

E11 may conclude only that the implemented research prototype behaves consistently with the saved ML pipeline and handles the evaluated usage/error scenarios.

E11 must not conclude:

- clinical validity;
- diagnostic safety;
- hospital readiness;
- regulatory compliance;
- real-world clinical effectiveness.
