# E09 - Random Forest vs CNN Comparison

## Status

Implemented, not yet confirmed.

E09 must not be marked completed until `compare_rf_cnn.py` has run successfully and the generated outputs have been reviewed.

## Objective

Compare the corrected handcrafted-feature Random Forest baseline with the ResNet18 CNN at subject level.

The comparison is a research classification comparison. It is not clinical validation and must not be described as diagnostic superiority.

## Why E04 And E08 Were Not Perfectly Symmetric

E04 aggregated E03 Random Forest predictions that came from the historical E03/E00-style image-balanced test set.

That E04 RF subject-level result used 5,612 balanced test slices.

E08 CNN subject-level evaluation used all 6,954 frozen E03 test slices.

Therefore E09 reconstructs the E03 Random Forest and applies it to all 6,954 frozen test slices before comparing RF and CNN. This preserves E04 as a valid historical pipeline result while creating a more symmetric primary model comparison.

## Inputs

E09 uses saved artifacts and results only:

- `experiments/E03_clean_rf/artifacts/binary_train_features.pkl`;
- `experiments/E03_clean_rf/artifacts/binary_test_features.pkl`;
- `experiments/E03_clean_rf/artifacts/e02_full_features.pkl`;
- `experiments/E03_clean_rf/results/image_level_metrics.json`;
- `experiments/E03_clean_rf/results/test_subjects.csv`;
- `experiments/E04_subject_level/results/e04_summary.json`;
- `experiments/E08_cnn_subject_level/results/e08_summary.json`;
- `experiments/E08_cnn_subject_level/results/subject_level_predictions.csv`.

E09 does not retrain the CNN, modify the CNN checkpoint, perform new MRI preprocessing, or perform new handcrafted feature extraction.

## Random Forest Reconstruction

The Random Forest is reconstructed from the saved E03 training features using the E03 configuration:

```python
RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
```

The script uses the same 89 feature columns as E03.

Feature matrices are handled as in E03:

- infinities are replaced with missing values;
- missing values are filled with `0`;
- no scaler is used.

This is a reconstruction of the already-defined E03 model configuration, not a new hyperparameter search.

## Mandatory E03 Reproduction Check

Before the reconstructed RF is used for the full-test comparison, it is evaluated on:

- `experiments/E03_clean_rf/artifacts/binary_test_features.pkl`

The reproduced image-level metrics and confusion matrix must match the saved E03 result.

If this check fails, the script stops and does not continue to the RF-vs-CNN comparison.

The check is saved as:

- `results/e03_rf_reproduction_check.json`

## Primary Comparison

The primary comparison uses:

- E02 Age >= 60 cohort;
- frozen E03 test subjects;
- exactly 29 subjects;
- 17 Non Demented subjects;
- 12 Very mild Dementia subjects;
- all 6,954 frozen test slices;
- subject-level evaluation;
- mean probability aggregation;
- default threshold `0.5`;
- Very mild Dementia as the positive class.

Both models are compared using exactly one row per subject.

## Metrics

E09 reports:

- accuracy;
- balanced accuracy;
- sensitivity / recall for Very mild Dementia;
- specificity for Non Demented;
- precision for Very mild Dementia;
- F1 for Very mild Dementia;
- macro F1;
- ROC-AUC;
- PR-AUC;
- confusion matrix.

Metric differences are defined as:

CNN metric - RF metric

## Paired Comparison

Because both models are evaluated on the same 29 subjects, E09 also records paired correctness counts:

- both correct;
- both wrong;
- RF only correct;
- CNN only correct.

If SciPy is available, the script calculates an exact McNemar supplementary test from the paired binary correctness outcomes.

This is interpreted cautiously because the test set contains only 29 subjects. A non-significant result does not prove equivalence, and a significant result would not establish clinical superiority.

## Generated Outputs

The script writes:

- `results/e03_rf_reproduction_check.json`;
- `results/rf_full_test_slice_predictions.csv`;
- `results/rf_full_test_subject_predictions.csv`;
- `results/rf_full_test_subject_metrics.json`;
- `results/rf_full_test_subject_confusion_matrix.csv`;
- `results/rf_full_test_subject_confusion_matrix.png`;
- `results/paired_subject_predictions.csv`;
- `results/model_comparison.csv`;
- `results/paired_correctness_summary.json`;
- `results/e04_vs_e09_rf_comparison.csv`;
- `results/rf_vs_cnn_subject_metrics.png`;
- `results/rf_vs_cnn_roc_curves.png`;
- `results/rf_vs_cnn_pr_curves.png`;
- `results/e09_summary.json`.

## Methodological Restrictions

E09 does not:

- modify E00-E08;
- retrain the CNN;
- tune RF hyperparameters;
- optimize thresholds;
- use test labels for model selection;
- perform new feature engineering;
- perform new MRI preprocessing;
- describe image features as biomarkers;
- make clinical diagnostic claims.

## How To Run

From the repository root:

```powershell
python experiments/E09_rf_vs_cnn/compare_rf_cnn.py
```
