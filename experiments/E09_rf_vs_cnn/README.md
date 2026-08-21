# E09 - Random Forest vs CNN Comparison

## Status

Completed.

`compare_rf_cnn.py` was executed successfully and the confirmed outputs are saved in `results/`.

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

The script wrote:

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

## Confirmed Results

The E09 primary comparison used the E02 Age >= 60 cohort, the same frozen 29 E03 test subjects, and all 6,954 frozen test slices.

Subject counts:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

The reconstructed Random Forest reproduced the saved E03 image-level result exactly:

| Metric | Expected | Reproduced | Absolute difference |
|---|---:|---:|---:|
| Accuracy | 0.5787598004276551 | 0.5787598004276551 | 0.0 |
| Balanced accuracy | 0.5787598004276551 | 0.5787598004276551 | 0.0 |
| ROC-AUC | 0.6254816707300159 | 0.6254816707300159 | 0.0 |

The E03 reproduction confusion matrix equality check passed.

Primary subject-level comparison at threshold `0.5` using mean probability:

| Model | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 positive | Macro F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6206896551724138 | 0.5906862745098039 | 0.4166666666666667 | 0.7647058823529411 | 0.5555555555555556 | 0.47619047619047616 | 0.5894465894465895 | 0.6666666666666666 | 0.5676781776588278 |
| ResNet18 CNN | 0.7241379310344828 | 0.7034313725490196 | 0.5833333333333334 | 0.8235294117647058 | 0.7 | 0.6363636363636364 | 0.7070707070707071 | 0.6911764705882353 | 0.6007440476190475 |

Metric differences are defined as CNN metric minus RF metric:

| Metric | Difference |
|---|---:|
| Accuracy | 0.10344827586206895 |
| Balanced accuracy | 0.11274509803921562 |
| Sensitivity | 0.16666666666666669 |
| Specificity | 0.05882352941176472 |
| Precision | 0.14444444444444438 |
| F1 positive | 0.1601731601731602 |
| Macro F1 | 0.11762411762411762 |
| ROC-AUC | 0.02450980392156865 |
| PR-AUC | 0.03306586996021976 |

Paired correctness:

| Outcome | Subjects |
|---|---:|
| Both correct | 15 |
| Both wrong | 5 |
| RF only correct | 3 |
| CNN only correct | 6 |

The exact McNemar supplementary test used 9 discordant pairs and gave p = 0.5078125.

The CNN had higher observed metrics on this frozen 29-subject test set. However, McNemar exact p = 0.5078125 does not establish statistically proven superiority.

These results must not be described using clinical-superiority or diagnostic-superiority language.

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
