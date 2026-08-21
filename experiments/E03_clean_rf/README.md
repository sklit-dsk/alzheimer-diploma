# E03 - Clean Random Forest Baseline

## Status

Completed.

`run_clean_rf_baseline.py` was executed successfully and the confirmed outputs are saved in `results/`.

## Objective

Repeat the original E00 handcrafted-feature Random Forest baseline on the E02 Age >= 60 cohort.

The purpose of E03 is to isolate the effect of cohort construction as much as possible. E03 does not tune the model, optimize thresholds, aggregate predictions to subject level, or perform corrected grouped Feature Importance.

## Relationship To E00

E03 preserves the main E00 baseline choices:

- grayscale image loading;
- resize to `(128, 64)`;
- bilinear resampling;
- float32 conversion and normalization to `[0, 1]`;
- Otsu-based brain-like mask with the same fallback behavior;
- the same 89 handcrafted image features;
- subject-level train/test split with seed `42`;
- image-level undersampling after the subject split;
- the same Random Forest hyperparameters;
- image-level metrics for direct E00-style comparison.

E00 remains frozen and is not modified.

## Relationship To E02

E02 is the authoritative cohort definition for E03.

E03 uses only subjects and scans from:

- `experiments/E02_age_control/results/e02_subject_level_cohort.csv`
- `experiments/E02_age_control/results/e02_scan_level_cohort.csv`

The expected E02 cohort is:

| Class | Subjects |
|---|---:|
| Non Demented | 85 |
| Very mild Dementia | 58 |
| Total | 143 |

The expected scan count is 143.

## Input Files

- `experiments/E00_baseline_rf/dataset_metadata.csv`
- `experiments/E02_age_control/results/e02_subject_level_cohort.csv`
- `experiments/E02_age_control/results/e02_scan_level_cohort.csv`
- `data/raw/imagesoasis/Data`

The obsolete Linux paths in E00 metadata are not used for image loading. E03 reconstructs local image paths from the raw ImagesOASIS root, class directory, and filename.

## Preprocessing

Each image is:

1. loaded as a JPG file;
2. converted to grayscale;
3. resized to `(128, 64)` with bilinear resampling;
4. converted to `float32`;
5. normalized to `[0, 1]`;
6. masked using the same Otsu-based logic used in E00.

## Handcrafted Features

E03 extracts exactly 89 numeric features:

- global intensity: `global_mean`, `global_std`, `global_p10`, `global_p50`, `global_p90`, `global_iqr`;
- intensity and mask geometry: `dark_ratio`, `bright_ratio`, `brain_area_ratio`, `bbox_area_ratio`, `centroid_x`, `centroid_y`;
- edges and entropy: `edge_mean`, `edge_p90`, `entropy`;
- regional intensity/contrast: mean and standard deviation for every cell in a 4x8 grid;
- texture: `lbp_uniform_bin_0` through `lbp_uniform_bin_9`.

## Subject-Level Split

The split reproduces E00 as closely as possible:

- random seed: `42`;
- approximately 80% train and 20% test;
- subject IDs are sorted and shuffled separately by class;
- all images from a subject are assigned to exactly one split;
- train/test subject overlap is explicitly validated as zero.

## Image-Level Balancing

After the subject split, E03 reproduces E00-style image-level undersampling separately inside train and test.

Balancing is not performed before subject splitting.

Because image-level undersampling can remove all retained images for a subject, E03 reports any such subjects in `results/subjects_eliminated_by_balancing.csv`.

## Random Forest Configuration

E03 uses exactly:

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

No hyperparameter tuning, cross-validation tuning, feature selection, threshold optimization, or calibration is performed.

## Generated Outputs

Artifacts:

- `artifacts/e02_full_features.pkl`
- `artifacts/binary_train_features.pkl`
- `artifacts/binary_test_features.pkl`

Results:

- `results/selected_image_metadata.csv`
- `results/train_subjects.csv`
- `results/test_subjects.csv`
- `results/split_summary.csv`
- `results/subjects_eliminated_by_balancing.csv`
- `results/image_level_predictions.csv`
- `results/image_level_metrics.json`
- `results/image_level_classification_report.csv`
- `results/image_level_confusion_matrix.csv`
- `results/image_level_confusion_matrix.png`
- `results/feature_importance.csv`
- `results/legacy_unadjusted_category_importance.csv`
- `results/top20_permutation_importance.png`
- `results/validation_summary.json`

## Confirmed Results

E03 used the E02 Age >= 60 cohort:

| Class | Subjects |
|---|---:|
| Non Demented | 85 |
| Very mild Dementia | 58 |
| Total | 143 |

The cohort contains 143 scans and 34,404 selected images. The model used exactly 89 handcrafted features. Train/test subject overlap was zero.

Subject split:

| Split | Non Demented subjects | Very mild Dementia subjects |
|---|---:|---:|
| Train | 68 | 46 |
| Test | 17 | 12 |

Image counts before balancing:

| Split | Non Demented images | Very mild Dementia images |
|---|---:|---:|
| Train | 16,531 | 10,919 |
| Test | 4,148 | 2,806 |

Image counts after E00-style image-level balancing:

| Split | Non Demented images | Very mild Dementia images |
|---|---:|---:|
| Train | 10,919 | 10,919 |
| Test | 2,806 | 2,806 |

No subjects were eliminated by image-level undersampling.

Confirmed image/slice-level metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.5787598004 |
| Balanced accuracy | 0.5787598004 |
| Macro F1 | 0.5732442943 |
| ROC-AUC | 0.6254816707 |

Class recall:

| Class | Recall |
|---|---:|
| Non Demented | 0.6924447612 |
| Very mild Dementia | 0.4650748396 |

Confusion matrix:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 1,943 | 863 |
| Very mild Dementia | 1,501 | 1,305 |

## Methodological Limitations

E03 metrics are image/slice-level metrics. They are not the final diploma evaluation.

Subject-level aggregation belongs to E04. Threshold analysis belongs to E05. Corrected grouped Feature Importance belongs to E06.

Compared with E00, the lower image-level ROC-AUC is evidence that cohort construction materially changes observed model performance. It should not be interpreted as proof that age alone caused the difference, because E00 and E03 differ by cohort composition and the final subject-level evaluation has not yet been performed.

The legacy category-importance table is saved only for E00 comparability. It must not be presented as corrected grouped Feature Importance, and image features or ROI regions must not be interpreted as medical biomarkers.

## How To Run

From the repository root:

```powershell
python experiments/E03_clean_rf/run_clean_rf_baseline.py
```

Feature extraction processes tens of thousands of JPG slices and prints progress while running.
