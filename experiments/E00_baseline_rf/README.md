# E00 — Original Random Forest Baseline

## Status

Completed and frozen.

This directory preserves the original experiment that was reviewed by the scientific supervisor.

E00 must not be modified to incorporate improvements from later experiments.

Any methodological improvements must be implemented as new experiments.

---

## Objective

Create an initial baseline for distinguishing:

- Non Demented
- Very mild Dementia

using handcrafted MRI image features and a Random Forest classifier.

---

## Main script

`analyze_imagesoasis_feature_importance.py`

This is the original Python script used for the baseline experiment and Feature Importance analysis.

---

## Dataset metadata

`dataset_metadata.csv`

Contains metadata generated/used during the original ImagesOASIS analysis.

The original dataset analysis reported:

| Class | Images | Subjects |
|---|---:|---:|
| Non Demented | 67,222 | 266 |
| Very mild Dementia | 13,725 | 58 |
| Mild Dementia | 5,002 | 21 |
| Moderate Dementia | 488 | 2 |

The binary early-detection experiment used:

- Non Demented
- Very mild Dementia

---

## Data split

The original experiment performed the train/test split at subject level.

The purpose was to prevent MRI slices belonging to the same subject from appearing in both training and test sets.

This methodological decision was positively noted by the scientific supervisor.

---

## Artifacts

The `artifacts/` directory contains derived feature datasets:

- `binary_train_features.pkl`
- `binary_test_features.pkl`

These are local experiment artifacts and are not intended to be committed to Git.

---

## Results

The `results/` directory contains the original outputs:

- `metrics.json`
- `category_importance.csv`
- `feature_importance.csv`
- `feature_importance_report.md`
- `class_distribution.png`
- `confusion_matrix.png`
- `roi_importance_heatmap.png`
- `top20_permutation_importance.png`

---

## Original baseline metrics

The recorded initial results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.609 |
| Balanced Accuracy | 0.609 |
| Macro F1 | 0.583 |
| ROC-AUC | 0.770 |

Class-level recall:

| Class | Recall |
|---|---:|
| Non Demented | 0.859 |
| Very mild Dementia | 0.360 |

These values belong to the original experiment and must not be presented as the final performance of the diploma system.

---

## Known limitations

### Slice-level evaluation

Although the train/test split was performed at subject level, the recorded model metrics were calculated primarily at MRI slice level.

The final diploma evaluation must instead provide subject-level metrics after aggregation of slice predictions.

### Very mild Dementia sensitivity

Recall for Very mild Dementia was approximately 0.36 at slice level.

This is an important limitation because the diploma focuses on early detection.

### Potential age confounding

Clinical age information was not included in the current dataset metadata.

Therefore the original experiment could not determine whether age differences between diagnostic groups influenced classification.

This issue is addressed starting with E01.

### Grouped Feature Importance

Feature groups contain different numbers of individual features.

The original analysis compared summed importance values between groups, which may favor groups containing more features.

This must be corrected in E06 using normalized or mean importance.

### Interpretation

Image features identified as important by the model must be described as image characteristics used by the classifier.

They must not be interpreted as confirmed medical biomarkers.

---

## Supervisor feedback

The scientific supervisor considered the initial approach appropriate, especially:

1. selection of Non Demented vs Very mild Dementia as the early-recognition scenario;
2. subject-level train/test separation.

Before development of the main image-based model, the supervisor requested:

1. integration of clinical metadata;
2. analysis of Age, Sex, CDR and MMSE where available;
3. investigation of age as a potential confounding factor;
4. construction of a cleaner cohort;
5. repetition of the baseline;
6. subject-level final evaluation;
7. sensitivity/specificity threshold analysis;
8. corrected grouped Feature Importance;
9. development of an image-based model such as a transfer-learning CNN;
10. comparison between the CNN and Random Forest baseline.

---

## Next experiment

E01 — Clinical metadata and confounder analysis.

E01 will investigate clinical metadata and determine the methodological basis for constructing the cohort used in subsequent experiments.