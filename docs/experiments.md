# Experiment Log

This document is the central experiment log for the diploma project.

Only results obtained from actually executed experiments may be recorded as confirmed results.

Do not overwrite historical experiment results when methodology changes.

---

# Experiment status

| ID | Experiment | Status |
|---|---|---|
| E00 | Original Random Forest baseline | Completed |
| E01 | Clinical metadata and confounder analysis | Completed |
| E02 | Cohort construction / age control | Completed |
| E03 | Clean Random Forest baseline | Completed |
| E04 | Subject-level aggregation | Completed |
| E05 | Threshold and sensitivity-specificity analysis | Current / in progress |
| E06 | Corrected Feature Importance | Completed |
| E07 | Transfer-learning CNN | Completed |
| E08 | CNN subject-level evaluation | Completed |
| E09 | Random Forest vs CNN comparison | Current / in progress |
| E10 | Explainability | Optional |
| E11 | Final information-system evaluation | Planned |

---

# E00 — Original Random Forest baseline

## Status

Completed.

## Objective

Create an initial baseline for distinguishing:

- Non Demented
- Very mild Dementia

using handcrafted image features and a Random Forest classifier.

## Dataset

ImagesOASIS.

The analyzed dataset contained multiple MRI slices for each subject.

Observed class distribution in the initial dataset analysis:

| Class | Images | Subjects |
|---|---:|---:|
| Non Demented | 67,222 | 266 |
| Very mild Dementia | 13,725 | 58 |
| Mild Dementia | 5,002 | 21 |
| Moderate Dementia | 488 | 2 |

The early-detection experiment used only:

- Non Demented
- Very mild Dementia

## Data splitting

Train/test separation was performed at subject level.

MRI slices belonging to the same subject were not intentionally distributed between both training and test sets.

This was done to reduce subject leakage.

## Model

Random Forest classifier using handcrafted image features.

## Initial results

Recorded initial metrics:

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

## Important limitation

These initial metrics were calculated primarily at MRI slice level.

Therefore they must not be treated as the final evaluation of the diploma system.

The Very mild Dementia recall was relatively low.

Because the research focuses on early detection, this limitation is especially important.

## Feature Importance

Feature Importance analysis was performed.

An important methodological limitation was identified:

feature groups contained different numbers of individual features, while group importance was compared using summed positive importance values.

Therefore grouped importance must later be normalized relative to the number of features in each group.

The extracted image characteristics must not automatically be interpreted as medical biomarkers.

## Supervisor assessment

The supervisor considered the initial approach appropriate, especially:

- selection of Non Demented vs Very mild Dementia for early recognition;
- subject-level train/test separation.

Before development of the main model, additional methodological improvements were requested.

## Conclusion

E00 is retained as the original baseline.

It must not be overwritten.

Future experiments will determine how clinical metadata, cohort construction, subject-level evaluation and alternative models affect the results.

---

# E01 — Clinical metadata and confounder analysis

## Status

Completed.

## Objective

Connect ImagesOASIS subjects with available clinical metadata and investigate possible confounding factors before developing the main model.

## Required metadata

Where available:

- Age
- Sex
- CDR
- MMSE

## Main questions

1. What is the age distribution of Non Demented subjects?
2. What is the age distribution of Very mild Dementia subjects?
3. Are the two groups substantially different in age?
4. What is the sex distribution of both groups?
5. Are CDR values consistent with the selected classes?
6. How much MMSE data is available?
7. Are important clinical values missing?
8. Is age likely to act as a confounding factor?
9. Is a 60+ cohort feasible?
10. Is approximate age matching feasible?

## Generated outputs

E01 produced:

- `experiments/E01_clinical_metadata/results/scan_level_metadata.csv`;
- `experiments/E01_clinical_metadata/results/subject_level_metadata.csv`;
- `experiments/E01_clinical_metadata/results/missing_values.csv`;
- `experiments/E01_clinical_metadata/results/age_summary.csv`;
- `experiments/E01_clinical_metadata/results/age_comparison.csv`;
- `experiments/E01_clinical_metadata/results/age_distribution.png`;
- `experiments/E01_clinical_metadata/results/sex_distribution.csv`;
- `experiments/E01_clinical_metadata/results/sex_distribution.png`;
- `experiments/E01_clinical_metadata/results/cdr_distribution.csv`;
- `experiments/E01_clinical_metadata/results/cdr_consistency.csv`;
- `experiments/E01_clinical_metadata/results/mmse_summary.csv`;
- `experiments/E01_clinical_metadata/results/age_60plus_summary.csv`;
- `experiments/E01_clinical_metadata/results/age_60plus_comparison.csv`;
- `experiments/E01_clinical_metadata/results/age_matching_exact_age_counts.csv`;
- `experiments/E01_clinical_metadata/results/age_matching_5year_bins.csv`;
- `experiments/E01_clinical_metadata/results/e01_summary.json`.

Additional validation reports were also saved for unmatched scans, unmatched subjects and conflicting values.

## Confirmed results

The binary ImagesOASIS cohort contained:

| Class | Subjects |
|---|---:|
| Non Demented | 266 |
| Very mild Dementia | 58 |
| Total | 324 |

The scan-level metadata table contained 343 unique scans.

Clinical metadata matching coverage was complete:

| Unit | Matched | Total | Coverage |
|---|---:|---:|---:|
| Scans | 343 | 343 | 100.0% |
| Subjects with all scans matched | 324 | 324 | 100.0% |

Subject-level missing values:

| Variable | Available | Missing | Missing % |
|---|---:|---:|---:|
| Age | 324 | 0 | 0.0 |
| Sex | 324 | 0 | 0.0 |
| CDR | 177 | 147 | 45.3704 |
| MMSE | 177 | 147 | 45.3704 |

No scan-level or subject-level clinical value conflicts were detected.

CDR consistency:

| Class | Expected CDR when available | Available CDR | Missing CDR | Inconsistent CDR |
|---|---:|---:|---:|---:|
| Non Demented | 0.0 | 119 | 147 | 0 |
| Very mild Dementia | 0.5 | 58 | 0 | 0 |

Missing CDR was preserved as missing and was not interpreted as CDR = 0.

Age summary in the original binary groups:

| Class | n | Mean | Median | Min | Max | Q1 | Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non Demented | 266 | 45.7819548872 | 43.5 | 18.0 | 94.0 | 22.0 | 69.0 |
| Very mild Dementia | 58 | 75.9655172414 | 77.0 | 62.0 | 90.0 | 71.0 | 81.0 |

Age comparison in the original binary groups:

| Comparison | Value |
|---|---:|
| Mean difference, Very mild Dementia - Non Demented | 30.1835623542 |
| Median difference | 33.5 |
| Mann-Whitney U | 12923.0 |
| Two-sided p-value | 7.552687005154834e-16 |
| Cliff's delta | 0.6752657506 |

This confirms a strong age imbalance between the original groups. The statistical result must not be interpreted as clinical importance by itself.

Exploratory Age >= 60 cohort:

| Class | Original subjects | Retained subjects | Retained % |
|---|---:|---:|---:|
| Non Demented | 266 | 85 | 31.9549 |
| Very mild Dementia | 58 | 58 | 100.0 |

Age comparison within the exploratory Age >= 60 cohort:

| Comparison | Value |
|---|---:|
| Mean difference, Very mild Dementia - Non Demented | -0.4462474645 |
| Median difference | 3.0 |
| Mann-Whitney U | 2425.0 |
| Two-sided p-value | 0.8708722006254727 |
| Cliff's delta | -0.0162271805 |

This does not prove that the age distributions are identical. However, the severe original age imbalance is substantially reduced within the Age >= 60 cohort.

## Conclusion

Age is an important potential confounding factor in the original Non Demented vs Very mild Dementia ImagesOASIS groups.

Age >= 60 is selected as the primary cohort strategy to be implemented and formally validated in E02. E01 does not create E02 data and does not train or evaluate a model on the selected cohort.

Approximate age matching remains a possible sensitivity or additional analysis rather than the primary strategy, because it would reduce the available sample further.

---

# E02 — Cohort construction / age control

## Status

Completed.

## Objective

Construct a methodologically cleaner Non Demented vs Very mild Dementia cohort based on the findings from E01.

The primary strategy selected from E01 is to restrict the cohort to subjects aged 60+.

E02 implements and validates this cohort for downstream experiments.

Approximate age matching remains a possible sensitivity or additional analysis, but it is not the primary strategy because it would reduce the available sample further.

## Generated outputs

E02 produced:

- `experiments/E02_age_control/results/e02_subject_level_cohort.csv`;
- `experiments/E02_age_control/results/e02_scan_level_cohort.csv`;
- `experiments/E02_age_control/results/e02_age_summary.csv`;
- `experiments/E02_age_control/results/e02_age_comparison.csv`;
- `experiments/E02_age_control/results/e02_sex_distribution.csv`;
- `experiments/E02_age_control/results/e02_cdr_distribution.csv`;
- `experiments/E02_age_control/results/e02_cdr_consistency.csv`;
- `experiments/E02_age_control/results/e02_mmse_summary.csv`;
- `experiments/E02_age_control/results/e02_missing_values.csv`;
- `experiments/E02_age_control/results/e02_retained_percentages.csv`;
- `experiments/E02_age_control/results/e02_summary.json`.

## Confirmed results

The E02 Age >= 60 cohort contains:

| Class | Subjects |
|---|---:|
| Non Demented | 85 |
| Very mild Dementia | 58 |
| Total | 143 |

The scan-level cohort contains 143 scans.

Age statistics:

| Class | Mean | Std | Median | Range |
|---|---:|---:|---:|---:|
| Non Demented | 76.411765 | 8.897959 | 74.0 | 60-94 |
| Very mild Dementia | 75.965517 | 6.986115 | 77.0 | 62-90 |

Age comparison, Very mild Dementia minus Non Demented:

| Measure | Value |
|---|---:|
| Mean difference | -0.446247 |
| Median difference | 3.0 |
| Mann-Whitney U | 2425.0 |
| Two-sided p-value | 0.870872 |
| Cliff's delta | -0.016227 |

## Conclusion

Age >= 60 is the selected primary cohort for downstream experiments.

This cohort substantially reduces the severe original age imbalance documented in E01, but it does not prove that the class age distributions are identical or that all possible confounding has been removed.

Approximate age matching remains optional as a sensitivity or additional analysis.

---

# E03 — Clean Random Forest baseline

## Status

Completed.

## Objective

Repeat the Random Forest baseline using the cohort selected in E02.

The experiment should preserve subject-level separation and provide a clean baseline for comparison with later models.

E03 must use the E02 Age >= 60 cohort and reproduce the E00 handcrafted-feature Random Forest baseline as faithfully as possible without hyperparameter tuning, threshold optimization or subject-level aggregation.

## Generated outputs

E03 produced:

- `experiments/E03_clean_rf/artifacts/e02_full_features.pkl`;
- `experiments/E03_clean_rf/artifacts/binary_train_features.pkl`;
- `experiments/E03_clean_rf/artifacts/binary_test_features.pkl`;
- `experiments/E03_clean_rf/results/selected_image_metadata.csv`;
- `experiments/E03_clean_rf/results/train_subjects.csv`;
- `experiments/E03_clean_rf/results/test_subjects.csv`;
- `experiments/E03_clean_rf/results/split_summary.csv`;
- `experiments/E03_clean_rf/results/subjects_eliminated_by_balancing.csv`;
- `experiments/E03_clean_rf/results/image_level_predictions.csv`;
- `experiments/E03_clean_rf/results/image_level_metrics.json`;
- `experiments/E03_clean_rf/results/image_level_classification_report.csv`;
- `experiments/E03_clean_rf/results/image_level_confusion_matrix.csv`;
- `experiments/E03_clean_rf/results/image_level_confusion_matrix.png`;
- `experiments/E03_clean_rf/results/feature_importance.csv`;
- `experiments/E03_clean_rf/results/legacy_unadjusted_category_importance.csv`;
- `experiments/E03_clean_rf/results/top20_permutation_importance.png`;
- `experiments/E03_clean_rf/results/validation_summary.json`.

## Confirmed results

E03 reproduced the E00-style handcrafted-feature Random Forest baseline on the E02 Age >= 60 cohort.

E03 cohort and validation:

| Item | Value |
|---|---:|
| Total subjects | 143 |
| Non Demented subjects | 85 |
| Very mild Dementia subjects | 58 |
| Total scans | 143 |
| Selected images | 34,404 |
| Handcrafted features | 89 |
| Train/test subject overlap | 0 |

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

Image counts after balancing:

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

## Conclusion

E03 confirms that the E00-style Random Forest can be reproduced on the E02 Age >= 60 cohort.

The metrics remain image/slice-level only and must not be treated as final model performance.

The decrease in ROC-AUC relative to E00 is evidence that cohort construction materially changes observed model performance, but it is not proof that age alone caused the difference.

Subject-level evaluation is still required in E04 before drawing final conclusions about model performance.

---

# E04 — Subject-level aggregation

## Status

Completed.

## Objective

Convert slice-level model predictions into one final prediction per subject.

Candidate aggregation methods:

- mean probability;
- median probability;
- majority voting.

The aggregation strategies should be compared before selecting the primary method.

E04 must use E03 image-level predictions and calculate metrics from exactly one row per subject. Threshold optimization remains part of E05.

## Generated outputs

E04 produced:

- `experiments/E04_subject_level/results/subject_level_predictions.csv`;
- `experiments/E04_subject_level/results/mean_probability_metrics.json`;
- `experiments/E04_subject_level/results/median_probability_metrics.json`;
- `experiments/E04_subject_level/results/majority_vote_metrics.json`;
- `experiments/E04_subject_level/results/mean_probability_confusion_matrix.csv`;
- `experiments/E04_subject_level/results/median_probability_confusion_matrix.csv`;
- `experiments/E04_subject_level/results/majority_vote_confusion_matrix.csv`;
- `experiments/E04_subject_level/results/mean_probability_confusion_matrix.png`;
- `experiments/E04_subject_level/results/median_probability_confusion_matrix.png`;
- `experiments/E04_subject_level/results/majority_vote_confusion_matrix.png`;
- `experiments/E04_subject_level/results/aggregation_comparison.csv`;
- `experiments/E04_subject_level/results/e04_summary.json`.

## Confirmed results

The subject-level E04 test set contains:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

All E04 metrics are calculated from exactly one row per subject.

Aggregation comparison:

| Method | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 positive | Macro F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean probability | 0.6206896552 | 0.5906862745 | 0.4166666667 | 0.7647058824 | 0.5555555556 | 0.4761904762 | 0.5894465894 | 0.6568627451 | 0.5628292498 |
| Median probability | 0.5862068966 | 0.5612745098 | 0.4166666667 | 0.7058823529 | 0.5 | 0.4545454545 | 0.5606060606 | 0.6617647059 | 0.5643022487 |
| Majority vote | 0.6206896552 | 0.5906862745 | 0.4166666667 | 0.7647058824 | 0.5555555556 | 0.4761904762 | 0.5894465894 | 0.6764705882 | 0.5561421583 |

Confusion matrices:

Mean probability:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 13 | 4 |
| Very mild Dementia | 7 | 5 |

Median probability:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 12 | 5 |
| Very mild Dementia | 7 | 5 |

Majority vote:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 13 | 4 |
| Very mild Dementia | 7 | 5 |

One subject had an exact majority-vote tie. The documented rule assigns exact ties to Non Demented.

## Conclusion

E04 converts E03 image-level predictions into subject-level predictions and evaluates three aggregation strategies at the default threshold of 0.5.

The default-threshold sensitivity for Very mild Dementia remains limited at approximately 41.7%.

Mean probability and majority voting give identical default-threshold classifications in this run. Different continuous scores produce somewhat different ROC-AUC values.

No aggregation method should be declared universally superior from this small 29-subject test set.

Mean probability is selected as the primary working probability aggregation for E05 because it provides a direct continuous subject-level probability and can later be applied consistently to CNN outputs. This is a methodological working choice, not proof that mean aggregation is clinically optimal.

Threshold behavior must be analyzed in E05 before final model-performance claims are made.

---

# E05 — Threshold and sensitivity-specificity analysis

## Status

Completed.

## Objective

Analyze model behavior at different classification thresholds.

Special attention should be paid to:

- sensitivity for Very mild Dementia;
- specificity;
- trade-off between sensitivity and specificity.

The final interpretation should not depend only on the default threshold of 0.5.

## Planned input

E05 uses the E04 subject-level prediction table:

- `experiments/E04_subject_level/results/subject_level_predictions.csv`

The primary score for threshold analysis is `mean_probability`.

Mean probability is used as the primary working aggregation from E04 because it provides a direct continuous subject-level score and can later be applied consistently to CNN outputs. This is a methodological working choice, not proof that mean aggregation is clinically optimal.

## Planned methodology

E05 evaluates a predefined threshold grid from 0.10 to 0.90 in increments of 0.05.

Very mild Dementia is the positive class.

For each threshold, the prediction rule is:

- Very mild Dementia if `mean_probability >= threshold`;
- Non Demented otherwise.

E05 must calculate subject-level sensitivity, specificity, balanced accuracy, precision, F1 and confusion-matrix counts across the predefined grid.

ROC-AUC and PR-AUC must be calculated from the continuous subject-level `mean_probability` score, not separately for each binary threshold.

The default threshold of 0.50 remains the E04 reference point and must reproduce the confirmed E04 mean-probability metrics.

## Interpretation rules

E05 is descriptive threshold analysis on the existing E03/E04 29-subject test set.

It must not be treated as deployment threshold optimization.

No threshold identified in E05 should be described as optimal, validated, clinically recommended, or final.

Validated threshold selection would require an independent validation cohort or nested cross-validation.

---

# E06 — Corrected Feature Importance

## Status

Completed.

## Objective

Correct grouped Feature Importance analysis.

For each feature group, record:

- number of features;
- total importance;
- normalized or mean importance per feature.

Interpret the results as image characteristics used by the model, not as confirmed medical biomarkers.

## Planned input

E06 uses the E03 feature-importance outputs:

- `experiments/E03_clean_rf/results/feature_importance.csv`;
- `experiments/E03_clean_rf/results/legacy_unadjusted_category_importance.csv`.

E06 must not retrain the Random Forest or modify E03 predictions and metrics.

## Planned methodology

E06 preserves the raw per-feature E03 importance values and creates corrected grouped summaries that account for the number of features in each category.

The main corrected grouped comparisons are:

- mean permutation importance per feature;
- mean positive permutation importance per feature;
- mean RF impurity importance per feature.

Legacy summed category importance is retained only as an unadjusted comparison. It should be described as potentially biased toward larger feature groups, not as a final corrected grouped interpretation.

## Interpretation rules

Feature Importance measures model reliance, not clinical causality.

High feature importance must not be interpreted as proof of a medical biomarker.

Regional image-grid features may be described only as image regions whose handcrafted features contributed to the model, not as disease regions or clinically confirmed anatomical findings.

---

# E07 — Transfer-learning CNN

## Status

Completed.

## Objective

Develop the main image-based model after E01-E06 have addressed the methodological issues identified by the supervisor.

E07 implements transfer learning with ResNet18 and ImageNet pretrained weights.

The model should use the same clean cohort and subject separation principles used for the corrected baseline.

## Planned input

E07 uses:

- `experiments/E03_clean_rf/results/train_subjects.csv`;
- `experiments/E03_clean_rf/results/test_subjects.csv`;
- `experiments/E03_clean_rf/results/selected_image_metadata.csv`;
- `data/raw/imagesoasis/Data`.

The E03 test subjects are frozen and must not be used for training, validation, early stopping, model selection, hyperparameter tuning, augmentation decisions or threshold selection.

## Planned methodology

Only the E03 train subjects are split internally into CNN train and validation subjects using a deterministic subject-level stratified split with seed `42`.

The E03 test subjects remain the frozen test set for one-time inference after checkpoint selection.

The model is `torchvision.models.resnet18` with ImageNet pretrained weights and a two-class final layer.

MRI JPG images are loaded as grayscale, converted to three channels for ResNet18, resized to `224 x 224`, and normalized with ImageNet normalization.

Training augmentation is conservative: small rotation, translation and scale variation only. Flipping, aggressive cropping and large rotations are not used.

Checkpoint selection uses validation subject-level ROC-AUC based on mean-probability aggregation. Validation loss is used only as a deterministic tie-breaker.

E07 does not optimize a classification threshold. The default threshold remains `0.5` for descriptive validation metrics.

## Interpretation rules

E07 saves CNN test slice predictions only after the best validation checkpoint is selected.

Final CNN subject-level test evaluation belongs to E08.

E07 must not claim CNN superiority over the Random Forest baseline and must not present the model as clinically validated.

## Confirmed execution status

E07 Attempt 2 completed training and postprocess-only recovery using the locally saved best checkpoint.

The frozen test inference produced:

- `experiments/E07_cnn/results/test_slice_predictions_attempt2.csv`

The file contains 6,954 slice-level predictions for the frozen 29-subject test set.

E07 does not calculate final CNN subject-level metrics. That evaluation belongs to E08.

---

# E08 — CNN subject-level evaluation

## Status

Current / in progress.

## Objective

Evaluate CNN predictions at subject level using the same methodological principles and metrics used for the Random Forest model.

## Input

E08 uses the E07 Attempt 2 frozen-test slice prediction table:

- `experiments/E07_cnn/results/test_slice_predictions_attempt2.csv`

## Methodology

E08 aggregates CNN slice-level predictions into exactly one row per subject using:

- mean probability;
- median probability;
- majority voting.

Very mild Dementia is the positive class.

The default threshold remains `0.5`.

E08 must not retrain the CNN, modify the checkpoint, choose a different checkpoint, optimize thresholds, or use test labels for training decisions.

## Generated outputs

E08 produced:

- `experiments/E08_cnn_subject_level/results/subject_level_predictions.csv`;
- subject-level metrics for each aggregation strategy;
- confusion matrices;
- `experiments/E08_cnn_subject_level/results/aggregation_comparison.csv`;
- `experiments/E08_cnn_subject_level/results/e08_summary.json`.

## Confirmed results

The E08 frozen CNN test set contains:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

The test set contains 6,954 slice predictions.

All E08 metrics are calculated from exactly one row per subject.

Mean-probability aggregation at threshold `0.5`:

| Metric | Value |
|---|---:|
| Accuracy | 0.7241379310 |
| Balanced accuracy | 0.7034313725 |
| Sensitivity | 0.5833333333 |
| Specificity | 0.8235294118 |
| Precision | 0.7 |
| F1 positive | 0.6363636364 |
| Macro F1 | 0.7070707071 |
| ROC-AUC | 0.6911764706 |
| PR-AUC | 0.6007440476 |

Confusion matrix for mean-probability aggregation:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 14 | 3 |
| Very mild Dementia | 5 | 7 |

## Interpretation rules

All E08 metrics must be calculated from exactly one row per subject.

The frozen test set contains only 29 subjects, so results must be interpreted cautiously.

E08 does not perform threshold optimization. It does not prove clinical diagnostic validity.

---

# E09 — Random Forest vs CNN comparison

## Status

Current / in progress.

## Objective

Compare the traditional handcrafted-feature Random Forest approach with the image-based CNN approach.

The comparison should focus not only on overall accuracy but also on:

- sensitivity;
- specificity;
- F1;
- ROC-AUC;
- PR-AUC;
- subject-level performance.

## Planned methodology

E09 reconstructs the E03 Random Forest from saved E03 feature artifacts and first verifies that the reconstructed model reproduces the saved E03 image-level result on `binary_test_features.pkl`.

Only after that check passes, E09 applies the reconstructed Random Forest to all 6,954 frozen E03 test slices. This creates a more symmetric comparison with E08, because historical E04 RF aggregation used 5,612 image-balanced test slices while E08 CNN aggregation used all frozen test slices.

The primary RF-vs-CNN comparison uses:

- the same 29 frozen E03 test subjects;
- mean probability aggregation;
- subject-level metrics;
- default threshold `0.5`;
- Very mild Dementia as the positive class.

E09 must not retrain the CNN, tune RF hyperparameters, optimize thresholds, perform new feature extraction, or make clinical diagnostic claims.

## Planned outputs

E09 should produce:

- RF reconstruction / E03 reproduction check;
- RF full-test slice predictions;
- RF full-test subject predictions;
- paired RF and CNN subject predictions;
- subject-level model comparison table;
- paired correctness summary;
- historical E04 RF vs symmetric E09 RF comparison;
- comparison figures;
- `experiments/E09_rf_vs_cnn/results/e09_summary.json`.

## Interpretation rules

Metric differences must be interpreted as descriptive observations on the frozen 29-subject test set.

E09 must not be marked completed until the script has run successfully and its generated results have been reviewed.

---

# E10 — Explainability

## Status

Optional.

## Objective

If sufficient time remains, investigate model explainability.

Possible methods may include visualization approaches such as Grad-CAM for the CNN.

Explainability results must be interpreted cautiously and must not automatically be described as medical biomarkers.

---

# E11 — Final information-system evaluation

## Status

Planned.

## Objective

Evaluate the final research information-system prototype.

Potential evaluation areas:

- correct input handling;
- preprocessing;
- model inference;
- subject-level aggregation;
- result presentation;
- error handling;
- reproducibility.

Exact requirements will be defined after the ML pipeline is stable.
