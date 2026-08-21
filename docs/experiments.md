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
| E02 | Cohort construction / age control | Planned |
| E03 | Clean Random Forest baseline | Planned |
| E04 | Subject-level aggregation | Planned |
| E05 | Threshold and sensitivity-specificity analysis | Planned |
| E06 | Corrected Feature Importance | Planned |
| E07 | Transfer-learning CNN | Planned |
| E08 | CNN subject-level evaluation | Planned |
| E09 | Random Forest vs CNN comparison | Planned |
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

Planned.

## Objective

Construct a methodologically cleaner Non Demented vs Very mild Dementia cohort based on the findings from E01.

The primary strategy selected from E01 is to restrict the cohort to subjects aged 60+.

E02 must implement and formally validate this cohort before it is used for corrected modeling.

Approximate age matching remains a possible sensitivity or additional analysis, but it is not the primary strategy because it would reduce the available sample further.

---

# E03 — Clean Random Forest baseline

## Status

Planned.

## Objective

Repeat the Random Forest baseline using the cohort selected in E02.

The experiment should preserve subject-level separation and provide a clean baseline for comparison with later models.

---

# E04 — Subject-level aggregation

## Status

Planned.

## Objective

Convert slice-level model predictions into one final prediction per subject.

Candidate aggregation methods:

- mean probability;
- median probability;
- majority voting.

The aggregation strategies should be compared before selecting the primary method.

---

# E05 — Threshold and sensitivity-specificity analysis

## Status

Planned.

## Objective

Analyze model behavior at different classification thresholds.

Special attention should be paid to:

- sensitivity for Very mild Dementia;
- specificity;
- trade-off between sensitivity and specificity.

The final interpretation should not depend only on the default threshold of 0.5.

---

# E06 — Corrected Feature Importance

## Status

Planned.

## Objective

Correct grouped Feature Importance analysis.

For each feature group, record:

- number of features;
- total importance;
- normalized or mean importance per feature.

Interpret the results as image characteristics used by the model, not as confirmed medical biomarkers.

---

# E07 — Transfer-learning CNN

## Status

Planned.

## Objective

Develop the main image-based model after E01-E06 have addressed the methodological issues identified by the supervisor.

The exact CNN architecture has not yet been selected.

The model should use the same clean cohort and subject separation principles used for the corrected baseline.

---

# E08 — CNN subject-level evaluation

## Status

Planned.

## Objective

Evaluate CNN predictions at subject level using the same methodological principles and metrics used for the Random Forest model.

---

# E09 — Random Forest vs CNN comparison

## Status

Planned.

## Objective

Compare the traditional handcrafted-feature Random Forest approach with the image-based CNN approach.

The comparison should focus not only on overall accuracy but also on:

- sensitivity;
- specificity;
- F1;
- ROC-AUC;
- PR-AUC;
- subject-level performance.

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
