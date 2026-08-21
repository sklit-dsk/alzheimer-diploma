# E05 - Subject-Level Threshold Analysis

## Status

Completed.

`analyze_thresholds.py` was executed successfully and the confirmed outputs are saved in `results/`.

## Objective

Analyze subject-level sensitivity and specificity across multiple classification thresholds for the Random Forest baseline evaluated in E04.

E05 is descriptive threshold analysis. It does not retrain the Random Forest, modify E03 predictions, change E04 aggregation, or select a final validated threshold.

## Relationship To E04

E04 converted image/slice-level E03 predictions into exactly one row per subject.

E05 uses the E04 subject-level prediction table:

- `experiments/E04_subject_level/results/subject_level_predictions.csv`

The confirmed E04 test cohort contains:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

Very mild Dementia is treated as the positive class.

## Score Used

The primary E05 score is:

- `mean_probability`

Mean probability is used because E04 selected it as the primary working aggregation for threshold analysis. It provides a direct continuous subject-level score and can later be applied consistently to CNN outputs.

This is a methodological working choice, not proof that mean aggregation is clinically optimal.

## Input Validation

The script validates that:

- there are exactly 29 subject rows;
- there are exactly 29 unique `subject_id` values;
- the test set contains 17 Non Demented and 12 Very mild Dementia subjects;
- Non Demented maps to label `0`;
- Very mild Dementia maps to label `1`;
- every subject has one true label;
- `mean_probability` has no missing values;
- all `mean_probability` values are within `[0, 1]`.

The script stops with a clear error if validation fails.

## Threshold Grid

E05 evaluates the following predefined thresholds:

`0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90`

Prediction rule:

- Very mild Dementia if `mean_probability >= threshold`;
- Non Demented otherwise.

The threshold `0.50` is treated explicitly as the E04 default reference.

## Metrics

For every threshold, E05 calculates:

- TN;
- FP;
- FN;
- TP;
- accuracy;
- balanced accuracy;
- sensitivity for Very mild Dementia;
- specificity for Non Demented;
- precision for Very mild Dementia;
- F1 for Very mild Dementia;
- macro F1.

The script validates that threshold `0.50` reproduces the confirmed E04 mean-probability default-threshold metrics.

## ROC Analysis

ROC analysis uses the continuous subject-level `mean_probability` score.

ROC-AUC is calculated once from the continuous score, not separately for each binary threshold.

The script saves the ROC curve points and a ROC figure.

## Precision-Recall Analysis

Precision-recall analysis also uses the continuous subject-level `mean_probability` score.

Average precision / PR-AUC is calculated consistently with E04.

The script saves the precision-recall curve points and a figure.

## Sensitivity-Specificity Analysis

The main supervisor-requested output is the sensitivity-specificity tradeoff across thresholds.

The script saves:

- a threshold table with sensitivity, specificity, balanced accuracy, precision and F1;
- a figure showing sensitivity and specificity as functions of the threshold.

## Descriptive Operating Points

E05 identifies descriptive operating points on the predefined threshold grid:

- the default threshold `0.50`;
- the highest sensitivity observed in the grid;
- the highest balanced accuracy observed in the grid;
- threshold(s) where sensitivity and specificity are closest;
- the lowest grid threshold reaching sensitivity at least `0.70`, if present;
- the lowest grid threshold reaching sensitivity at least `0.80`, if present;
- the highest grid threshold still reaching sensitivity at least `0.70`, if present;
- the highest grid threshold still reaching sensitivity at least `0.80`, if present.

The highest threshold satisfying a sensitivity target can be more informative than the lowest threshold satisfying that target. Lower thresholds generally increase sensitivity, but they can also severely reduce specificity. Reporting the highest threshold that still meets a sensitivity target helps describe the sensitivity-specificity tradeoff without selecting an optimal, recommended, validated, or final threshold.

These are not optimal thresholds, validated thresholds, clinically recommended thresholds, or final thresholds.

## Generated Outputs

The script wrote outputs to `experiments/E05_threshold_analysis/results/`:

- `threshold_metrics.csv`
- `roc_curve_points.csv`
- `roc_curve.png`
- `pr_curve_points.csv`
- `precision_recall_curve.png`
- `sensitivity_specificity_by_threshold.png`
- `sensitivity_specificity_table.csv`
- `subject_probability_distribution.png`
- `descriptive_operating_points.csv`
- `e05_summary.json`

## Confirmed Results

E05 evaluated 29 subject-level E04 predictions:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

The default threshold `0.50` reproduced the confirmed E04 mean-probability result:

| Metric | Value |
|---|---:|
| Sensitivity | 0.4166666666666667 |
| Specificity | 0.7647058823529411 |
| Balanced accuracy | 0.5906862745098039 |
| Precision | 0.5555555555555556 |
| F1 positive | 0.47619047619047616 |

Confusion matrix at threshold `0.50`:

| Actual / Predicted | Non Demented | Very mild Dementia |
|---|---:|---:|
| Non Demented | 13 | 4 |
| Very mild Dementia | 7 | 5 |

Continuous-score metrics:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6568627450980392 |
| PR-AUC / average precision | 0.5628292497835842 |

Descriptive operating points from the predefined grid:

| Operating point | Threshold | Sensitivity | Specificity | Balanced accuracy | Precision | F1 positive |
|---|---:|---:|---:|---:|---:|---:|
| Highest balanced accuracy in grid | 0.4 | 0.9166666666666666 | 0.47058823529411764 | 0.6936274509803921 | 0.55 | 0.6875 |
| Sensitivity/specificity closest in grid | 0.45 | 0.5833333333333334 | 0.5882352941176471 | 0.5857843137254902 | 0.5 | 0.5384615384615384 |
| Highest threshold with sensitivity at least 0.70 | 0.4 | 0.9166666666666666 | 0.47058823529411764 | 0.6936274509803921 | 0.55 | 0.6875 |
| Highest threshold with sensitivity at least 0.80 | 0.4 | 0.9166666666666666 | 0.47058823529411764 | 0.6936274509803921 | 0.55 | 0.6875 |

The lowest threshold with sensitivity at least `0.70` and at least `0.80` was `0.10`, with sensitivity `1.0` and specificity `0.0`.

No final, optimal, clinically recommended, or independently validated threshold was selected.

## Methodological Interpretation

E05 uses the existing 29-subject E03/E04 test set to describe threshold behavior.

Therefore E05 must not be interpreted as validated threshold optimization. The results must not be used to tune the Random Forest, replace the default threshold as a final deployment threshold, or claim that a selected threshold will generalize to unseen subjects.

A validated threshold-selection procedure would require an independent validation cohort or nested cross-validation.

The probability score should be interpreted as a model score for the research classification task, not as a clinical diagnostic probability.

## Limitations

The test set contains only 29 subjects, so threshold-dependent metrics can be unstable.

No final threshold is selected in E05. Threshold behavior is documented to support transparent interpretation and to guide later methodology.

## How To Run

From the repository root:

```powershell
python experiments/E05_threshold_analysis/analyze_thresholds.py
```
