# E08 - CNN Subject-Level Evaluation

## Status

Implemented, not yet confirmed.

E08 must not be marked completed until `aggregate_cnn_subject_predictions.py` has been executed successfully and the generated outputs have been reviewed.

## Objective

Evaluate the E07 Attempt 2 CNN predictions at subject level using the frozen test split.

E08 converts slice-level CNN predictions into one row per subject and calculates subject-level metrics. The model output remains a research classification output, not a medical diagnosis.

## Relationship To E07

E07 trains the ResNet18 CNN and saves frozen-test slice predictions from the selected Attempt 2 checkpoint.

E08 does not:

- retrain the CNN;
- modify the CNN checkpoint;
- choose a different checkpoint;
- optimize thresholds;
- tune hyperparameters;
- use test labels for training decisions.

The primary input is:

- `experiments/E07_cnn/results/test_slice_predictions_attempt2.csv`

## Frozen Test Set

The frozen test set is inherited from the E03 split and E07 Attempt 2 postprocessing.

Expected validation checks:

- 6,954 test slice predictions;
- 29 subjects;
- 17 Non Demented subjects;
- 12 Very mild Dementia subjects;
- one true class and one true label per subject;
- no missing probabilities;
- probabilities in `[0, 1]`;
- true label mapping: Non Demented = `0`, Very mild Dementia = `1`.

## Positive Class

Very mild Dementia is the positive class.

## Why Aggregation Is Required

The CNN produces predictions for individual MRI slices.

The diploma methodology requires final evaluation at subject level because multiple slices belong to the same subject. Treating slices as independent final observations would overstate the effective test sample size and would not represent the final subject-level classification workflow.

## Aggregation Strategies

E08 evaluates three aggregation methods:

- mean probability;
- median probability;
- majority voting.

Mean probability predicts Very mild Dementia when the subject-level mean probability is at least `0.5`.

Median probability predicts Very mild Dementia when the subject-level median probability is at least `0.5`.

Majority voting uses the stored slice-level `predicted_label_at_0_5` values. It predicts Very mild Dementia only when the positive vote fraction is greater than `0.5`.

Exact majority-vote ties are explicitly flagged and assigned to Non Demented using the same strict rule used in E04.

## Evaluation Metrics

For each aggregation method, E08 calculates:

- accuracy;
- balanced accuracy;
- sensitivity / recall for Very mild Dementia;
- specificity for Non Demented;
- precision for Very mild Dementia;
- F1 for Very mild Dementia;
- macro F1;
- ROC-AUC;
- PR-AUC / average precision;
- confusion matrix.

All metrics are calculated from exactly one row per subject.

## Threshold

The default threshold is `0.5`.

No threshold optimization is performed in E08. Any later CNN threshold trade-off analysis must be clearly separated from this default-threshold subject-level evaluation.

## Generated Outputs

The script writes:

- `results/subject_level_predictions.csv`;
- `results/mean_probability_metrics.json`;
- `results/median_probability_metrics.json`;
- `results/majority_vote_metrics.json`;
- `results/mean_probability_confusion_matrix.csv`;
- `results/median_probability_confusion_matrix.csv`;
- `results/majority_vote_confusion_matrix.csv`;
- `results/mean_probability_confusion_matrix.png`;
- `results/median_probability_confusion_matrix.png`;
- `results/majority_vote_confusion_matrix.png`;
- `results/aggregation_comparison.csv`;
- `results/e08_summary.json`.

## Methodological Limitations

The frozen test set contains only 29 subjects, so E08 results must be interpreted cautiously.

E08 should not be used to declare a universally superior aggregation method from one metric alone.

E08 does not establish clinical diagnostic validity.

## How To Run

From the repository root:

```powershell
python experiments/E08_cnn_subject_level/aggregate_cnn_subject_predictions.py
```
