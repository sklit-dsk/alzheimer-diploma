# E04 - Subject-Level Aggregation

## Status

Completed.

`aggregate_subject_predictions.py` was executed successfully and the confirmed outputs are saved in `results/`.

## Objective

Aggregate E03 image/slice-level Random Forest predictions into one prediction per subject and evaluate subject-level performance.

E04 does not retrain the Random Forest. It uses the E03 test-set prediction file as input.

## Why Subject-Level Aggregation Is Required

The ImagesOASIS dataset contains multiple MRI slices per subject. Treating slices as independent final observations can overstate the amount of independent evidence in the test set.

The main evaluation unit for the diploma is therefore the subject. E04 converts E03 image-level predictions into exactly one row per test subject before calculating metrics.

## Relationship To E03

E03 produced image/slice-level predictions for the balanced test images:

- `experiments/E03_clean_rf/results/image_level_predictions.csv`

E04 uses this file and preserves the E03 labels and probabilities. It does not change the trained model, image features, train/test split, or image-level predictions.

The confirmed E03 test split contains 29 subjects:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

Very mild Dementia is treated as the positive class.

## Aggregation Strategies

E04 evaluates three default-threshold aggregation strategies:

1. Mean probability: predicted positive when `mean_probability >= 0.5`.
2. Median probability: predicted positive when `median_probability >= 0.5`.
3. Majority voting: predicted positive when `positive_vote_fraction > 0.5`.

Exact majority-vote ties are explicitly flagged. Under the strict `> 0.5` rule, ties are assigned to Non Demented and documented rather than silently treated as positive.

## Default Threshold

E04 uses only the default `0.5` threshold.

Threshold optimization and sensitivity-specificity trade-off analysis belong to E05.

## Evaluation Metrics

For each aggregation method, E04 calculates subject-level:

- accuracy;
- balanced accuracy;
- sensitivity / recall for Very mild Dementia;
- specificity for Non Demented;
- precision for Very mild Dementia;
- F1 for Very mild Dementia;
- macro F1;
- confusion matrix.

For mean probability and median probability, ROC-AUC and PR-AUC use the corresponding continuous probability score.

For majority voting, ROC-AUC and PR-AUC use `positive_vote_fraction` as the continuous score and are labeled accordingly in the metrics files.

## Generated Outputs

The script writes outputs to `experiments/E04_subject_level/results/`:

- `subject_level_predictions.csv`
- `mean_probability_metrics.json`
- `median_probability_metrics.json`
- `majority_vote_metrics.json`
- `mean_probability_confusion_matrix.csv`
- `median_probability_confusion_matrix.csv`
- `majority_vote_confusion_matrix.csv`
- `mean_probability_confusion_matrix.png`
- `median_probability_confusion_matrix.png`
- `majority_vote_confusion_matrix.png`
- `aggregation_comparison.csv`
- `e04_summary.json`

## Confirmed Results

The subject-level E04 test set contains 29 subjects:

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

One subject had an exact majority-vote tie. The documented tie rule assigns exact ties to Non Demented.

## Methodological Limitations

The E04 test set contains only 29 subjects, so subject-level metrics can be unstable.

E04 should not be treated as final model optimization. It compares aggregation methods at the default threshold only. E05 is required before discussing threshold-dependent sensitivity and specificity.

Sensitivity at the default threshold remains limited at approximately 41.7%.

Mean probability and majority voting give identical default-threshold classifications in this run, while their continuous scores produce different ROC-AUC values. Median probability also has a slightly different ROC-AUC and PR-AUC.

No aggregation method should be declared universally superior from this small test set.

Mean probability is the primary working probability aggregation for E05 because it provides a direct continuous subject-level probability and can later be applied consistently to CNN outputs. This is a methodological working choice, not proof that mean aggregation is clinically optimal.

Threshold behavior must be analyzed in E05. Do not claim final model performance until threshold analysis is documented.

## How To Run

From the repository root:

```powershell
python experiments/E04_subject_level/aggregate_subject_predictions.py
```
