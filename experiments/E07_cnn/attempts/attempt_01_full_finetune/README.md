# E07 Attempt 1 - Immediate Full Fine-Tuning

## Status

Incomplete development attempt.

Attempt 1 was started but not completed. It must not be reported as a final E07 result.

## Configuration

Attempt 1 used the initial E07 implementation:

- ResNet18 with ImageNet pretrained weights;
- all pretrained layers fine-tuned immediately;
- AdamW optimizer;
- learning rate `1e-4`;
- weight decay `1e-4`;
- batch size `16`;
- maximum epochs `15`;
- seed `42`;
- E03 frozen test subjects preserved;
- CNN train/validation split created only from E03 train subjects;
- validation subject-level mean-probability aggregation;
- checkpoint selection by validation subject ROC-AUC.

The saved runtime configuration is archived at:

- `results/training_config.json`

## Preserved Evidence

This archive contains copies of the partial Attempt 1 evidence:

- `results/training_config.json`
- `results/split_summary.csv`
- `results/train_subjects.csv`
- `results/validation_subjects.csv`
- `results/test_subjects.csv`
- `results/train_image_metadata.csv`
- `results/validation_image_metadata.csv`
- `results/test_image_metadata.csv`
- `results/validation_epoch_predictions/`
- `artifacts/best_resnet18.pt`
- `validation_history_reconstructed.csv`

The original partial files are not deleted.

## Confirmed Progress

Epochs 1-3 completed.

Training was manually interrupted during epoch 4.

Frozen test inference was not performed.

## Reconstructed Validation History

`validation_history_reconstructed.csv` records only the confirmed terminal values supplied after Attempt 1. It is not a full `training_history.csv` and should not be treated as a complete run log.

## Interpretation

Attempt 1 suggested rapid overfitting or unstable generalization:

- train loss decreased quickly;
- validation loss remained high and increased;
- validation subject ROC-AUC decreased after epoch 1.

This observation motivated Attempt 2 with staged transfer learning:

1. classifier-head-only training;
2. conservative fine-tuning of `layer4` and `fc`.

No final E07 result, frozen-test inference result, CNN superiority claim, or clinical interpretation is made from Attempt 1.
