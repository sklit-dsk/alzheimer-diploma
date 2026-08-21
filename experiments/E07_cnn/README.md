# E07 - Transfer-Learning CNN With ResNet18

## Status

Implemented, not yet confirmed.

`train_resnet18.py` has been updated for Attempt 2 staged transfer learning. E07 must not be marked completed until training is executed successfully and the generated outputs are reviewed.

## Objective

Train the main image-based transfer-learning CNN for the diploma project using ResNet18 with ImageNet pretrained weights.

The model is a research classification model for distinguishing:

- Non Demented
- Very mild Dementia

It is not a clinically validated diagnostic system.

## Relationship To E02

E07 uses the E02 Age >= 60 cohort as the downstream modeling cohort.

The cohort was selected because E01 showed strong original age imbalance and E02 implemented the primary age-control strategy.

## Relationship To E03

E03 provides the authoritative subject split for comparability with the corrected Random Forest baseline.

Frozen E03 test subjects are loaded from:

- `experiments/E03_clean_rf/results/test_subjects.csv`

These subjects must not be used for training, validation, early stopping, model selection, hyperparameter tuning, augmentation decisions, or threshold selection.

E03 train subjects are loaded from:

- `experiments/E03_clean_rf/results/train_subjects.csv`

Only those E03 train subjects are split internally into CNN train and CNN validation subjects.

Image metadata is loaded from:

- `experiments/E03_clean_rf/results/selected_image_metadata.csv`

E07 ignores E03 `images_after_balancing` values and uses all available slices for each selected subject.

## Attempt 1

Attempt 1 used immediate full fine-tuning of ResNet18.

It was started but not completed:

- epochs 1-3 completed;
- training was manually interrupted during epoch 4;
- frozen test inference was not performed.

The partial evidence is preserved in:

- `attempts/attempt_01_full_finetune/`

The archive includes the saved training configuration, split files, image metadata files, validation epoch prediction files, the partial best checkpoint, and a small reconstructed validation history based only on confirmed terminal values.

Attempt 1 is an incomplete development attempt, not a final E07 result.

The validation behavior suggested rapid overfitting or unstable generalization: train loss decreased quickly while validation loss remained high and increased, and validation subject ROC-AUC decreased after the first epoch.

This observation motivated a more conservative Attempt 2 strategy. The adaptation was based only on validation behavior. Frozen E03 test data remained untouched.

## Attempt 2

Attempt 2 uses staged transfer learning.

Attempt 2 training completed all configured epochs successfully. The post-training step failed when loading the locally created checkpoint because PyTorch 2.6+ defaults `torch.load` to `weights_only=True`.

No frozen-test inference occurred before that failure.

This was a technical checkpoint-loading issue after training, and it does not invalidate the completed training history or saved best checkpoint.

The script now supports postprocess-only recovery mode to finish the same trained experiment without retraining.

The following remain unchanged from the original E07 design:

- E03 frozen test subjects;
- E03 train subjects;
- CNN train/validation subject split;
- seed `42`;
- `224 x 224` inputs;
- preprocessing;
- conservative augmentation;
- class weights based on CNN train subject counts;
- batch size `16`;
- ResNet18 ImageNet pretrained weights;
- validation subject-level mean aggregation;
- checkpoint selection using validation subject ROC-AUC;
- frozen test policy.

Attempt 2 output files use the `attempt2` suffix where needed so they cannot be confused with Attempt 1 evidence.

## Postprocess-Only Recovery

Use postprocess-only mode when Attempt 2 training has already completed and the existing checkpoint/history should be reused:

```powershell
python experiments/E07_cnn/train_resnet18.py --postprocess-only
```

This mode:

- does not run training;
- requires `artifacts/best_resnet18_attempt2.pt`;
- requires `results/training_history_attempt2.csv`;
- reconstructs the same subjects, image metadata, transforms, loaders, class weights and criterion;
- loads the locally generated Attempt 2 checkpoint;
- verifies checkpoint metadata against the saved training history;
- runs best-checkpoint validation inference first;
- only then runs frozen-test slice inference once;
- saves Attempt 2 validation/test prediction outputs and `e07_summary_attempt2.json`.

The locally generated checkpoint contains optimizer/config metadata in addition to tensors, so the script loads it with `weights_only=False`. This is used only for this project's own checkpoint, not arbitrary external checkpoints.

## Train / Validation / Test Split

The frozen E03 test split contains:

| Class | Subjects |
|---|---:|
| Non Demented | 17 |
| Very mild Dementia | 12 |
| Total | 29 |

The E03 train split contains:

| Class | Subjects |
|---|---:|
| Non Demented | 68 |
| Very mild Dementia | 46 |
| Total | 114 |

E07 splits only the 114 E03 train subjects into CNN train and CNN validation using a deterministic subject-level stratified split with seed `42` and an approximately 80/20 ratio.

Expected deterministic counts:

| Split | Non Demented | Very mild Dementia | Total |
|---|---:|---:|---:|
| CNN train | 54 | 37 | 91 |
| CNN validation | 14 | 9 | 23 |

The script explicitly checks that there is zero subject overlap among train, validation and frozen test sets.

## Image Preprocessing

Input size is `224 x 224`.

MRI JPG images are loaded as grayscale and converted to three channels by repeating the grayscale channel.

ImageNet normalization is applied using:

- mean: `[0.485, 0.456, 0.406]`
- standard deviation: `[0.229, 0.224, 0.225]`

Validation and test transforms are deterministic.

## Training Augmentation

Training augmentation is intentionally conservative:

- small rotation, approximately `+/- 5` degrees;
- small translation, approximately `2%`;
- small scale variation from `0.98` to `1.02`.

E07 does not use vertical flipping, horizontal flipping, aggressive cropping, large rotations, or augmentation that changes image semantics substantially.

## Model

E07 uses:

- `torchvision.models.resnet18`
- `ResNet18_Weights.IMAGENET1K_V1`

The final classification layer is replaced for two classes:

- `0` = Non Demented
- `1` = Very mild Dementia

The pretrained network is fine-tuned.

## Attempt 2 Training Configuration

Phase 1 trains only the classifier head:

- seed: `42`
- batch size: `16`
- optimizer: AdamW
- phase 1 epochs: `4`
- phase 1 trainable layers: `fc`
- phase 1 learning rate: `1e-3`
- weight decay: `1e-4`
- loss: CrossEntropyLoss
- default threshold for descriptive validation metrics: `0.5`

Phase 2 conservatively fine-tunes only `layer4` and `fc`:

- optimizer: new AdamW optimizer over trainable parameters only;
- phase 2 maximum epochs: `8`;
- phase 2 learning rate: `1e-5`;
- weight decay: `1e-4`;
- early stopping patience: `3`.

Class imbalance is handled using class weights based on CNN train subject counts only, not validation or test data.

Mixed precision CUDA training is used when CUDA is available. The script includes a CPU fallback.

For CUDA reproducibility, `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before importing torch, seeds are fixed, and deterministic settings are enabled where practical. Absolute bit-for-bit reproducibility is not promised across all CUDA environments.

## Checkpoint Selection

After each epoch, E07 calculates:

- train loss;
- validation loss;
- validation subject-level ROC-AUC;
- validation subject-level PR-AUC / average precision;
- validation subject-level accuracy at threshold `0.5`;
- validation subject-level balanced accuracy at threshold `0.5`;
- validation subject-level sensitivity at threshold `0.5`;
- validation subject-level specificity at threshold `0.5`.

Validation slice probabilities are aggregated by subject using mean probability.

The primary checkpoint-selection metric is validation subject-level ROC-AUC across both Attempt 2 phases. Validation loss is used only as a deterministic tie-breaker.

E07 does not optimize a classification threshold.

## Test Inference

Only after training is complete and the best checkpoint is selected, E07 runs inference once on the frozen E03 test subjects.

E07 saves slice-level CNN test predictions, but it does not calculate final CNN subject-level test metrics.

Final CNN subject-level aggregation and evaluation belong to E08.

## Generated Outputs

Attempt 2 artifacts:

- `artifacts/best_resnet18_attempt2.pt`

Attempt 2 results:

- `train_subjects_attempt2.csv`
- `validation_subjects_attempt2.csv`
- `test_subjects_attempt2.csv`
- `split_summary_attempt2.csv`
- `train_image_metadata_attempt2.csv`
- `validation_image_metadata_attempt2.csv`
- `test_image_metadata_attempt2.csv`
- `training_config_attempt2.json`
- `training_history_attempt2.csv`
- `validation_slice_predictions_attempt2.csv`
- `validation_subject_predictions_attempt2.csv`
- `test_slice_predictions_attempt2.csv`
- `e07_summary_attempt2.json`
- `training_validation_loss_attempt2.png`
- `validation_subject_roc_auc_by_epoch_attempt2.png`
- `validation_subject_sensitivity_specificity_by_epoch_attempt2.png`
- `validation_epoch_predictions_attempt2/`

## Methodological Limitations

E07 does not claim CNN superiority over the Random Forest baseline.

E07 does not perform final CNN subject-level test evaluation.

The frozen test set is small and must remain untouched for training decisions.

The model output must be described as a research classification output, not as a medical diagnosis.

## How To Run

From the repository root:

```powershell
python experiments/E07_cnn/train_resnet18.py
```

To finish post-training inference from the already completed Attempt 2 checkpoint without retraining:

```powershell
python experiments/E07_cnn/train_resnet18.py --postprocess-only
```
