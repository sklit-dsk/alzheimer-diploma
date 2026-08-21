# E02 - Cohort Construction and Age Control

## Status

Completed.

`build_age_controlled_cohort.py` was executed successfully and the confirmed outputs are saved in `results/`.

## Objective

Construct and validate the age-controlled cohort selected after E01 for the primary binary classification scenario:

- Non Demented
- Very mild Dementia

E02 prepares subject-level and scan-level metadata for later corrected modeling. It does not train or evaluate a model.

## Why Age >= 60 Was Selected

E01 confirmed a strong age imbalance in the original binary groups. The Very mild Dementia subjects were much older on average than the Non Demented subjects.

The exploratory Age >= 60 cohort retained:

- 85 Non Demented subjects;
- 58 Very mild Dementia subjects.

Within this exploratory 60+ cohort, the mean age difference was approximately -0.45 years and the two-sided Mann-Whitney U p-value was approximately 0.871. This does not prove that the age distributions are identical, but it substantially reduces the severe original age imbalance.

Approximate age matching remains optional as a sensitivity or additional analysis because it would reduce the available sample further.

## Input Files

- `experiments/E01_clinical_metadata/results/subject_level_metadata.csv`
- `experiments/E01_clinical_metadata/results/scan_level_metadata.csv`

E01 historical results are used as inputs and must not be modified by E02.

## Inclusion Criteria

- Subject belongs to one of the binary classes:
  - Non Demented
  - Very mild Dementia
- Subject has `Age >= 60`.
- Subject has non-missing Age and Sex.
- Subject has at least one scan in the E01 scan-level metadata.

## Exclusion Criteria

- Subject is outside the primary binary classes.
- Subject has `Age < 60`.
- Subject has missing Age or Sex.
- Subject has inconsistent class or clinical metadata.
- Subject has a CDR value inconsistent with the expected class mapping where CDR is available.

Missing CDR remains missing and is not interpreted as `CDR = 0`.

## Expected Cohort Size

The script validates the E02 subject counts against the E01-confirmed values:

| Class | Expected subjects |
|---|---:|
| Non Demented | 85 |
| Very mild Dementia | 58 |
| Total | 143 |

If these counts differ, the script stops with a clear error.

## Generated Outputs

The script writes outputs to `experiments/E02_age_control/results/`:

- `e02_subject_level_cohort.csv`
- `e02_scan_level_cohort.csv`
- `e02_age_summary.csv`
- `e02_age_comparison.csv`
- `e02_sex_distribution.csv`
- `e02_cdr_distribution.csv`
- `e02_cdr_consistency.csv`
- `e02_mmse_summary.csv`
- `e02_missing_values.csv`
- `e02_retained_percentages.csv`
- `e02_summary.json`

## Confirmed Results

The E02 Age >= 60 cohort contains 143 subjects:

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

## Methodological Interpretation

E02 implements the primary age-control strategy selected from E01. The purpose is to construct a cleaner cohort for the corrected Random Forest baseline in E03 and later comparable CNN experiments.

The Age >= 60 restriction reduces the strongest observed age imbalance, but it does not prove that the groups are clinically identical or free from all confounding. The cohort should still be described cautiously as an age-controlled research cohort.

Age >= 60 is the selected primary cohort for downstream experiments. Approximate age matching remains optional as a sensitivity or additional analysis.

## How to Run

From the repository root:

```powershell
python experiments/E02_age_control/build_age_controlled_cohort.py
```

The expected terminal output should report 143 total subjects, with 85 Non Demented and 58 Very mild Dementia subjects.
