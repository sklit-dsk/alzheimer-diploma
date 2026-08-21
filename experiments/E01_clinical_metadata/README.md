# E01 - Clinical Metadata and Confounder Analysis

## Status

Completed.

The experiment has been executed successfully and the confirmed outputs are saved in `results/`.

## Objective

Connect ImagesOASIS binary-class subjects with official OASIS-1 clinical metadata and analyze possible confounding factors before constructing the corrected cohort in E02.

The primary binary classes are:

- Non Demented
- Very mild Dementia

## Input Files

- `experiments/E00_baseline_rf/dataset_metadata.csv`
  - Columns: `path`, `class`, `subject_id`, `scan_id`, `filename`
- `data/metadata/oasis_cross-sectional-5708aa0a98d82080.xlsx`
  - Relevant clinical columns: `ID`, `M/F`, `Age`, `MMSE`, `CDR`

E00 was used only as an input source. It was not modified.

## Methodology

The script:

1. loads the E00 ImagesOASIS metadata;
2. filters to the two binary classes;
3. loads the official OASIS-1 clinical metadata;
4. performs scan-level matching using `scan_id` from ImagesOASIS and `ID` from OASIS-1;
5. validates scan and subject matching coverage;
6. creates one scan-level row per unique scan;
7. creates one subject-level row per subject;
8. validates repeated clinical values before resolving them;
9. reports conflicting values explicitly instead of silently choosing one;
10. keeps missing CDR values missing;
11. summarizes missing clinical metadata, age, sex, CDR and MMSE;
12. compares age distributions using descriptive statistics and a Mann-Whitney U test;
13. evaluates an exploratory Age >= 60 cohort;
14. produces age-overlap tables for judging whether approximate age matching may be feasible in E02.

## Generated Outputs

The script writes outputs to `experiments/E01_clinical_metadata/results/`:

- `scan_level_metadata.csv`
- `subject_level_metadata.csv`
- `missing_values.csv`
- `age_summary.csv`
- `age_comparison.csv`
- `age_distribution.png`
- `sex_distribution.csv`
- `sex_distribution.png`
- `cdr_distribution.csv`
- `cdr_consistency.csv`
- `mmse_summary.csv`
- `age_60plus_summary.csv`
- `age_60plus_comparison.csv`
- `age_matching_exact_age_counts.csv`
- `age_matching_5year_bins.csv`
- `unmatched_scans.csv`
- `unmatched_subjects.csv`
- `imagesoasis_metadata_conflicts.csv`
- `clinical_value_conflicts.csv`
- `subject_level_clinical_conflicts.csv`
- `e01_summary.json`

## Confirmed Results

The binary ImagesOASIS cohort contains 324 unique subjects and 343 unique scans:

| Class | Subjects |
|---|---:|
| Non Demented | 266 |
| Very mild Dementia | 58 |

Clinical metadata matching coverage was complete:

| Unit | Matched | Total | Coverage |
|---|---:|---:|---:|
| Scans | 343 | 343 | 100.0% |
| Subjects with all scans matched | 324 | 324 | 100.0% |

Subject-level missingness:

| Variable | Available | Missing | Missing % |
|---|---:|---:|---:|
| Age | 324 | 0 | 0.0 |
| Sex | 324 | 0 | 0.0 |
| CDR | 177 | 147 | 45.3704 |
| MMSE | 177 | 147 | 45.3704 |

No clinical value conflicts were detected during scan-level or subject-level resolution.

Age statistics in the original binary groups:

| Class | n | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Non Demented | 266 | 45.7819548872 | 43.5 | 18.0 | 94.0 |
| Very mild Dementia | 58 | 75.9655172414 | 77.0 | 62.0 | 90.0 |

The original groups show a strong age imbalance:

- mean difference, Very mild Dementia minus Non Demented: 30.1835623542 years;
- median difference: 33.5 years;
- Mann-Whitney U: 12923.0;
- two-sided p-value: 7.552687005154834e-16;
- Cliff's delta: 0.6752657506.

Exploratory Age >= 60 cohort:

| Class | Original subjects | Retained subjects | Retained % |
|---|---:|---:|---:|
| Non Demented | 266 | 85 | 31.9549 |
| Very mild Dementia | 58 | 58 | 100.0 |

Age comparison within the Age >= 60 cohort:

- mean difference, Very mild Dementia minus Non Demented: approximately -0.45 years;
- median difference: 3.0 years;
- Mann-Whitney U: 2425.0;
- two-sided p-value: approximately 0.871;
- Cliff's delta: -0.0162271805.

This does not prove that the age distributions are identical. However, compared with the original binary groups, the severe age imbalance is substantially reduced.

## Methodological Conclusion

Age is an important potential confounding factor in the original binary ImagesOASIS groups.

Age >= 60 is selected as the primary cohort strategy to be implemented and formally validated in E02. E01 does not create the E02 dataset and does not perform model training on this cohort.

Approximate age matching remains a possible sensitivity or additional analysis rather than the primary strategy, because it would reduce the available sample further.

## Important Interpretation Rules

- Missing CDR must not be interpreted as `CDR = 0`.
- Statistical significance must not be interpreted as clinical importance.
- The Age >= 60 strategy must be implemented and validated in E02 before being used for corrected modeling.
- Approximate age matching is assessed only for feasibility in E01.
- Model outputs and metadata summaries must not be described as clinical diagnosis or clinical validation.

## How to Run

From the repository root:

```powershell
python experiments/E01_clinical_metadata/analyze_clinical_metadata.py
```

If `python` is not available on PATH, run the script with the project virtual environment or another Python interpreter that has the dependencies from `requirements.txt` installed.
