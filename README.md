# Alzheimer Early Detection — Diploma Project

Practical and experimental repository for the diploma thesis:

**Informacioni sistem za rano otkrivanje Alchajmerove bolesti zasnovan na metodama mašinskog učenja**

University: Univerzitet Donja Gorica (UDG)  
Faculty: Fakultet za informacione sisteme i tehnologije (FIST)

---

## Project status

Current status:

**E01-E09 are completed.**

The original Random Forest baseline has been preserved as:

**E00 — Original Random Forest baseline**

The main ML experimental pipeline now includes clinical metadata analysis, age-control cohort construction, corrected Random Forest evaluation, threshold analysis, corrected Feature Importance, transfer-learning CNN evaluation, and RF-vs-CNN comparison.

Current priority:

**implementation of the information-system prototype and E11 final information-system evaluation.**

E10 Explainability remains explicitly optional and should not delay the first complete diploma version.

---

## Research objective

The project investigates machine-learning methods for distinguishing:

- Non Demented subjects
- Very mild Dementia subjects

using brain MRI images.

The research focuses particularly on:

- early-stage classification;
- control of possible confounding factors;
- subject-level data separation;
- subject-level evaluation;
- sensitivity to Very mild Dementia;
- comparison of traditional machine learning and deep learning approaches.

The final result is intended to be a research information-system prototype.

It is not intended to provide a medical diagnosis or replace clinical assessment.

---

## Dataset

The project uses the ImagesOASIS dataset derived from OASIS MRI data.

The dataset contains multiple MRI slices per subject.

Subject identity must therefore be preserved throughout the experimental pipeline.

MRI slices belonging to the same subject must not be split between training and test sets.

Large source datasets are stored locally and are not committed to Git.

---

## Experiment roadmap

| ID | Experiment |
|---|---|
| E00 | Original Random Forest baseline |
| E01 | Clinical metadata and confounder analysis |
| E02 | Cohort construction / age control |
| E03 | Clean Random Forest baseline |
| E04 | Subject-level aggregation |
| E05 | Threshold and sensitivity-specificity analysis |
| E06 | Corrected Feature Importance |
| E07 | Transfer-learning CNN |
| E08 | CNN subject-level evaluation |
| E09 | Random Forest vs CNN comparison |
| E10 | Explainability — optional |
| E11 | Final information-system evaluation |

Detailed experiment status is maintained in:

`docs/experiments.md`

---

## Repository structure

```text
alzheimer-diploma/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── requirements-lock.txt
│
├── data/
│   ├── raw/
│   ├── metadata/
│   ├── processed/
│   └── README.md
│
├── docs/
│   ├── project.md
│   ├── methodology.md
│   └── experiments.md
│
├── experiments/
│   ├── E00_baseline_rf/
│   └── E01_clinical_metadata/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── tables/
│
├── src/
└── tests/
