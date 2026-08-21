# AGENTS.md

## Project

This repository contains the practical and experimental part of the diploma thesis:

"Informacioni sistem za rano otkrivanje Alchajmerove bolesti zasnovan na metodama mašinskog učenja"

University:
Univerzitet Donja Gorica (UDG)

Faculty:
Fakultet za informacione sisteme i tehnologije (FIST)

The project investigates machine-learning methods for early Alzheimer's disease detection using MRI images.

The primary classification scenario is:

Non Demented vs Very mild Dementia

The dataset currently used is ImagesOASIS, derived from OASIS MRI data.

---

## Main research principle

This is a scientific diploma project.

Never fabricate:

- experiment results;
- metrics;
- dataset properties;
- clinical conclusions;
- model performance;
- statistical results.

A planned experiment must never be described as completed.

A result may only be considered confirmed after the corresponding code has actually been executed and its output has been saved.

---

## Current research status

An initial Random Forest baseline has already been created using handcrafted image features.

The original experiment is called:

E00 — Original Random Forest baseline

The train/test split was performed at subject level so that MRI slices belonging to the same subject do not appear in both training and test sets.

The initial model showed relatively low recall for the Very mild Dementia class.

This baseline must be preserved for later comparison.

Do not overwrite E00 with results from newer experiments.

---

## Supervisor requirements

Before development of the main image-based CNN model, the following work must be completed.

### Clinical metadata

Connect subjects with available clinical information:

- Age
- Sex
- CDR
- MMSE, where available

### Confounding analysis

Investigate whether age differs systematically between:

- Non Demented
- Very mild Dementia

Age must be treated as a potential confounding factor.

Possible cohort strategies include:

- subjects aged 60+
- approximately age-matched groups

Do not choose a strategy before analyzing the actual data.

### Subject-level evaluation

Final evaluation must be performed at subject level.

MRI slice predictions should be aggregated using methods such as:

- mean probability
- median probability
- majority voting

Subject-level metrics should include where applicable:

- accuracy
- balanced accuracy
- sensitivity
- specificity
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

### Threshold analysis

Do not rely exclusively on a decision threshold of 0.5.

Analyze the relationship between sensitivity and specificity at different thresholds.

Sensitivity for the Very mild Dementia class is especially important because the research focuses on early detection.

### Feature Importance

Feature groups containing different numbers of individual features must not be compared only using summed importance.

Calculate normalized or mean importance per feature where appropriate.

Image characteristics used by the model must not automatically be described as medical biomarkers.

### Main image model

Only after the methodological issues above are addressed should the project proceed to an image-based model such as a transfer-learning CNN.

The final study should compare the CNN with the Random Forest baseline using the same methodological principles.

---

## Experiment roadmap

Experiments are identified as:

E00 — Original Random Forest baseline

E01 — Clinical metadata and confounder analysis

E02 — Cohort construction / age control

E03 — Clean Random Forest baseline

E04 — Subject-level aggregation

E05 — Threshold and sensitivity-specificity analysis

E06 — Corrected Feature Importance

E07 — Transfer-learning CNN

E08 — CNN subject-level evaluation

E09 — Random Forest vs CNN comparison

E10 — Explainability, if time permits

E11 — Final information-system evaluation

Do not silently change experiment definitions.

If methodology changes substantially, create a new experiment or document the change.

---

## Reproducibility

Every experiment should record, where applicable:

- experiment ID;
- date;
- objective;
- dataset;
- cohort;
- inclusion criteria;
- exclusion criteria;
- number of subjects;
- class distribution;
- preprocessing;
- train/test strategy;
- random seed;
- features;
- model;
- hyperparameters;
- aggregation strategy;
- classification threshold;
- metrics;
- generated figures;
- generated tables;
- conclusion.

Prefer deterministic behavior where technically possible.

Random seeds should be explicitly defined.

---

## Repository structure

The repository is organized as:

data/
    Dataset metadata and local data references.

docs/
    Project documentation, methodology and experiment log.

src/
    Reusable Python source code.

experiments/
    Experiment-specific scripts and configurations.

results/
    Generated experiment results.

tests/
    Automated tests for reusable project code.

Do not place reusable project logic only inside notebooks or experiment scripts if it belongs in src/.

---

## Data policy

Large MRI datasets should not be committed to Git.

Dataset files should remain local unless explicitly required otherwise.

Do not modify original raw data.

If transformed data is required, create derived files separately.

Maintain a clear distinction between:

raw data

and:

processed/derived data

Never overwrite raw source data during preprocessing.

---

## Coding rules

Use Python for the ML and data-analysis pipeline.

Prefer:

- pathlib for filesystem paths;
- type hints for reusable functions;
- small reusable functions;
- clear variable names;
- explicit configuration;
- reproducible random seeds.

Avoid:

- unnecessary global state;
- duplicated preprocessing logic;
- hardcoded absolute paths;
- silent exception handling;
- unnecessary dependencies.

Do not perform large refactors unless they are necessary for the current task.

Preserve working functionality.

---

## Working with the student

The student is learning while building the project.

When proposing implementation work:

1. Explain briefly what is being changed.
2. State which file should be created or modified.
3. Provide the exact command when a command is required.
4. Prefer complete file contents when modifying a small file.
5. Explain how to run the code.
6. Explain what output should be expected.
7. Wait for the actual output before treating the step as successful.

Do not assume a command succeeded without seeing its result.

When an error occurs, diagnose the current error before proposing unrelated changes.

---

## Medical interpretation

This project is a research prototype.

Do not claim that the system:

- diagnoses Alzheimer's disease;
- replaces a physician;
- is clinically validated;
- is ready for clinical use.

Prefer terminology such as:

- classification;
- prediction;
- research prototype;
- decision-support;
- model output;
- early detection research.

Clinical interpretation must remain appropriately cautious.

---

## Research question

Current working research question:

U kojoj mjeri metode mašinskog učenja zasnovane na MRI snimcima mogu razlikovati subjekte bez demencije od subjekata sa veoma blagim stadijumom demencije, uz kontrolu mogućih confounding faktora i evaluaciju na nivou subjekta?

---

## Working hypothesis

Current working hypothesis:

Primjenom metoda mašinskog učenja na MRI snimke mozga moguće je ostvariti razlikovanje subjekata bez demencije od subjekata sa veoma blagim stadijumom demencije, pri čemu kontrola sastava kohorte i evaluacija na nivou subjekta omogućavaju metodološki pouzdaniju procjenu performansi modela.

The hypothesis is provisional and may be refined based on experimental evidence and supervisor feedback.

---

## Current priority

The current experiment is:

E01 — Clinical metadata and confounder analysis

Do not begin CNN development before E01-E06 are addressed unless explicitly requested.

## Supervisor feedback policy

The latest written feedback from the scientific supervisor is treated as the final available methodological guidance for the current diploma development cycle.

Do not assume that additional supervisor feedback will be available before the first working version of the thesis.

The project must therefore proceed independently using:

1. the latest supervisor feedback;
2. official UDG diploma requirements;
3. actual experimental evidence;
4. reliable scientific literature;
5. documented methodological reasoning.

Do not invent or assume supervisor approval for decisions that were not explicitly approved.

When a methodological decision is not directly specified by the supervisor:

- analyze the available data;
- consult relevant scientific literature where necessary;
- compare reasonable alternatives where feasible;
- select the most defensible approach;
- document the reason for the decision.

Do not block normal project progress waiting for supervisor confirmation.

## Immediate delivery priority

A complete first working version of the diploma is required soon.

The immediate objective is therefore to produce a coherent end-to-end research prototype and thesis draft before pursuing optional improvements.

Priority order:

1. E01 — Clinical metadata and confounder analysis
2. E02 — Cohort construction / age control
3. E03 — Clean Random Forest baseline
4. E04 — Subject-level aggregation
5. E05 — Threshold analysis
6. E06 — Corrected Feature Importance
7. E07 — Initial transfer-learning CNN
8. E08 — CNN subject-level evaluation
9. Preliminary RF vs CNN comparison
10. Initial information-system prototype
11. Complete first thesis draft

Optional improvements such as extensive hyperparameter optimization, additional CNN architectures, Grad-CAM and advanced UI development must not delay the first working version.

Thesis writing should proceed in parallel with experiments whenever the relevant information has already been confirmed.