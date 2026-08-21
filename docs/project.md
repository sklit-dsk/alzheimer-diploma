# Project Overview

## Diploma topic

Informacioni sistem za rano otkrivanje Alchajmerove bolesti zasnovan na metodama mašinskog učenja

## University

Univerzitet Donja Gorica

Fakultet za informacione sisteme i tehnologije

---

## Project goal

The goal of the project is to develop and evaluate a research information system that uses machine-learning methods to distinguish subjects without dementia from subjects with Very mild Dementia based on brain MRI images.

The project focuses on early-stage classification and on methodological reliability of the evaluation.

Special attention is given to:

- subject-level separation of training and test data;
- possible confounding factors, especially age;
- subject-level evaluation instead of evaluation only on individual MRI slices;
- sensitivity to the Very mild Dementia class;
- comparison between a handcrafted-feature Random Forest baseline and an image-based deep-learning model.

The resulting system is intended as a research prototype and not as a clinically validated diagnostic system.

---

## Main research question

U kojoj mjeri metode mašinskog učenja zasnovane na MRI snimcima mogu razlikovati subjekte bez demencije od subjekata sa veoma blagim stadijumom demencije, uz kontrolu mogućih confounding faktora i evaluaciju na nivou subjekta?

---

## Working hypothesis

Primjenom metoda mašinskog učenja na MRI snimke mozga moguće je ostvariti razlikovanje subjekata bez demencije od subjekata sa veoma blagim stadijumom demencije, pri čemu kontrola sastava kohorte i evaluacija na nivou subjekta omogućavaju metodološki pouzdaniju procjenu performansi modela.

The hypothesis is provisional and may be refined after further analysis and supervisor feedback.

---

## Dataset

The project currently uses the ImagesOASIS dataset available on Kaggle.

The dataset is derived from OASIS MRI data.

Current primary classification scenario:

- Non Demented
- Very mild Dementia

The dataset contains multiple MRI slices per subject, therefore subject identity must be preserved throughout the experimental pipeline.

Individual MRI slices belonging to one subject must not be distributed across both training and test sets.

---

## Current baseline

The existing baseline uses:

- handcrafted image features;
- Random Forest classification;
- subject-level train/test split.

This experiment is identified as:

E00 — Original Random Forest baseline

The initial results showed that recall for the Very mild Dementia class was substantially lower than recall for the Non Demented class.

Because the diploma focuses on early detection, sensitivity toward the Very mild Dementia class is a major evaluation concern.

---

## Supervisor feedback

Before developing the main image-based model, the following issues must be addressed:

1. connect subjects with clinical metadata;
2. analyze Age, Sex, CDR and MMSE where available;
3. investigate age as a potential confounding factor;
4. construct a methodologically cleaner cohort;
5. repeat the Random Forest baseline on the new cohort;
6. perform final evaluation at subject level;
7. analyze sensitivity and specificity at multiple classification thresholds;
8. correct grouped Feature Importance interpretation;
9. only then develop an image-based model such as a transfer-learning CNN;
10. compare the CNN with the Random Forest baseline.

---

## System concept

The final project should contain both an experimental machine-learning pipeline and a simple information-system layer.

The conceptual flow is:

MRI data
→ preprocessing
→ machine-learning model
→ slice-level prediction
→ subject-level aggregation
→ subject-level result
→ presentation inside the information system

The information system should allow a user to provide or select MRI data and receive a model-generated research classification result.

The output should be presented as a probabilistic or classification result and must not be presented as a medical diagnosis.

---

## Research methodology

The main methodological stages are:

1. dataset analysis;
2. clinical metadata integration;
3. confounder analysis;
4. cohort construction;
5. subject-level train/test separation;
6. Random Forest baseline;
7. subject-level aggregation;
8. threshold analysis;
9. Feature Importance analysis;
10. transfer-learning CNN;
11. comparison of models;
12. implementation of the final research information system.

---

## Evaluation principles

The main evaluation unit is the subject.

Slice-level predictions may be used internally, but the main final metrics should be calculated after predictions are aggregated at subject level.

Possible aggregation strategies include:

- mean probability;
- median probability;
- majority voting.

Main metrics include:

- accuracy;
- balanced accuracy;
- sensitivity;
- specificity;
- precision;
- recall;
- F1;
- ROC-AUC;
- PR-AUC;
- confusion matrix.

Sensitivity for the Very mild Dementia class is a particularly important metric.

---

## Scope

The project does not aim to prove clinical diagnostic validity.

The project aims to investigate:

- whether MRI-based machine-learning models can distinguish the two selected groups;
- how dataset cohort construction influences model performance;
- how subject-level evaluation changes the interpretation of results;
- how a traditional machine-learning baseline compares with an image-based deep-learning model.

---

## Out of scope

The following are not currently primary goals:

- clinical deployment;
- diagnosis of Alzheimer's disease;
- replacement of medical specialists;
- real-world hospital integration;
- regulatory validation;
- full production deployment.

These may be mentioned as possible future work but should not be presented as completed functionality.

---

## Experiment roadmap

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

---

## Current project stage

Current active experiment:

E01 — Clinical metadata and confounder analysis

The next technical objective is to obtain and connect clinical metadata with ImagesOASIS subject identifiers and analyze potential confounding factors before modifying the baseline model.