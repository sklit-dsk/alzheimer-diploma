# Research Methodology

## Purpose

This document records the methodological decisions of the diploma research.

It should be updated when a methodological decision is made or changed.

Experimental results belong primarily in `docs/experiments.md`.

---

# 1. Research problem

The diploma investigates the use of machine-learning methods for early Alzheimer's disease classification based on brain MRI images.

The primary classification scenario is:

- Non Demented
- Very mild Dementia

The focus on Very mild Dementia is intended to investigate an early-stage classification scenario rather than classification of more advanced dementia stages.

---

# 2. Research question

Current working research question:

**U kojoj mjeri metode mašinskog učenja zasnovane na MRI snimcima mogu razlikovati subjekte bez demencije od subjekata sa veoma blagim stadijumom demencije, uz kontrolu mogućih confounding faktora i evaluaciju na nivou subjekta?**

---

# 3. Working hypothesis

Current working hypothesis:

**Primjenom metoda mašinskog učenja na MRI snimke mozga moguće je ostvariti razlikovanje subjekata bez demencije od subjekata sa veoma blagim stadijumom demencije, pri čemu kontrola sastava kohorte i evaluacija na nivou subjekta omogućavaju metodološki pouzdaniju procjenu performansi modela.**

This hypothesis is provisional and may be refined if experimental evidence requires it.

---

# 4. Dataset

The image dataset currently used is ImagesOASIS, derived from OASIS MRI data.

The dataset contains multiple MRI slices for individual subjects.

Therefore the image itself is not treated as an independent subject for the main final evaluation.

Clinical metadata was integrated from OASIS-1 where subject identifiers could be reliably connected.

The clinical metadata source and matching procedure were documented in E01.

---

# 5. Classification scenario

The primary binary classification problem is:

**Non Demented vs Very mild Dementia**

Other dementia classes present in the dataset are not part of the primary early-detection experiment.

The original E00 dataset analysis also contained Mild Dementia and Moderate Dementia, but these classes were excluded from the binary baseline scenario.

---

# 6. Subject-level separation

A subject may have multiple MRI slices.

Slices belonging to the same subject must not be intentionally distributed between training and test sets.

The train/test split must therefore be performed using subject identity.

This principle applies to both:

- Random Forest experiments;
- CNN experiments.

This reduces the risk of subject leakage caused by highly related images from the same person appearing in both sets.

---

# 7. Clinical metadata and potential confounding

The scientific supervisor identified age as an important potential confounding factor.

The clinical analysis included, where available:

- Age;
- Sex;
- CDR;
- MMSE.

Before selecting the final cohort, the distributions of these variables were examined.

In particular, the age distributions of:

- Non Demented;
- Very mild Dementia

were compared.

E01 identified a severe original age imbalance and E02 selected the Age >= 60 cohort as the primary downstream cohort strategy.

Candidate strategies considered included:

1. restricting the analysis to an older population such as subjects aged 60+;
2. constructing approximately age-matched diagnostic groups;
3. another strategy justified by the observed data and scientific literature.

Approximate age matching remains optional as a sensitivity or additional analysis, but it was not selected as the primary strategy because it would reduce the available sample further.

---

# 8. Baseline model

The traditional machine-learning baseline uses handcrafted image features and a Random Forest classifier.

E00 represents the original baseline.

After clinical and cohort analysis, the corrected Random Forest baseline was produced as E03 on the E02 Age >= 60 cohort.

E00 must remain unchanged so that the effect of methodological changes can be analyzed.

---

# 9. Evaluation level

The main final evaluation unit is the subject rather than the individual MRI slice.

Models may generate predictions for individual slices.

These predictions should then be aggregated into one prediction per subject.

Evaluated aggregation strategies include:

- mean predicted probability;
- median predicted probability;
- majority voting.

The strategies were compared experimentally.

Mean probability is used as the primary working subject-level probability aggregation for downstream comparison and for the information-system prototype. This is a methodological working choice, not proof that mean aggregation is clinically optimal.

---

# 10. Evaluation metrics

Where applicable, evaluation should include:

- accuracy;
- balanced accuracy;
- sensitivity;
- specificity;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- confusion matrix.

For the primary binary classification task:

**Very mild Dementia is treated as the positive class for early-detection evaluation.**

Therefore sensitivity refers primarily to the proportion of Very mild Dementia subjects correctly identified by the model.

Because early detection is the focus of the research, sensitivity must receive particular attention.

Overall accuracy alone is not sufficient for evaluating the final model.

---

# 11. Decision threshold

The default probability threshold of 0.5 must not be treated as automatically optimal.

The relationship between:

- sensitivity;
- specificity;

was analyzed across multiple classification thresholds in E05.

Threshold selection must be reported transparently.

The final RF-vs-CNN descriptive comparison and the information-system prototype use threshold `0.5`.

E05 threshold analysis remains descriptive on the test set and is not validated threshold optimization.

No alternative threshold was selected as final or clinically validated.

---

# 12. Feature Importance

The Random Forest baseline includes Feature Importance analysis.

Feature groups contain different numbers of individual features.

Therefore comparison based only on total summed importance can be misleading.

The corrected analysis should report, where appropriate:

- number of features per group;
- total importance;
- normalized or mean importance per feature.

Feature Importance describes characteristics used by the model.

It must not automatically be interpreted as identification of medical biomarkers.

---

# 13. Main image-based model

After the methodological corrections to the baseline were completed, an image-based deep-learning model was developed.

The selected approach is transfer learning using a CNN.

E07 selected ResNet18 transfer learning with ImageNet pretrained weights and a two-class final layer.

Model selection considered:

- suitability for the available dataset;
- computational requirements;
- reproducibility;
- ability to perform subject-level evaluation;
- comparability with the Random Forest baseline.

The final CNN follows the same subject-separation principles used by the corrected baseline.

---

# 14. Model comparison

The final comparison includes:

- corrected Random Forest baseline;
- selected CNN model, ResNet18 CNN E07 Attempt 2.

Comparison was performed using compatible cohorts and evaluation principles.

The comparison must not focus only on accuracy.

Special attention was paid to:

- sensitivity;
- specificity;
- F1;
- ROC-AUC;
- PR-AUC;
- subject-level performance.

---

# 15. Reproducibility

Experiments should explicitly record, where applicable:

- random seed;
- cohort definition;
- subject count;
- class distribution;
- train/test strategy;
- preprocessing;
- feature extraction;
- model hyperparameters;
- aggregation strategy;
- decision threshold;
- evaluation metrics.

Historical results must not be silently replaced when methodology changes.

---

# 16. Information-system prototype

The final practical component exposes the selected ML pipeline through a simple research information system.

Implemented pipeline:

MRI input
→ preprocessing
→ model inference
→ slice predictions
→ subject-level aggregation
→ result presentation

The implemented architecture is a local FastAPI application with server-rendered HTML/Jinja2 pages.

It accepts multiple JPG/JPEG MRI slices for one subject, processes uploads in memory, applies E07-compatible preprocessing, uses the saved E07 ResNet18 Attempt 2 checkpoint, aggregates slice predictions by mean probability, and applies threshold `0.5`.

The prototype does not intentionally store uploaded images permanently and does not use a database.

The prototype must not present model output as a confirmed medical diagnosis.

---

# 17. Interpretation limitations

The study evaluates machine-learning classification within the available dataset.

Results must not automatically be generalized to clinical populations outside the study dataset.

The project does not establish:

- clinical diagnostic validity;
- regulatory approval;
- replacement of medical specialists;
- confirmed medical biomarkers.

Limitations of the dataset, cohort, sample size and experimental methodology must be reported in the final thesis.

Major study limitations include the frozen subject-level test set of 29 subjects, absence of external validation, descriptive E05 threshold analysis on the test set and absence of clinical validation.

---

# 18. Current methodological priority

The current status is:

**E01-E09 and E11 are completed.**

The main ML experimental pipeline has addressed clinical metadata integration, cohort construction, corrected Random Forest evaluation, subject-level aggregation, threshold analysis, corrected Feature Importance, transfer-learning CNN evaluation, and RF-vs-CNN comparison.

The information-system prototype has been implemented and E11 final information-system evaluation completed successfully.

E10 Explainability remains optional and was not performed.

The mandatory experimental roadmap is complete except E10, which remains optional.

The next major stage is:

**preparation of the first complete diploma draft** based on confirmed results.
