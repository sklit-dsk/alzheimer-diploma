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

Clinical metadata will be integrated from an appropriate OASIS source where subject identifiers can be reliably connected.

The exact clinical metadata source and matching procedure must be documented during E01.

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

The clinical analysis should include, where available:

- Age;
- Sex;
- CDR;
- MMSE.

Before selecting the final cohort, the distributions of these variables must be examined.

In particular, the age distributions of:

- Non Demented;
- Very mild Dementia

must be compared.

No final age-control strategy is selected before E01 is completed.

Candidate strategies include:

1. restricting the analysis to an older population such as subjects aged 60+;
2. constructing approximately age-matched diagnostic groups;
3. another strategy justified by the observed data and scientific literature.

The selected strategy must be documented in E02.

---

# 8. Baseline model

The traditional machine-learning baseline uses handcrafted image features and a Random Forest classifier.

E00 represents the original baseline.

After clinical and cohort analysis, a corrected baseline will be produced as E03.

E00 must remain unchanged so that the effect of methodological changes can be analyzed.

---

# 9. Evaluation level

The main final evaluation unit is the subject rather than the individual MRI slice.

Models may generate predictions for individual slices.

These predictions should then be aggregated into one prediction per subject.

Candidate aggregation strategies include:

- mean predicted probability;
- median predicted probability;
- majority voting.

The strategies may be compared experimentally.

The selected primary aggregation method must be documented.

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

should be analyzed across multiple classification thresholds.

Threshold selection must be reported transparently.

If a threshold other than 0.5 is selected for a primary result, the reason must be documented.

Threshold selection must not be based on the final test set in a way that introduces optimistic bias.

The exact threshold-selection procedure will be defined before final model evaluation.

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

After the methodological corrections to the baseline are completed, an image-based deep-learning model will be developed.

The current intended approach is transfer learning using a CNN.

The exact architecture has not yet been selected.

Model selection should consider:

- suitability for the available dataset;
- computational requirements;
- reproducibility;
- ability to perform subject-level evaluation;
- comparability with the Random Forest baseline.

The final CNN must follow the same subject-separation principles used by the corrected baseline.

---

# 14. Model comparison

The final comparison should include:

- corrected Random Forest baseline;
- selected CNN model.

Comparison should be performed using compatible cohorts and evaluation principles.

The comparison must not focus only on accuracy.

Special attention should be paid to:

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

The final practical component should expose the selected ML pipeline through a simple research information system.

Conceptual pipeline:

MRI input
→ preprocessing
→ model inference
→ slice predictions
→ subject-level aggregation
→ result presentation

The exact application architecture will be selected after the main ML pipeline becomes sufficiently stable.

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

---

# 18. Current methodological priority

The current active stage is:

**E01 — Clinical metadata and confounder analysis**

No final cohort strategy has yet been selected.

The next methodological decision will be based on the actual clinical metadata analysis performed in E01.