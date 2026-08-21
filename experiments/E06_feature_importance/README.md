# E06 - Corrected Grouped Feature Importance

## Status

Implemented, not yet confirmed.

`analyze_feature_importance.py` has been prepared. E06 should not be marked completed until the script is executed successfully and the generated outputs are reviewed.

## Objective

Correct the grouped Feature Importance interpretation for the E03 Random Forest baseline.

E06 does not retrain the Random Forest, modify the E02 cohort, change E03 predictions, change E03 metrics, perform threshold analysis, or implement a CNN.

## Relationship To E00

E00 included an original grouped Feature Importance analysis. A methodological limitation was identified: feature groups contain different numbers of individual features, while the original grouped comparison used summed importance values.

E06 addresses that limitation in a new experiment. E00 remains frozen.

## Relationship To E03

E03 reproduced the E00-style Random Forest baseline on the E02 Age >= 60 cohort and saved per-feature importance values.

E06 uses the saved E03 files:

- `experiments/E03_clean_rf/results/feature_importance.csv`
- `experiments/E03_clean_rf/results/legacy_unadjusted_category_importance.csv`

The primary input contains one row per handcrafted feature and includes:

- `feature`
- `permutation_importance_mean`
- `permutation_importance_std`
- `rf_impurity_importance`
- `category`

## Why Correction Is Needed

The legacy grouped summary summed positive permutation importance values within each category.

Summed values can be useful for preserving E00-style comparability, but they can favor categories that contain many individual features. E06 therefore reports category size and normalized mean-per-feature summaries.

The legacy analysis should be described as unadjusted and potentially biased toward larger feature groups, not simply as wrong.

## Feature Categories

E06 uses the same feature categories defined in E03:

- Global intensity / contrast
- Intensity distribution
- Shape / brain mask geometry
- Edges / structural boundaries
- Regional intensity / contrast
- Texture: Local Binary Pattern

No medical or anatomical categories are invented.

## Grouped Normalization Approach

For each category, E06 records:

- `feature_count`
- total raw permutation importance
- summed positive permutation importance
- mean raw permutation importance per feature
- mean positive permutation importance per feature
- median permutation importance per feature
- maximum individual permutation importance
- count and fraction of features with positive permutation importance

The main corrected grouped permutation measures are:

- `mean_permutation_importance_per_feature`
- `mean_positive_permutation_importance_per_feature`

E06 also reports RF impurity importance by category using:

- total RF impurity importance
- mean RF impurity importance per feature
- median RF impurity importance per feature
- maximum RF impurity importance

Because categories have different sizes, mean-per-feature values should be the main grouped comparison.

## Permutation vs Impurity Importance

Permutation importance estimates how much model performance changes when a feature is disrupted on the E03 image-level test data.

RF impurity importance comes from the fitted Random Forest itself. It can favor certain feature types and should not be interpreted alone.

The two measures answer related but different questions. E06 does not silently choose one measure as universally correct.

## ROI / Regional Features

The regional 4x8 features are summarized descriptively as image grid regions whose handcrafted features contributed to the model.

If the ROI heatmap is generated, it must be interpreted only as model feature importance over image grid regions. It must not be described as disease regions, Alzheimer biomarkers, pathological regions, or clinically confirmed anatomical findings.

## Generated Outputs

When executed, the script writes outputs to `experiments/E06_feature_importance/results/`:

- `category_feature_counts.csv`
- `corrected_category_permutation_importance.csv`
- `corrected_category_rf_importance.csv`
- `legacy_vs_corrected_category_comparison.csv`
- `top_individual_features.csv`
- `category_rankings.csv`
- `roi_grid_importance.csv`
- `normalized_category_permutation_importance.png`
- `normalized_category_rf_importance.png`
- `legacy_vs_corrected_importance.png`
- `roi_grid_importance_heatmap.png`
- `e06_summary.json`

## Interpretation Limitations

Feature Importance measures model reliance, not clinical causality.

High importance does not prove a medical biomarker.

Handcrafted image features are algorithmic descriptors.

Permutation importance is evaluated on the E03 image-level test data.

Correlated features may share or redistribute importance.

RF impurity importance can favor certain feature types and should not be interpreted alone.

Category comparison should account for the number of features in each category.

Final medical interpretation is outside the scope of this experiment.

No medical biomarkers are identified in E06.

## How To Run

From the repository root:

```powershell
python experiments/E06_feature_importance/analyze_feature_importance.py
```
