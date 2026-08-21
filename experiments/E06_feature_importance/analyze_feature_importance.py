"""E06 - Corrected grouped Feature Importance analysis.

This script reuses the saved E03 per-feature importance values. It does not
retrain the Random Forest, modify predictions, or recalculate E03 metrics.
Grouped summaries are normalized by category feature count so categories with
different numbers of features are not compared only by summed importance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FEATURE_COUNT = 89
EXPECTED_CATEGORIES = [
    "Global intensity / contrast",
    "Intensity distribution",
    "Shape / brain mask geometry",
    "Edges / structural boundaries",
    "Regional intensity / contrast",
    "Texture: Local Binary Pattern",
]
REQUIRED_IMPORTANCE_COLUMNS = [
    "feature",
    "permutation_importance_mean",
    "permutation_importance_std",
    "rf_impurity_importance",
    "category",
]
LEGACY_REQUIRED_COLUMNS = [
    "category",
    "legacy_unadjusted_positive_permutation_importance",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
E03_RESULTS_DIR = PROJECT_ROOT / "experiments" / "E03_clean_rf" / "results"
FEATURE_IMPORTANCE_PATH = E03_RESULTS_DIR / "feature_importance.csv"
LEGACY_CATEGORY_IMPORTANCE_PATH = E03_RESULTS_DIR / "legacy_unadjusted_category_importance.csv"
RESULTS_DIR = SCRIPT_DIR / "results"

INTERPRETATION_RULES = [
    "Feature Importance measures model reliance, not clinical causality.",
    "High importance does not prove a medical biomarker.",
    "Handcrafted image features are algorithmic descriptors.",
    "Permutation importance is evaluated on the E03 image-level test data.",
    "Correlated features may share or redistribute importance.",
    "RF impurity importance can favor certain feature types and should not be interpreted alone.",
    "Category comparison should account for the number of features in each category.",
    "Final medical interpretation is outside the scope of this experiment.",
]
LEGACY_COMPARISON_NOTE = (
    "The legacy E03 category summary uses unadjusted summed positive permutation "
    "importance. That is useful for preserving E00-style comparability, but it can "
    "favor larger feature groups. E06 reports mean-per-feature summaries as the main "
    "corrected grouped comparison."
)
NO_BIOMARKER_STATEMENT = (
    "E06 identifies model-used image feature patterns only; no medical biomarkers "
    "are identified."
)


def to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_python(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def ensure_required_columns(frame: pd.DataFrame, required_columns: list[str], source: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def expected_category_for_feature(feature: str) -> str:
    if feature.startswith("roi_"):
        return "Regional intensity / contrast"
    if feature.startswith("lbp_"):
        return "Texture: Local Binary Pattern"
    if feature.startswith("edge_"):
        return "Edges / structural boundaries"
    if feature in {"brain_area_ratio", "bbox_area_ratio", "centroid_x", "centroid_y"}:
        return "Shape / brain mask geometry"
    if feature in {"entropy", "dark_ratio", "bright_ratio"}:
        return "Intensity distribution"
    if feature.startswith("global_"):
        return "Global intensity / contrast"
    return "Other"


def load_feature_importance() -> pd.DataFrame:
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    ensure_required_columns(importance, REQUIRED_IMPORTANCE_COLUMNS, "E03 feature importance")
    importance = importance.copy()

    for column in [
        "permutation_importance_mean",
        "permutation_importance_std",
        "rf_impurity_importance",
    ]:
        importance[column] = pd.to_numeric(importance[column], errors="raise")

    validate_feature_importance(importance)
    return importance


def validate_feature_importance(importance: pd.DataFrame) -> None:
    if len(importance) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature rows; observed {len(importance)}."
        )

    if importance["feature"].duplicated().any():
        duplicates = importance.loc[importance["feature"].duplicated(), "feature"].tolist()
        raise ValueError(f"Duplicate feature names found: {duplicates}")

    missing_category = importance["category"].isna() | (
        importance["category"].astype(str).str.strip() == ""
    )
    if missing_category.any():
        features = importance.loc[missing_category, "feature"].tolist()
        raise ValueError(f"Features with missing category assignments: {features}")

    missing_impurity = importance["rf_impurity_importance"].isna()
    if missing_impurity.any():
        features = importance.loc[missing_impurity, "feature"].tolist()
        raise ValueError(f"Features with missing RF impurity importance: {features}")

    numeric_missing = importance[
        ["permutation_importance_mean", "permutation_importance_std", "rf_impurity_importance"]
    ].isna()
    if numeric_missing.any().any():
        raise ValueError(
            "Numeric importance columns contain missing values: "
            f"{numeric_missing.sum().to_dict()}"
        )

    observed_categories = set(importance["category"])
    expected_categories = set(EXPECTED_CATEGORIES)
    if observed_categories != expected_categories:
        raise ValueError(
            "Unexpected category set. "
            f"Observed {sorted(observed_categories)}; expected {EXPECTED_CATEGORIES}."
        )

    category_check = importance[["feature", "category"]].copy()
    category_check["expected_category"] = category_check["feature"].map(expected_category_for_feature)
    mismatched = category_check[category_check["category"] != category_check["expected_category"]]
    if not mismatched.empty:
        raise ValueError(
            "Feature category assignments are not internally consistent with E03 mapping: "
            f"{mismatched.to_dict(orient='records')}"
        )


def load_legacy_category_importance() -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_CATEGORY_IMPORTANCE_PATH)
    ensure_required_columns(
        legacy, LEGACY_REQUIRED_COLUMNS, "E03 legacy unadjusted category importance"
    )
    legacy = legacy.copy()
    legacy["legacy_unadjusted_positive_permutation_importance"] = pd.to_numeric(
        legacy["legacy_unadjusted_positive_permutation_importance"], errors="raise"
    )
    observed_categories = set(legacy["category"])
    if observed_categories != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "Legacy category importance has unexpected categories. "
            f"Observed {sorted(observed_categories)}; expected {EXPECTED_CATEGORIES}."
        )
    return legacy


def category_feature_counts(importance: pd.DataFrame) -> pd.DataFrame:
    return (
        importance.groupby("category", as_index=False)
        .agg(feature_count=("feature", "count"))
        .sort_values("category")
        .reset_index(drop=True)
    )


def corrected_category_permutation_importance(importance: pd.DataFrame) -> pd.DataFrame:
    frame = importance.copy()
    frame["positive_permutation_importance"] = frame["permutation_importance_mean"].clip(lower=0)
    frame["has_positive_permutation_importance"] = frame["permutation_importance_mean"] > 0

    grouped = frame.groupby("category", as_index=False).agg(
        feature_count=("feature", "count"),
        total_permutation_importance=("permutation_importance_mean", "sum"),
        positive_permutation_importance_sum=("positive_permutation_importance", "sum"),
        mean_permutation_importance_per_feature=("permutation_importance_mean", "mean"),
        median_permutation_importance_per_feature=("permutation_importance_mean", "median"),
        maximum_individual_permutation_importance=("permutation_importance_mean", "max"),
        positive_feature_count=("has_positive_permutation_importance", "sum"),
    )
    grouped["mean_positive_permutation_importance_per_feature"] = (
        grouped["positive_permutation_importance_sum"] / grouped["feature_count"]
    )
    grouped["fraction_positive_permutation_features"] = (
        grouped["positive_feature_count"] / grouped["feature_count"]
    )

    return grouped[
        [
            "category",
            "feature_count",
            "total_permutation_importance",
            "positive_permutation_importance_sum",
            "mean_permutation_importance_per_feature",
            "mean_positive_permutation_importance_per_feature",
            "median_permutation_importance_per_feature",
            "maximum_individual_permutation_importance",
            "positive_feature_count",
            "fraction_positive_permutation_features",
        ]
    ].sort_values(
        "mean_positive_permutation_importance_per_feature", ascending=False
    ).reset_index(drop=True)


def corrected_category_rf_importance(importance: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        importance.groupby("category", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            total_rf_impurity_importance=("rf_impurity_importance", "sum"),
            mean_rf_impurity_importance_per_feature=("rf_impurity_importance", "mean"),
            median_rf_impurity_importance_per_feature=("rf_impurity_importance", "median"),
            maximum_rf_impurity_importance=("rf_impurity_importance", "max"),
        )
        .sort_values("mean_rf_impurity_importance_per_feature", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def legacy_vs_corrected_comparison(
    legacy: pd.DataFrame, corrected_permutation: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "category",
        "feature_count",
        "positive_permutation_importance_sum",
        "mean_positive_permutation_importance_per_feature",
        "mean_permutation_importance_per_feature",
    ]
    comparison = legacy.merge(
        corrected_permutation[columns],
        on="category",
        how="inner",
        validate="one_to_one",
    )
    comparison = comparison.rename(
        columns={
            "legacy_unadjusted_positive_permutation_importance": (
                "legacy_summed_positive_permutation_importance"
            ),
            "positive_permutation_importance_sum": (
                "recomputed_summed_positive_permutation_importance"
            ),
        }
    )
    comparison["legacy_minus_recomputed_sum_abs"] = (
        comparison["legacy_summed_positive_permutation_importance"]
        - comparison["recomputed_summed_positive_permutation_importance"]
    ).abs()
    comparison["interpretation_note"] = LEGACY_COMPARISON_NOTE
    return comparison.sort_values(
        "legacy_summed_positive_permutation_importance", ascending=False
    ).reset_index(drop=True)


def top_individual_features(importance: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    top = (
        importance.sort_values("permutation_importance_mean", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top[
        [
            "rank",
            "feature",
            "category",
            "permutation_importance_mean",
            "permutation_importance_std",
            "rf_impurity_importance",
        ]
    ]


def category_rankings(
    corrected_permutation: pd.DataFrame, corrected_rf: pd.DataFrame
) -> pd.DataFrame:
    ranking = corrected_permutation[
        [
            "category",
            "feature_count",
            "mean_permutation_importance_per_feature",
            "mean_positive_permutation_importance_per_feature",
        ]
    ].merge(
        corrected_rf[["category", "mean_rf_impurity_importance_per_feature"]],
        on="category",
        how="inner",
        validate="one_to_one",
    )
    ranking["rank_by_mean_permutation_importance_per_feature"] = ranking[
        "mean_permutation_importance_per_feature"
    ].rank(method="min", ascending=False).astype(int)
    ranking["rank_by_mean_positive_permutation_importance_per_feature"] = ranking[
        "mean_positive_permutation_importance_per_feature"
    ].rank(method="min", ascending=False).astype(int)
    ranking["rank_by_mean_rf_impurity_importance_per_feature"] = ranking[
        "mean_rf_impurity_importance_per_feature"
    ].rank(method="min", ascending=False).astype(int)
    return ranking.sort_values(
        [
            "rank_by_mean_positive_permutation_importance_per_feature",
            "rank_by_mean_permutation_importance_per_feature",
            "category",
        ]
    ).reset_index(drop=True)


def roi_summary(importance: pd.DataFrame) -> pd.DataFrame:
    roi_rows = importance[importance["feature"].str.startswith("roi_")].copy()
    parsed = roi_rows["feature"].str.extract(r"^roi_r(?P<roi_row>\d+)_c(?P<roi_col>\d+)_(?P<stat>mean|std)$")
    if parsed.isna().any().any():
        invalid = roi_rows.loc[parsed.isna().any(axis=1), "feature"].tolist()
        raise ValueError(f"Unexpected ROI feature names: {invalid}")
    roi_rows = pd.concat([roi_rows, parsed], axis=1)
    roi_rows["roi_row"] = roi_rows["roi_row"].astype(int)
    roi_rows["roi_col"] = roi_rows["roi_col"].astype(int)
    roi_rows["positive_permutation_importance"] = roi_rows[
        "permutation_importance_mean"
    ].clip(lower=0)

    return (
        roi_rows.groupby(["roi_row", "roi_col"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_permutation_importance_per_feature=("permutation_importance_mean", "mean"),
            mean_positive_permutation_importance_per_feature=(
                "positive_permutation_importance",
                "mean",
            ),
            mean_rf_impurity_importance_per_feature=("rf_impurity_importance", "mean"),
        )
        .sort_values(["roi_row", "roi_col"])
        .reset_index(drop=True)
    )


def save_bar_chart(
    frame: pd.DataFrame,
    *,
    value_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plot_frame = frame.sort_values(value_column, ascending=False)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    ax.bar(plot_frame["category"], plot_frame[value_column], color="#4c78a8")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_legacy_vs_corrected_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    plot_frame = comparison.sort_values(
        "legacy_summed_positive_permutation_importance", ascending=False
    )
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)

    axes[0].barh(
        plot_frame["category"],
        plot_frame["legacy_summed_positive_permutation_importance"],
        color="#7f7f7f",
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Legacy unadjusted sum", loc="left", fontweight="bold")
    axes[0].set_xlabel("Summed positive permutation importance")
    axes[0].grid(axis="x", alpha=0.25)

    corrected = plot_frame.sort_values(
        "mean_positive_permutation_importance_per_feature", ascending=False
    )
    axes[1].barh(
        corrected["category"],
        corrected["mean_positive_permutation_importance_per_feature"],
        color="#59a14f",
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Corrected mean per feature", loc="left", fontweight="bold")
    axes[1].set_xlabel("Mean positive permutation importance per feature")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Legacy summed and corrected normalized category importance use different scales",
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_roi_heatmap(roi_frame: pd.DataFrame, output_path: Path) -> None:
    grid = roi_frame.pivot(
        index="roi_row",
        columns="roi_col",
        values="mean_positive_permutation_importance_per_feature",
    ).sort_index()

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    image = ax.imshow(grid.to_numpy(), cmap="viridis")
    ax.set_xticks(np.arange(grid.shape[1]), labels=[str(col) for col in grid.columns])
    ax.set_yticks(np.arange(grid.shape[0]), labels=[str(row) for row in grid.index])
    ax.set_xlabel("Image grid column")
    ax.set_ylabel("Image grid row")
    ax.set_title(
        "Model feature importance over image grid regions",
        loc="left",
        fontweight="bold",
    )
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            value = grid.to_numpy()[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value > np.nanmax(grid.to_numpy()) * 0.55 else "black",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def top_category_by_metric(ranking: pd.DataFrame, metric: str) -> dict[str, Any]:
    max_value = ranking[metric].max()
    categories = ranking.loc[np.isclose(ranking[metric], max_value), "category"].tolist()
    return {"metric": metric, "value": float(max_value), "categories": categories}


def ranking_difference_note(ranking: pd.DataFrame) -> str:
    top_mean_perm = set(
        ranking.loc[
            ranking["rank_by_mean_permutation_importance_per_feature"] == 1, "category"
        ]
    )
    top_mean_positive = set(
        ranking.loc[
            ranking["rank_by_mean_positive_permutation_importance_per_feature"] == 1,
            "category",
        ]
    )
    top_rf = set(
        ranking.loc[ranking["rank_by_mean_rf_impurity_importance_per_feature"] == 1, "category"]
    )
    if top_mean_perm == top_mean_positive == top_rf:
        return (
            "The same category is top-ranked by the corrected mean raw permutation, "
            "mean positive permutation, and mean RF impurity measures. This does not "
            "make the category a medical finding."
        )
    return (
        "Category rankings differ across corrected importance definitions, so E06 "
        "reports the differences rather than forcing one definitive ranking."
    )


def build_summary(
    importance: pd.DataFrame,
    counts: pd.DataFrame,
    ranking: pd.DataFrame,
    top_features: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "experiment_id": "E06",
        "objective": "Corrected grouped Feature Importance analysis",
        "input_file": str(FEATURE_IMPORTANCE_PATH.relative_to(PROJECT_ROOT)),
        "number_of_features": int(len(importance)),
        "category_counts": counts.to_dict(orient="records"),
        "top_category_by_corrected_metric": {
            "mean_permutation_importance_per_feature": top_category_by_metric(
                ranking, "mean_permutation_importance_per_feature"
            ),
            "mean_positive_permutation_importance_per_feature": top_category_by_metric(
                ranking, "mean_positive_permutation_importance_per_feature"
            ),
            "mean_rf_impurity_importance_per_feature": top_category_by_metric(
                ranking, "mean_rf_impurity_importance_per_feature"
            ),
        },
        "top_individual_features": top_features.head(20).to_dict(orient="records"),
        "legacy_vs_corrected_interpretation_note": LEGACY_COMPARISON_NOTE,
        "ranking_difference_note": ranking_difference_note(ranking),
        "limitations": INTERPRETATION_RULES,
        "no_medical_biomarkers_identified": True,
        "no_medical_biomarker_statement": NO_BIOMARKER_STATEMENT,
        "importance_measure_note": (
            "No single importance measure is treated as universally correct. "
            "Permutation and RF impurity importance answer related but different questions."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(to_plain_python(payload), output_file, indent=2, ensure_ascii=False)


def print_summary(
    counts: pd.DataFrame,
    ranking: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    print("\nE06 - Corrected grouped Feature Importance")
    print("=" * 48)
    print(f"Features: {EXPECTED_FEATURE_COUNT}")
    print("\nFeature counts by category:")
    print(counts.to_string(index=False))
    print("\nCategory rankings:")
    print(
        ranking[
            [
                "category",
                "rank_by_mean_permutation_importance_per_feature",
                "rank_by_mean_positive_permutation_importance_per_feature",
                "rank_by_mean_rf_impurity_importance_per_feature",
            ]
        ].to_string(index=False)
    )
    print("\nTop individual features by permutation importance:")
    print(top_features.head(10).to_string(index=False))
    print("\nNo medical biomarkers are identified in E06.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    importance = load_feature_importance()
    legacy = load_legacy_category_importance()

    counts = category_feature_counts(importance)
    corrected_permutation = corrected_category_permutation_importance(importance)
    corrected_rf = corrected_category_rf_importance(importance)
    legacy_comparison = legacy_vs_corrected_comparison(legacy, corrected_permutation)
    top_features = top_individual_features(importance)
    rankings = category_rankings(corrected_permutation, corrected_rf)
    roi = roi_summary(importance)

    counts.to_csv(RESULTS_DIR / "category_feature_counts.csv", index=False)
    corrected_permutation.to_csv(
        RESULTS_DIR / "corrected_category_permutation_importance.csv", index=False
    )
    corrected_rf.to_csv(RESULTS_DIR / "corrected_category_rf_importance.csv", index=False)
    legacy_comparison.to_csv(
        RESULTS_DIR / "legacy_vs_corrected_category_comparison.csv", index=False
    )
    top_features.to_csv(RESULTS_DIR / "top_individual_features.csv", index=False)
    rankings.to_csv(RESULTS_DIR / "category_rankings.csv", index=False)
    roi.to_csv(RESULTS_DIR / "roi_grid_importance.csv", index=False)

    save_bar_chart(
        corrected_permutation,
        value_column="mean_positive_permutation_importance_per_feature",
        output_path=RESULTS_DIR / "normalized_category_permutation_importance.png",
        title="Corrected category permutation importance",
        ylabel="Mean positive permutation importance per feature",
    )
    save_bar_chart(
        corrected_rf,
        value_column="mean_rf_impurity_importance_per_feature",
        output_path=RESULTS_DIR / "normalized_category_rf_importance.png",
        title="Corrected category RF impurity importance",
        ylabel="Mean RF impurity importance per feature",
    )
    save_legacy_vs_corrected_plot(
        legacy_comparison, RESULTS_DIR / "legacy_vs_corrected_importance.png"
    )
    save_roi_heatmap(roi, RESULTS_DIR / "roi_grid_importance_heatmap.png")

    summary = build_summary(
        importance=importance,
        counts=counts,
        ranking=rankings,
        top_features=top_features,
    )
    write_json(RESULTS_DIR / "e06_summary.json", summary)
    print_summary(counts, rankings, top_features)


if __name__ == "__main__":
    main()
