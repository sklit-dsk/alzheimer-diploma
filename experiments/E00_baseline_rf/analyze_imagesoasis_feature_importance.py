from pathlib import Path
import json
import math
import random
import textwrap
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from skimage.filters import threshold_otsu
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.measure import shannon_entropy

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("/home/user/workspace/imagesoasis/Data")
OUT = Path("/home/user/workspace/feature_importance_output")
OUT.mkdir(exist_ok=True)

CLASS_ORDER = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
BINARY_CLASSES = ["Non Demented", "Very mild Dementia"]


def subject_id(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else path.stem


def scan_id(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else subject_id(path)


def collect_metadata() -> pd.DataFrame:
    rows = []
    for cls_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for p in cls_dir.glob("*.jpg"):
            rows.append(
                {
                    "path": str(p),
                    "class": cls,
                    "subject_id": subject_id(p),
                    "scan_id": scan_id(p),
                    "filename": p.name,
                }
            )
    return pd.DataFrame(rows)


def stratified_group_split(df: pd.DataFrame, classes, train_frac=0.8):
    train_subjects = set()
    test_subjects = set()
    rng = random.Random(SEED)
    for cls in classes:
        subjects = sorted(df.loc[df["class"] == cls, "subject_id"].unique())
        rng.shuffle(subjects)
        n_train = max(1, int(round(len(subjects) * train_frac)))
        if len(subjects) > 1:
            n_train = min(n_train, len(subjects) - 1)
        train_subjects.update(subjects[:n_train])
        test_subjects.update(subjects[n_train:])
    train = df[df["subject_id"].isin(train_subjects)].copy()
    test = df[df["subject_id"].isin(test_subjects)].copy()
    return train, test


def balanced_sample_by_class(df: pd.DataFrame, classes, max_per_class=None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    counts = df["class"].value_counts()
    target = min(int(counts.loc[cls]) for cls in classes)
    if max_per_class is not None:
        target = min(target, max_per_class)
    parts = []
    for cls in classes:
        idx = df.index[df["class"] == cls].to_numpy()
        chosen = rng.choice(idx, size=target, replace=False)
        parts.append(df.loc[chosen])
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


def load_grayscale(path: str, size=(128, 64)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def safe_otsu_mask(arr: np.ndarray) -> np.ndarray:
    try:
        positive = arr[arr > 0.01]
        thr = threshold_otsu(positive) if positive.size > 50 else 0.05
        mask = arr > max(0.03, float(thr) * 0.35)
    except Exception:
        mask = arr > 0.05
    if mask.mean() < 0.03:
        mask = arr > 0.03
    return mask


def extract_features(path: str) -> dict:
    arr = load_grayscale(path)
    h, w = arr.shape
    mask = safe_otsu_mask(arr)
    vals = arr[mask]
    if vals.size < 10:
        vals = arr.reshape(-1)

    features = {}
    features["global_mean"] = float(vals.mean())
    features["global_std"] = float(vals.std())
    features["global_p10"] = float(np.percentile(vals, 10))
    features["global_p50"] = float(np.percentile(vals, 50))
    features["global_p90"] = float(np.percentile(vals, 90))
    features["global_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    features["dark_ratio"] = float((vals < 0.15).mean())
    features["bright_ratio"] = float((vals > 0.75).mean())
    features["brain_area_ratio"] = float(mask.mean())

    ys, xs = np.where(mask)
    if len(xs):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        features["bbox_area_ratio"] = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / (h * w))
        features["centroid_x"] = float(xs.mean() / w)
        features["centroid_y"] = float(ys.mean() / h)
    else:
        features["bbox_area_ratio"] = 0.0
        features["centroid_x"] = 0.5
        features["centroid_y"] = 0.5

    edges = sobel(arr)
    edge_vals = edges[mask] if mask.any() else edges.reshape(-1)
    features["edge_mean"] = float(edge_vals.mean())
    features["edge_p90"] = float(np.percentile(edge_vals, 90))
    features["entropy"] = float(shannon_entropy((arr * 255).astype(np.uint8)))

    grid_rows, grid_cols = 4, 8
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(round(r * h / grid_rows))
            y1 = int(round((r + 1) * h / grid_rows))
            x0 = int(round(c * w / grid_cols))
            x1 = int(round((c + 1) * w / grid_cols))
            cell = arr[y0:y1, x0:x1]
            cell_mask = mask[y0:y1, x0:x1]
            cell_vals = cell[cell_mask] if cell_mask.sum() > 5 else cell.reshape(-1)
            features[f"roi_r{r+1}_c{c+1}_mean"] = float(cell_vals.mean())
            features[f"roi_r{r+1}_c{c+1}_std"] = float(cell_vals.std())

    lbp = local_binary_pattern((arr * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_vals = lbp[mask] if mask.any() else lbp.reshape(-1)
    hist, _ = np.histogram(lbp_vals, bins=np.arange(0, 11), density=True)
    hist = np.nan_to_num(hist)
    for i, val in enumerate(hist):
        features[f"lbp_uniform_bin_{i}"] = float(val)

    return features


def extract_feature_matrix(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    rows = []
    total = len(df)
    for i, row in enumerate(df.to_dict("records"), start=1):
        feats = extract_features(row["path"])
        feats.update(
            {
                "path": row["path"],
                "class": row["class"],
                "subject_id": row["subject_id"],
                "scan_id": row["scan_id"],
            }
        )
        rows.append(feats)
        if i % 1000 == 0:
            print(f"features {i}/{total}", flush=True)
    feat_df = pd.DataFrame(rows)
    feat_df.to_pickle(cache_path)
    return feat_df


def plot_class_distribution(class_counts: pd.Series):
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ordered = class_counts.reindex(CLASS_ORDER)
    colors = ["#20808D", "#A84B2F", "#1B474D", "#944454"]
    bars = ax.bar(ordered.index, ordered.values, color=colors)
    ax.set_title("Class distribution in ImagesOASIS dataset", loc="left", fontweight="bold")
    ax.set_ylabel("Number of JPG images")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, labels):
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Confusion matrix: early-stage binary classifier", loc="left", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_importance(importance_df: pd.DataFrame):
    top = importance_df.head(20).iloc[::-1]
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    ax.barh(top["feature"], top["permutation_importance_mean"], xerr=top["permutation_importance_std"], color="#20808D")
    ax.set_title("Top 20 features by permutation importance", loc="left", fontweight="bold")
    ax.set_xlabel("Mean decrease in balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "top20_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_roi_heatmap(importance_df: pd.DataFrame):
    heat = np.zeros((4, 8), dtype=float)
    for r in range(1, 5):
        for c in range(1, 9):
            keys = [f"roi_r{r}_c{c}_mean", f"roi_r{r}_c{c}_std"]
            val = importance_df.loc[importance_df["feature"].isin(keys), "permutation_importance_mean"].clip(lower=0).sum()
            heat[r - 1, c - 1] = val
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    im = ax.imshow(heat, cmap="YlGnBu")
    ax.set_title("Regional MRI feature importance map, 4×8 grid", loc="left", fontweight="bold")
    ax.set_xlabel("Image column region")
    ax.set_ylabel("Image row region")
    ax.set_xticks(range(8), [f"C{c}" for c in range(1, 9)])
    ax.set_yticks(range(4), [f"R{r}" for r in range(1, 5)])
    for r in range(4):
        for c in range(8):
            ax.text(c, r, f"{heat[r, c]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Summed permutation importance")
    fig.tight_layout()
    fig.savefig(OUT / "roi_importance_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def feature_category(feature: str) -> str:
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


def write_report(
    meta: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict,
    importance_df: pd.DataFrame,
    category_importance: pd.DataFrame,
):
    class_counts = meta["class"].value_counts().reindex(CLASS_ORDER)
    subject_counts = meta.groupby("class")["subject_id"].nunique().reindex(CLASS_ORDER)
    top10 = importance_df.head(10)

    report = f"""# Feature Importance для ImagesOASIS / OASIS Alzheimer's Detection

Этот анализ подготовлен для дипломной темы **“Informacioni sistem za rano otkrivanje Alchajmerove bolesti”**. Датасет был загружен из Kaggle-архива `ninadaithal/imagesoasis`; публичная карточка Kaggle описывает его как `OASIS Alzheimer's Detection`, а исходная серия OASIS публикует открытые нейровизуализационные данные для исследований старения и болезни Альцгеймера ([Kaggle dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis), [OASIS project](https://sites.wustl.edu/oasisbrains/)).

## Что было проанализировано

- **Файлы**: {len(meta):,} JPG MRI-срезов.
- **Классы изображений**: {", ".join(CLASS_ORDER)}.
- **Задача feature importance**: бинарная ранняя классификация `Non Demented` vs `Very mild Dementia`, потому что она ближе всего к раннему обнаружению заболевания.
- **Защита от data leakage**: train/test split сделан по `subject_id`, а не по отдельным изображениям. Это важно, потому что один пациент имеет много соседних MRI-срезов.

## Баланс классов

| Класс | Изображений | Уникальных subject_id |
|---|---:|---:|
"""
    for cls in CLASS_ORDER:
        report += f"| {cls} | {int(class_counts.loc[cls]):,} | {int(subject_counts.loc[cls]):,} |\n"

    report += f"""
Главное ограничение датасета: класс `Moderate Dementia` содержит только **{int(class_counts.loc['Moderate Dementia']):,} изображений от {int(subject_counts.loc['Moderate Dementia'])} пациентов**, поэтому надежная 4-классовая feature importance с patient-level split для этого класса статистически нестабильна. Для дипломной работы лучше отдельно указать, что основная интерпретация признаков выполнена для раннего сценария `Non Demented` vs `Very mild Dementia`.

## Метод

- **Предобработка**: изображения переведены в grayscale и приведены к размеру 128×64.
- **Маска мозга**: фон отделялся адаптивным порогом Otsu с fallback-порогом для темного фона.
- **Признаки**: глобальная яркость/контраст, доля темных и ярких пикселей, энтропия, edge features, геометрия маски мозга, 4×8 региональные mean/std признаки и LBP-текстура.
- **Модель**: `RandomForestClassifier`, потому что она дает интерпретируемую impurity importance и хорошо работает с табличными признаками.
- **Основная importance**: permutation importance на test set с метрикой `balanced_accuracy`. Она предпочтительнее простой impurity importance, потому что измеряет падение качества модели при перемешивании конкретного признака.

## Train/test выборка

| Split | Non Demented | Very mild Dementia | Всего |
|---|---:|---:|---:|
| Train | {int((train_df['class']=='Non Demented').sum()):,} | {int((train_df['class']=='Very mild Dementia').sum()):,} | {len(train_df):,} |
| Test | {int((test_df['class']=='Non Demented').sum()):,} | {int((test_df['class']=='Very mild Dementia').sum()):,} | {len(test_df):,} |

## Качество модели

| Метрика | Значение |
|---|---:|
| Accuracy | {metrics['accuracy']:.3f} |
| Balanced accuracy | {metrics['balanced_accuracy']:.3f} |
| F1 macro | {metrics['f1_macro']:.3f} |
| ROC AUC | {metrics['roc_auc']:.3f} |

Эти метрики не следует трактовать как финальное качество медицинской CNN-модели. Здесь модель используется как инструмент интерпретации признаков, а не как production-классификатор.

## Top-10 признаков

| Rank | Признак | Тип | Permutation importance |
|---:|---|---|---:|
"""
    for i, row in enumerate(top10.itertuples(index=False), start=1):
        report += f"| {i} | `{row.feature}` | {row.category} | {row.permutation_importance_mean:.4f} |\n"

    report += """
## Важность по группам признаков

| Группа признаков | Суммарная положительная permutation importance |
|---|---:|
"""
    for row in category_importance.itertuples(index=False):
        report += f"| {row.category} | {row.positive_permutation_importance:.4f} |\n"

    report += """
## Интерпретация для дипломной работы

Наиболее важные признаки показывают, какие измеримые характеристики MRI-срезов помогают отличать `Very mild Dementia` от `Non Demented`. Если среди top features доминируют региональные признаки `roi_*`, это означает, что модель опирается не на один глобальный показатель яркости, а на локальные зоны изображения. Если высоко стоят `edge_*`, `entropy` или `lbp_*`, это указывает на роль текстуры и границ структур мозга.

Для итоговой нейросетевой модели можно дополнить этот анализ Grad-CAM/Score-CAM поверх CNN: текущий Random Forest показывает важность инженерных признаков, а Grad-CAM покажет важные области уже внутри сверточной модели. В тексте диплома корректно назвать текущий этап **model-agnostic feature importance on interpretable MRI-derived features**.

## Файлы результата

- `class_distribution.png` — дисбаланс классов.
- `confusion_matrix.png` — матрица ошибок бинарной модели.
- `top20_permutation_importance.png` — 20 самых важных признаков.
- `roi_importance_heatmap.png` — карта важности регионов изображения.
- `feature_importance.csv` — таблица permutation/impurity importance.
- `metrics.json` — метрики и параметры эксперимента.

## Ограничения

- Изображения являются 2D JPG-срезами, а не исходными 3D NIfTI/DICOM объемами, поэтому часть медицински значимой информации может быть потеряна.
- В датасете сильный class imbalance; особенно мало пациентов в `Moderate Dementia`.
- Даже patient-level split не заменяет внешнюю клиническую валидацию. Для дипломного прототипа это приемлемо, но для медицинского применения нужна независимая проверка.
"""
    (OUT / "feature_importance_report.md").write_text(report, encoding="utf-8")


def main():
    meta = collect_metadata()
    meta.to_csv(OUT / "dataset_metadata.csv", index=False)
    class_counts = meta["class"].value_counts()
    plot_class_distribution(class_counts)

    binary_meta = meta[meta["class"].isin(BINARY_CLASSES)].copy()
    train_meta, test_meta = stratified_group_split(binary_meta, BINARY_CLASSES, train_frac=0.8)

    train_bal = balanced_sample_by_class(train_meta, BINARY_CLASSES)
    test_bal = balanced_sample_by_class(test_meta, BINARY_CLASSES)

    train_feats = extract_feature_matrix(train_bal, OUT / "binary_train_features.pkl")
    test_feats = extract_feature_matrix(test_bal, OUT / "binary_test_features.pkl")

    label_map = {"Non Demented": 0, "Very mild Dementia": 1}
    drop_cols = ["path", "class", "subject_id", "scan_id"]
    feature_cols = [c for c in train_feats.columns if c not in drop_cols]
    X_train = train_feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_feats["class"].map(label_map).to_numpy()
    y_test = test_feats["class"].map(label_map).to_numpy()

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "task": "Non Demented vs Very mild Dementia",
        "split": "patient-level stratified by subject_id",
        "seed": SEED,
        "train_size": int(len(train_feats)),
        "test_size": int(len(test_feats)),
        "train_class_counts": train_feats["class"].value_counts().to_dict(),
        "test_class_counts": test_feats["class"].value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "classification_report": classification_report(y_test, pred, target_names=BINARY_CLASSES, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    perm = permutation_importance(
        clf,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=8,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "rf_impurity_importance": clf.feature_importances_,
        }
    )
    importance_df["category"] = importance_df["feature"].map(feature_category)
    importance_df = importance_df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    importance_df.to_csv(OUT / "feature_importance.csv", index=False)

    category_importance = (
        importance_df.assign(positive=lambda d: d["permutation_importance_mean"].clip(lower=0))
        .groupby("category", as_index=False)["positive"]
        .sum()
        .rename(columns={"positive": "positive_permutation_importance"})
        .sort_values("positive_permutation_importance", ascending=False)
    )
    category_importance.to_csv(OUT / "category_importance.csv", index=False)

    plot_confusion(np.array(metrics["confusion_matrix"]), BINARY_CLASSES)
    plot_top_importance(importance_df)
    plot_roi_heatmap(importance_df)

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_report(meta, train_feats, test_feats, metrics, importance_df, category_importance)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
