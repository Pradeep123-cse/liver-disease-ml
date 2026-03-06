"""
Visualization utilities for the liver disease classification project.

All plot functions save figures to a configurable output directory and also
return the ``matplotlib.figure.Figure`` object so callers can display or
further customise them.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _save(fig: plt.Figure, filename: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Plot saved → {path}")


def plot_class_distribution(
    y: np.ndarray,
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Bar chart of class counts (1 = liver patient, 0 = healthy)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, counts = np.unique(y, return_counts=True)
    label_names = {0: "Healthy", 1: "Liver Patient"}
    bars = ax.bar(
        [label_names.get(int(l), str(l)) for l in labels],
        counts,
        color=["#2196F3", "#F44336"],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    _save(fig, "class_distribution.png", output_dir)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Heatmap of Pearson correlations for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png", output_dir)
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Dataset",
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Overlapping KDE/bar plots for each feature, separated by class."""
    n_cols = 3
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    palette = {1: "#F44336", 2: "#2196F3"}
    label_map = {1: "Liver Patient", 2: "Healthy"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        for cls, color in palette.items():
            subset = df[df[target_col] == cls][col].dropna()
            if is_numeric:
                sns.kdeplot(
                    subset,
                    ax=ax,
                    fill=True,
                    alpha=0.4,
                    color=color,
                    label=label_map[cls],
                    warn_singular=False,
                )
            else:
                counts = subset.value_counts(normalize=True)
                ax.bar(
                    [str(v) for v in counts.index],
                    counts.values,
                    alpha=0.6,
                    color=color,
                    label=label_map[cls],
                )
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "feature_distributions.png", output_dir)
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Grouped bar chart comparing models across multiple metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())

    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    for name in model_names:
        for m in metrics:
            metric_values[m].append(results[name].get(m, 0.0))

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, metric_values[metric], width, label=metric.upper(), color=color, alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    _save(fig, "model_comparison.png", output_dir)
    return fig


def plot_confusion_matrix(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Heatmap of the confusion matrix for a single model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy (0)", "Liver Patient (1)"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png", output_dir)
    return fig


def plot_roc_curves(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = PLOTS_DIR,
) -> plt.Figure:
    """Overlay ROC curves for all models that support probability estimates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "roc_curves.png", output_dir)
    return fig


def plot_feature_importance(
    model: object,
    feature_names: List[str],
    model_name: str,
    output_dir: str = PLOTS_DIR,
) -> Optional[plt.Figure]:
    """
    Horizontal bar chart of feature importances.
    Works for Random Forest and Gradient Boosting.  Returns ``None`` for
    models that do not expose ``feature_importances_``.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="#2196F3",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances – {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"feature_importance_{model_name.replace(' ', '_').lower()}.png", output_dir)
    return fig
