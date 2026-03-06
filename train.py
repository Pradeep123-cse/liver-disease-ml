"""
Main pipeline for liver disease classification.

Usage
-----
    python train.py                  # use default dataset path
    python train.py --data path/to/ilpd.csv
    python train.py --no-plots       # skip visualisations

The script will:
    1. Download (or generate) the ILPD dataset if it is not present.
    2. Preprocess the data.
    3. Train and evaluate four classifiers.
    4. Save all plots to the ``plots/`` directory.
    5. Save the best model to the ``models/`` directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from data.download_dataset import DATASET_PATH, download_dataset
from models import build_models, get_best_model, save_model, train_and_evaluate
from preprocessing import get_feature_names, load_data, preprocess
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liver Disease Classification Pipeline")
    parser.add_argument(
        "--data",
        default=DATASET_PATH,
        help="Path to the ILPD CSV dataset (default: data/ilpd.csv)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualisation plots",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Ensure dataset is available                                       #
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.data):
        print(f"Dataset not found at '{args.data}'. Attempting download...")
        download_dataset(args.data)

    # ------------------------------------------------------------------ #
    # 2. Load and preprocess                                               #
    # ------------------------------------------------------------------ #
    print("\nLoading dataset...")
    df = load_data(args.data)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()}")

    X_train, X_test, y_train, y_test, scaler = preprocess(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples    : {len(X_test)}")
    print(f"  Class balance (train) – Liver: {y_train.sum()}, Healthy: {(y_train==0).sum()}")

    # ------------------------------------------------------------------ #
    # 3. Optional: visualise raw data                                      #
    # ------------------------------------------------------------------ #
    if not args.no_plots:
        print("\nGenerating exploratory plots...")
        numeric_df = df.copy()
        numeric_df["Gender"] = numeric_df["Gender"].map({"Male": 1, "Female": 0})
        plot_class_distribution(y_train, )
        plot_correlation_heatmap(numeric_df)
        feature_cols = get_feature_names()
        plot_feature_distributions(df, feature_cols)

    # ------------------------------------------------------------------ #
    # 4. Train and evaluate all models                                     #
    # ------------------------------------------------------------------ #
    print("\n\nTraining and evaluating classifiers...")
    classifiers = build_models()
    results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------ #
    # 5. Summary table                                                     #
    # ------------------------------------------------------------------ #
    print("\n\n" + "=" * 60)
    print("SUMMARY – All Models")
    print("=" * 60)
    summary = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ #
    # 6. Model comparison plot                                             #
    # ------------------------------------------------------------------ #
    if not args.no_plots:
        print("\nGenerating model comparison plots...")
        plot_model_comparison(results)
        plot_roc_curves(classifiers, X_test, y_test)
        for name, model in classifiers.items():
            plot_confusion_matrix(model, X_test, y_test, name)
            plot_feature_importance(model, get_feature_names(), name)

    # ------------------------------------------------------------------ #
    # 7. Save the best model                                               #
    # ------------------------------------------------------------------ #
    best_name, best_model = get_best_model(classifiers, results, metric="f1")
    print(f"\nBest model by F1-score: {best_name}  (F1={results[best_name]['f1']:.4f})")
    save_model(best_model, best_name)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
