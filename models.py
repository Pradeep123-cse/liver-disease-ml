"""
Classification models for liver disease prediction.

Provides helpers to build, train and evaluate multiple scikit-learn classifiers,
as well as utilities to persist (save/load) trained models.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def build_models() -> Dict[str, object]:
    """
    Return a dictionary of classifier instances with sensible default hyper-
    parameters.

    Returns:
        Mapping of model name → unfitted classifier.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }


def train_and_evaluate(
    models: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Train each model and compute evaluation metrics on the test set.

    Args:
        models: Mapping returned by :func:`build_models`.
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Nested mapping of model name → metrics dict with keys:
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        results[name] = metrics
        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        for metric, value in metrics.items():
            print(f"  {metric:<12}: {value:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Healthy (0)", "Liver Patient (1)"],
                zero_division=0,
            )
        )

    return results


def get_best_model(
    models: Dict[str, object],
    results: Dict[str, Dict[str, float]],
    metric: str = "f1",
) -> Tuple[str, object]:
    """
    Return the name and fitted instance of the best-performing model.

    Args:
        models: Dictionary of fitted classifiers.
        results: Metrics returned by :func:`train_and_evaluate`.
        metric: The metric to rank models by (default ``"f1"``).

    Returns:
        Tuple of ``(best_name, best_model)``.
    """
    best_name = max(results, key=lambda name: results[name].get(metric, 0.0))
    return best_name, models[best_name]


def save_model(model: object, name: str, directory: str = MODELS_DIR) -> str:
    """
    Persist a fitted model to disk using joblib.

    Args:
        model: Fitted scikit-learn estimator.
        name: Descriptive name used in the filename (spaces replaced by ``_``).
        directory: Directory to save the file in.

    Returns:
        Full path of the saved file.
    """
    os.makedirs(directory, exist_ok=True)
    filename = name.replace(" ", "_").lower() + ".pkl"
    path = os.path.join(directory, filename)
    joblib.dump(model, path)
    print(f"Model saved → {path}")
    return path


def load_model(path: str) -> object:
    """
    Load a previously saved model from disk.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        Fitted scikit-learn estimator.
    """
    model = joblib.load(path)
    print(f"Model loaded ← {path}")
    return model
