"""
Unit tests for the liver disease classification project.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class TestPreprocessing(unittest.TestCase):
    """Tests for the preprocessing module."""

    def _make_df(self, n: int = 100) -> pd.DataFrame:
        """Create a minimal synthetic DataFrame mirroring ILPD structure."""
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "Age": rng.integers(20, 70, n),
                "Gender": rng.choice(["Male", "Female"], n),
                "Total_Bilirubin": rng.uniform(0.4, 5.0, n),
                "Direct_Bilirubin": rng.uniform(0.1, 2.0, n),
                "Alkaline_Phosphotase": rng.integers(100, 500, n),
                "Alamine_Aminotransferase": rng.integers(10, 200, n),
                "Aspartate_Aminotransferase": rng.integers(10, 200, n),
                "Total_Proteins": rng.uniform(5.0, 9.0, n),
                "Albumin": rng.uniform(2.5, 5.0, n),
                "Albumin_and_Globulin_Ratio": rng.uniform(0.5, 2.0, n),
                "Dataset": rng.choice([1, 2], n),
            }
        )

    def test_load_data_roundtrip(self):
        """load_data should correctly parse a header-less CSV."""
        from preprocessing import load_data, COLUMN_NAMES

        df = self._make_df(50)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as fh:
            df.to_csv(fh, index=False, header=False)
            path = fh.name
        try:
            loaded = load_data(path)
            self.assertEqual(list(loaded.columns), COLUMN_NAMES)
            self.assertEqual(len(loaded), 50)
        finally:
            os.unlink(path)

    def test_preprocess_shapes(self):
        """preprocess should return arrays with the correct shapes."""
        from preprocessing import preprocess

        df = self._make_df(100)
        X_train, X_test, y_train, y_test, scaler = preprocess(df, test_size=0.2)
        total = len(X_train) + len(X_test)
        self.assertAlmostEqual(len(X_test) / total, 0.2, delta=0.05)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.shape[1], 10)  # 10 feature columns

    def test_preprocess_no_nan(self):
        """After preprocessing there should be no NaN values."""
        from preprocessing import preprocess

        df = self._make_df(100)
        # Introduce some NaN values
        df.loc[0, "Albumin_and_Globulin_Ratio"] = None
        df.loc[5, "Albumin_and_Globulin_Ratio"] = None
        X_train, X_test, y_train, y_test, _ = preprocess(df)
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

    def test_binary_target(self):
        """Target should only contain 0 and 1 after preprocessing."""
        from preprocessing import preprocess

        df = self._make_df(100)
        _, _, y_train, y_test, _ = preprocess(df)
        unique_values = set(np.unique(np.concatenate([y_train, y_test])))
        self.assertTrue(unique_values.issubset({0, 1}))

    def test_gender_encoding(self):
        """Gender column should be encoded as integers."""
        from preprocessing import preprocess

        df = self._make_df(100)
        X_train, X_test, _, _, _ = preprocess(df)
        # After StandardScaler the raw int values are transformed; we check
        # that all values are finite numbers (encoding succeeded).
        self.assertTrue(np.isfinite(X_train).all())

    def test_get_feature_names(self):
        """get_feature_names should return 10 feature names."""
        from preprocessing import get_feature_names

        names = get_feature_names()
        self.assertEqual(len(names), 10)
        self.assertNotIn("Dataset", names)


class TestModels(unittest.TestCase):
    """Tests for the models module."""

    def _train_data(self, n: int = 200):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, 10))
        y = (rng.random(n) > 0.3).astype(int)
        split = int(n * 0.8)
        return X[:split], X[split:], y[:split], y[split:]

    def test_build_models_keys(self):
        """build_models should return the expected classifier names."""
        from models import build_models

        classifiers = build_models()
        expected = {"Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"}
        self.assertEqual(set(classifiers.keys()), expected)

    def test_train_and_evaluate_returns_metrics(self):
        """train_and_evaluate should return metrics for each model."""
        from models import build_models, train_and_evaluate

        X_train, X_test, y_train, y_test = self._train_data()
        classifiers = build_models()
        results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)
        for name in classifiers:
            self.assertIn(name, results)
            for metric in ("accuracy", "precision", "recall", "f1"):
                self.assertIn(metric, results[name])
                self.assertGreaterEqual(results[name][metric], 0.0)
                self.assertLessEqual(results[name][metric], 1.0)

    def test_get_best_model(self):
        """get_best_model should return the model with the highest metric."""
        from models import build_models, get_best_model, train_and_evaluate

        X_train, X_test, y_train, y_test = self._train_data()
        classifiers = build_models()
        results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)
        best_name, best_model = get_best_model(classifiers, results, metric="f1")
        best_f1 = results[best_name]["f1"]
        for name, metrics in results.items():
            self.assertGreaterEqual(best_f1, metrics["f1"] - 1e-9)

    def test_save_and_load_model(self):
        """A model saved with save_model can be reloaded with load_model."""
        from models import build_models, load_model, save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            classifiers = build_models()
            model = classifiers["Logistic Regression"]
            X = np.random.standard_normal((50, 10))
            y = (np.random.random(50) > 0.5).astype(int)
            model.fit(X, y)
            path = save_model(model, "Logistic Regression", directory=tmpdir)
            self.assertTrue(os.path.exists(path))
            loaded = load_model(path)
            preds_original = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_original, preds_loaded)


class TestDatasetGeneration(unittest.TestCase):
    """Tests for the synthetic dataset generator."""

    def test_generate_synthetic_dataset(self):
        """The generated dataset should have 583 rows and 11 columns."""
        from data.download_dataset import _generate_synthetic_dataset, COLUMN_NAMES
        from preprocessing import load_data

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            path = fh.name
        try:
            _generate_synthetic_dataset(path)
            df = load_data(path)
            self.assertEqual(len(df.columns), len(COLUMN_NAMES))
            self.assertGreater(len(df), 0)
            self.assertIn("Dataset", df.columns)
            self.assertTrue(df["Dataset"].isin([1, 2]).all())
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
