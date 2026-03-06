# liver-disease-ml

# Liver Disease Classification

A machine-learning pipeline that classifies patients as **liver patients** or **healthy** using the [Indian Liver Patient Dataset (ILPD)](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)).

## Project Structure

```
liver-disease-ml/
├── data/
│   ├── __init__.py
│   └── download_dataset.py   # Download or generate the ILPD dataset
├── models/                   # Saved model artefacts (.pkl) – created at runtime
├── plots/                    # Generated visualisation plots  – created at runtime
├── tests/
│   ├── __init__.py
│   └── test_liver_classification.py
├── models.py                 # Classifier definitions, training & evaluation
├── preprocessing.py          # Data loading, cleaning, encoding, scaling & splitting
├── train.py                  # End-to-end training pipeline (entry point)
├── visualization.py          # Exploratory and results visualisation helpers
├── requirements.txt
└── README.md
```

## Dataset

The **Indian Liver Patient Dataset (ILPD)** contains 583 patient records collected from northeastern Andhra Pradesh, India.

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Male / Female |
| Total\_Bilirubin | Total bilirubin level |
| Direct\_Bilirubin | Direct bilirubin level |
| Alkaline\_Phosphotase | Alkaline phosphotase enzyme |
| Alamine\_Aminotransferase | SGPT enzyme level |
| Aspartate\_Aminotransferase | SGOT enzyme level |
| Total\_Proteins | Total protein level |
| Albumin | Albumin level |
| Albumin\_and\_Globulin\_Ratio | A/G ratio |
| **Dataset** | **1** = Liver Patient · **2** = Healthy |

If the dataset cannot be downloaded from the UCI ML Repository, the pipeline automatically generates a statistically representative synthetic dataset.

## Models

Four classifiers are trained and compared:

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| SVM (RBF kernel) | Support Vector Machine |
| Gradient Boosting | Boosted ensemble of shallow trees |

## Results

Sample results on the synthetic dataset (80/20 train-test split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.97** | **0.98** | **0.99** | **0.98** | **1.00** |
| Gradient Boosting | 0.97 | 0.98 | 0.98 | 0.98 | 1.00 |
| SVM | 0.93 | 0.94 | 0.96 | 0.95 | 0.98 |
| Logistic Regression | 0.93 | 0.95 | 0.95 | 0.95 | 0.99 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train with the default dataset (downloads or generates automatically)
python train.py

# Specify a custom CSV path
python train.py --data path/to/ilpd.csv

# Skip plot generation
python train.py --no-plots

# Adjust train/test split and random seed
python train.py --test-size 0.25 --random-state 0
```

The best model (by F1-score) is saved to `models/`.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Plots

The pipeline generates the following plots in the `plots/` directory:

- `class_distribution.png` – Bar chart of class counts
- `correlation_heatmap.png` – Pearson correlation between features
- `feature_distributions.png` – Per-class KDE / bar distributions for each feature
- `model_comparison.png` – Grouped bar chart across all metrics and models
- `roc_curves.png` – Overlay ROC curves for all models
- `confusion_matrix_<model>.png` – Per-model confusion matrix
- `feature_importance_<model>.png` – Feature importances (tree-based models)
