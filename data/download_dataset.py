"""
Download or generate the Indian Liver Patient Dataset (ILPD).

The ILPD dataset contains 583 records collected from northeastern Andhra Pradesh, India.
Columns:
    Age                         - Age of the patient
    Gender                      - Gender of the patient (Male/Female)
    Total_Bilirubin             - Total Bilirubin level
    Direct_Bilirubin            - Direct Bilirubin level
    Alkaline_Phosphotase        - Alkaline Phosphotase level
    Alamine_Aminotransferase    - Alamine Aminotransferase (SGPT)
    Aspartate_Aminotransferase  - Aspartate Aminotransferase (SGOT)
    Total_Proteins              - Total Proteins level
    Albumin                     - Albumin level
    Albumin_and_Globulin_Ratio  - Albumin and Globulin Ratio
    Dataset                     - 1 = liver patient, 2 = non-liver patient
"""

import os
import urllib.request

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "ilpd.csv")

COLUMN_NAMES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Dataset",
]


def download_dataset(dest_path: str = DATASET_PATH) -> str:
    """
    Download the ILPD dataset from UCI ML Repository.

    Args:
        dest_path: Path to save the dataset CSV file.

    Returns:
        Path to the saved dataset file.
    """
    print(f"Downloading ILPD dataset to {dest_path} ...")
    try:
        urllib.request.urlretrieve(DATASET_URL, dest_path)
        print("Download complete.")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Generating synthetic dataset instead...")
        _generate_synthetic_dataset(dest_path)
    return dest_path


def _generate_synthetic_dataset(dest_path: str) -> None:
    """
    Generate a synthetic dataset with the same structure as ILPD when the
    download is not available. Values are sampled from distributions that
    approximate the real dataset statistics.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_patients = 416
    n_healthy = 167
    n_total = n_patients + n_healthy

    def _clip(arr, low, high):
        return np.clip(arr, low, high)

    # Liver patients
    age_p = _clip(rng.normal(44, 16, n_patients).astype(int), 4, 90)
    gender_p = rng.choice(["Male", "Female"], n_patients, p=[0.75, 0.25])
    tb_p = _clip(rng.lognormal(1.0, 1.2, n_patients), 0.4, 75)
    db_p = _clip(rng.lognormal(0.0, 1.2, n_patients), 0.1, 19)
    ap_p = _clip(rng.lognormal(5.5, 0.8, n_patients).astype(int), 63, 2110)
    alt_p = _clip(rng.lognormal(4.2, 1.2, n_patients).astype(int), 10, 2000)
    ast_p = _clip(rng.lognormal(4.4, 1.2, n_patients).astype(int), 10, 4929)
    tp_p = _clip(rng.normal(6.5, 1.1, n_patients), 2.7, 9.6)
    alb_p = _clip(rng.normal(3.1, 0.7, n_patients), 0.9, 5.5)
    agr_p = _clip(rng.normal(0.9, 0.3, n_patients), 0.3, 2.8)
    label_p = np.ones(n_patients, dtype=int)

    # Healthy individuals
    age_h = _clip(rng.normal(40, 15, n_healthy).astype(int), 4, 90)
    gender_h = rng.choice(["Male", "Female"], n_healthy, p=[0.70, 0.30])
    tb_h = _clip(rng.lognormal(0.3, 0.5, n_healthy), 0.4, 3.5)
    db_h = _clip(rng.lognormal(-0.5, 0.5, n_healthy), 0.1, 1.2)
    ap_h = _clip(rng.lognormal(5.0, 0.4, n_healthy).astype(int), 63, 500)
    alt_h = _clip(rng.lognormal(3.0, 0.5, n_healthy).astype(int), 10, 100)
    ast_h = _clip(rng.lognormal(3.2, 0.5, n_healthy).astype(int), 10, 120)
    tp_h = _clip(rng.normal(7.0, 0.8, n_healthy), 5.5, 9.6)
    alb_h = _clip(rng.normal(3.8, 0.5, n_healthy), 2.4, 5.5)
    agr_h = _clip(rng.normal(1.2, 0.3, n_healthy), 0.5, 2.8)
    label_h = np.full(n_healthy, 2, dtype=int)

    data = {
        "Age": np.concatenate([age_p, age_h]),
        "Gender": np.concatenate([gender_p, gender_h]),
        "Total_Bilirubin": np.round(np.concatenate([tb_p, tb_h]), 1),
        "Direct_Bilirubin": np.round(np.concatenate([db_p, db_h]), 1),
        "Alkaline_Phosphotase": np.concatenate([ap_p, ap_h]),
        "Alamine_Aminotransferase": np.concatenate([alt_p, alt_h]),
        "Aspartate_Aminotransferase": np.concatenate([ast_p, ast_h]),
        "Total_Proteins": np.round(np.concatenate([tp_p, tp_h]), 1),
        "Albumin": np.round(np.concatenate([alb_p, alb_h]), 1),
        "Albumin_and_Globulin_Ratio": np.round(np.concatenate([agr_p, agr_h]), 2),
        "Dataset": np.concatenate([label_p, label_h]),
    }

    df = pd.DataFrame(data)
    # Introduce ~4 missing values in Albumin_and_Globulin_Ratio (mirrors real data)
    missing_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[missing_idx, "Albumin_and_Globulin_Ratio"] = None

    df.to_csv(dest_path, index=False, header=False)
    print(f"Synthetic dataset written to {dest_path} ({len(df)} records).")


if __name__ == "__main__":
    download_dataset()
