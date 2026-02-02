import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, score_func
from library.data_manager import DataManager
from library.linear_models import DualModel, generate_submission


def create_mini_metadata():
    """
    Creates small subsets of the metadata files to speed up the demonstration.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Define paths
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset (Take top 10 patients for train, 5 for val, 5 for test)
    # Note: We group by patient to keep patient records together
    train_patients = df_train["Patient"].unique()[:5]
    val_patients = df_val["Patient"].unique()[:2]
    test_patients = df_test["Patient"].unique()[:2]

    df_train_mini = df_train[df_train["Patient"].isin(train_patients)].copy()
    df_val_mini = df_val[df_val["Patient"].isin(val_patients)].copy()
    df_test_mini = df_test[df_test["Patient"].isin(test_patients)].copy()

    # Save mini metadata
    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    return (
        mini_train_path,
        mini_val_path,
        mini_test_path,
        len(df_train_mini),
        len(df_val_mini),
        len(df_test_mini),
    )


def patch_config(train_path, val_path, test_path):
    """
    Patches the Config class to use mini-datasets and faster processing parameters.
    """
    print("Patching configuration for speed...")

    # Point to mini metadata
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Use a specific cache dir for this run to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce image processing load
    Config.N_SLICES = 2  # Reduce from 5 to 2
    Config.PCA_COMPONENTS = 5  # Reduce from 30 to 5
    Config.BATCH_SIZE = 4

    # Ensure submission path is in working directory for safety
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Prepare Mini Data & Patch Config
    t_path, v_path, ts_path, n_train, n_val, n_test = create_mini_metadata()
    patch_config(t_path, v_path, ts_path)

    print(f"Mini-dataset sizes: Train={n_train}, Val={n_val}, Test={n_test}")

    # 3. Data Management Pipeline
    print("\n--- Initializing Data Manager ---")
    dm = DataManager(device=Config.DEVICE)

    # Force re-computation (load_cached_data=False) to demonstrate the pipeline logic
    # In a real run, we might use True, but here we want to prove extraction works.
    print("Running data preparation (Feature Extraction + PCA + Preprocessing)...")
    train_data, val_data, test_data = dm.prepare_data(load_cached_data=False)

    # 4. Verify Data Integrity
    print("\n--- Verifying Data Integrity ---")

    # Check sample counts
    assert (
        len(train_data["y"]) == n_train
    ), f"Expected {n_train} training samples, got {len(train_data['y'])}"
    assert (
        len(val_data["y"]) == n_val
    ), f"Expected {n_val} validation samples, got {len(val_data['y'])}"
    assert (
        len(test_data["patient_week"]) == n_test
    ), f"Expected {n_test} test samples, got {len(test_data['patient_week'])}"

    # Check Feature Matrix Shapes
    # X_fvc structure: [X_base, t, X_base * t]
    # X_base dim = PCA_COMPONENTS (5) + OneHot_Clinical (approx 5-7 columns)
    # Let's just ensure they are 2D arrays and not empty
    assert train_data["X_fvc"].ndim == 2
    assert train_data["X_unc"].ndim == 2
    assert train_data["X_fvc"].shape[1] > Config.PCA_COMPONENTS

    print("Data shapes verified successfully.")

    # 5. Model Training
    print("\n--- Training Dual Model ---")
    model = DualModel(fvc_alpha=0.5, unc_alpha=0.1)

    model.fit(train_data, val_data)

    # 6. Verify Predictions & Metric
    print("\n--- Verifying Predictions ---")
    # Predict on validation set manually to check values
    fvc_pred, sigma_pred = model.predict(val_data)

    # Check for NaNs or Infs
    assert np.all(np.isfinite(fvc_pred)), "FVC predictions contain NaNs or Infs"
    assert np.all(np.isfinite(sigma_pred)), "Sigma predictions contain NaNs or Infs"

    # Check Sigma is non-negative (DualModel handles this, but good to verify)
    assert np.all(sigma_pred >= 0), "Sigma predictions must be non-negative"

    # Calculate score manually
    score = score_func(val_data["y"], fvc_pred, sigma_pred)
    print(f"Manual Validation Score Check: {score:.4f}")
    assert isinstance(score, float), "Score should be a float"

    # 7. Generate Submission
    print("\n--- Generating Submission ---")
    generate_submission(model, test_data)

    # 8. Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(sub_df)}")

    assert (
        len(sub_df) == n_test
    ), f"Submission should have {n_test} rows, found {len(sub_df)}"
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Submission missing required columns"

    # Check if confidence clipping worked (min 70)
    min_conf = sub_df["Confidence"].min()
    assert (
        min_conf >= 70
    ), f"Confidence values should be clipped at 70, found min {min_conf}"

    print("\n=== Demonstration Completed Successfully ===")
