import os
import numpy as np
import pandas as pd
import warnings
import torch

# Import the provided library modules
import library.config as config
import library.image_utils as image_utils
import library.feature_pipeline as feature_pipeline
import library.model_wrapper as model_wrapper


def main():
    print("=== OSIC Pulmonary Fibrosis Progression - Pipeline Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. SETUP MINI-DATASET (Optimization for Speed)
    # ------------------------------------------------------------------------
    # We create a small subset of the data to ensure the demo runs quickly.
    # We will patch the library modules to use these temporary metadata files.

    print("[1/5] Creating mini-dataset for rapid execution...")
    working_dir = "./working"
    demo_dir = os.path.join(working_dir, "demo_run")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Load original metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Select a small number of patients (e.g., 3 from train, 2 from val/test)
    # We filter by Patient ID to keep all visits for selected patients
    train_pids = df_train["Patient"].unique()[:3]
    val_pids = df_val["Patient"].unique()[:2]
    test_pids = df_test["Patient"].unique()[:2]

    mini_train = df_train[df_train["Patient"].isin(train_pids)].copy()
    mini_val = df_val[df_val["Patient"].isin(val_pids)].copy()
    mini_test = df_test[df_test["Patient"].isin(test_pids)].copy()

    # Save mini metadata files
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"    Mini-Train Samples: {len(mini_train)}")
    print(f"    Mini-Val Samples:   {len(mini_val)}")
    print(f"    Mini-Test Samples:  {len(mini_test)}")

    # ------------------------------------------------------------------------
    # 2. PATCH LIBRARY CONFIGURATION
    # ------------------------------------------------------------------------
    # We dynamically update the module variables to point to our mini-dataset
    # and temporary cache. This avoids modifying the library files directly.

    print("\n[2/5] Patching library configuration...")

    # Patch Feature Pipeline paths
    feature_pipeline.TRAIN_METADATA_PATH = mini_train_path
    feature_pipeline.VAL_METADATA_PATH = mini_val_path
    feature_pipeline.TEST_METADATA_PATH = mini_test_path
    feature_pipeline.CACHE_DIR = demo_cache_dir

    # Patch Image Utils cache
    image_utils.CACHE_DIR = demo_cache_dir

    # Patch Model Wrapper cache
    model_wrapper.CACHE_DIR = demo_cache_dir

    # Patch Config seed (optional, but good practice)
    config.seed_everything(42)

    # ------------------------------------------------------------------------
    # 3. DEMONSTRATE IMAGE PROCESSING (image_utils)
    # ------------------------------------------------------------------------
    print("\n[3/5] Testing Image Processing (image_utils.process_patient)...")

    # Select a sample patient from our mini-train set
    sample_pid = train_pids[0]
    sample_path = mini_train[mini_train["Patient"] == sample_pid].iloc[0]["dcm_path"]

    print(f"    Processing Patient: {sample_pid}")
    print(f"    DICOM Path: {sample_path}")

    # Run processing (force computation by ignoring cache)
    img_tensor, volumetrics = image_utils.process_patient(
        sample_pid, sample_path, load_cached_data=False
    )

    print(f"    Output Image Shape: {img_tensor.shape}")
    print(f"    Output Volumetrics: {volumetrics}")

    # Validation
    # Image should be (3 channels, 224, 224)
    assert img_tensor.shape == (
        3,
        224,
        224,
    ), f"Unexpected image shape: {img_tensor.shape}"
    # Volumetrics should be (Volume, Density)
    assert volumetrics.shape == (
        2,
    ), f"Unexpected volumetrics shape: {volumetrics.shape}"
    # Image should be normalized [0, 1]
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image not normalized to [0, 1]"

    print("    -> Image processing validation passed.")

    # ------------------------------------------------------------------------
    # 4. DEMONSTRATE FEATURE PIPELINE (feature_pipeline)
    # ------------------------------------------------------------------------
    print("\n[4/5] Running Feature Pipeline...")

    # Run the full pipeline on the mini-dataset
    # This extracts CNN features, computes PCA, and builds tabular features
    train_data, val_data, test_data = feature_pipeline.run_feature_pipeline(
        load_cached_data=False
    )

    # Unpack Training Data
    X_fvc_train, y_fvc_train, X_unc_train, df_train_proc = train_data

    print(f"    Train FVC Feature Matrix (X): {X_fvc_train.shape}")
    print(f"    Train Target Vector (y):      {y_fvc_train.shape}")
    print(f"    Train Uncertainty Matrix (X): {X_unc_train.shape}")

    # Validation
    assert len(X_fvc_train) == len(mini_train), "Mismatch in training sample count"
    assert not np.isnan(X_fvc_train).any(), "NaN values detected in feature matrix"
    assert y_fvc_train is not None, "Training targets are missing"

    print("    -> Feature pipeline validation passed.")

    # ------------------------------------------------------------------------
    # 5. DEMONSTRATE MODEL TRAINING & INFERENCE (model_wrapper)
    # ------------------------------------------------------------------------
    print("\n[5/5] Training Model and Running Inference...")

    # Unpack Validation Data
    X_fvc_val, y_fvc_val, X_unc_val, df_val_proc = val_data

    # Initialize the Stratified Quantile GLM
    # We use fewer iterations for the demo to ensure speed
    model = model_wrapper.StratifiedQuantileGLM(quantile=0.5, max_iter=100)

    # Fit the model
    print("    Fitting model...")
    model.fit(X_fvc_train, y_fvc_train, X_unc_train)

    # Predict on Validation Set
    print("    Predicting on validation set...")
    val_fvc_pred, val_sigma_pred = model.predict(X_fvc_val, X_unc_val)

    # Display sample predictions
    print(f"    Sample Predictions (FVC):  {val_fvc_pred[:3]}")
    print(f"    Sample Confidence (Sigma): {val_sigma_pred[:3]}")

    # Validate Predictions
    assert len(val_fvc_pred) == len(mini_val), "Prediction length mismatch"
    assert np.all(val_sigma_pred > 0), "Confidence values must be positive"
    assert np.isfinite(val_fvc_pred).all(), "Non-finite FVC predictions detected"

    # Calculate Metric
    print("    Calculating Laplace Log Likelihood...")
    score = model_wrapper.calculate_laplace_metric(
        y_fvc_val, val_fvc_pred, val_sigma_pred
    )

    print(f"    Validation Score: {score:.4f}")

    # Validate Score
    assert isinstance(score, float), "Score must be a float"
    assert not np.isnan(score), "Score is NaN"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
