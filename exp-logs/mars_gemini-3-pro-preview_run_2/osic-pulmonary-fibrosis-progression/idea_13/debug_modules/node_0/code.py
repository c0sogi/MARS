import os
import shutil
import numpy as np
import pandas as pd
import warnings
import torch

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.dicom_processing import process_patient
from library.cnn_feature_extractor import VisualFeatureExtractor
from library.dataset_builder import DatasetBuilder
from library.regressors import QuantileMedianRegressor, ResidualUncertaintyRegressor


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("Initializing Demo...")
    seed_everything(42)
    warnings.filterwarnings("ignore")

    # Define paths for the demo run
    DEMO_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")

    # Clean up previous demo runs if they exist
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Create Mini-Dataset (Subsetting Metadata)
    # -------------------------------------------------------------------------
    print("Creating mini-dataset for rapid execution...")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select a small subset of patients (e.g., 5 Train, 3 Val, 3 Test)
    # We filter by Patient ID to ensure all weeks for a patient are included
    train_pats = orig_train["Patient"].unique()[:5]
    val_pats = orig_val["Patient"].unique()[:3]
    test_pats = orig_test["Patient"].unique()[:3]

    mini_train = orig_train[orig_train["Patient"].isin(train_pats)].copy()
    mini_val = orig_val[orig_val["Patient"].isin(val_pats)].copy()
    mini_test = orig_test[orig_test["Patient"].isin(test_pats)].copy()

    # Save mini-metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini-Train samples: {len(mini_train)}")
    print(f"Mini-Val samples: {len(mini_val)}")
    print(f"Mini-Test samples: {len(mini_test)}")

    # -------------------------------------------------------------------------
    # 3. Monkey-Patch Config for Demo
    # -------------------------------------------------------------------------
    # We modify the Config class attributes to point to our demo files and cache.
    # This affects the behavior of the library classes instantiated afterwards.
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # Reduce PCA components to fit the small dataset (cannot be > min(n_samples, n_features))
    Config.N_PCA_COMPONENTS = 10

    # -------------------------------------------------------------------------
    # 4. Demonstrate Individual Components
    # -------------------------------------------------------------------------

    # --- A. DICOM Processing ---
    print("\n[Demo] DICOM Processing")
    # Pick a sample patient
    sample_pat_id = train_pats[0]
    sample_dcm_rel_path = mini_train[mini_train["Patient"] == sample_pat_id].iloc[0][
        "dcm_path"
    ]

    print(f"Processing patient: {sample_pat_id}")
    # Force processing (ignore cache) to verify logic
    processed_images = process_patient(
        sample_pat_id, sample_dcm_rel_path, load_cached_data=False
    )

    print(f"Output Shape: {processed_images.shape}")

    # Verification
    # Expected shape: (Total_Images, IMG_SIZE, IMG_SIZE) -> (6, 224, 224)
    assert processed_images.shape == (
        6,
        224,
        224,
    ), f"Unexpected shape: {processed_images.shape}"
    assert processed_images.dtype == np.float32
    assert processed_images.min() >= 0.0 and processed_images.max() <= 1.0

    # --- B. CNN Feature Extraction ---
    print("\n[Demo] CNN Feature Extraction")
    extractor = VisualFeatureExtractor()

    # Extract features for the single patient processed above
    features = extractor.extract_single_patient(processed_images)
    print(f"Feature Vector Shape: {features.shape}")

    # Verification
    # EfficientNet-B0 output is 1280. We have 6 images. 1280 * 6 = 7680.
    assert features.shape == (7680,), f"Unexpected feature shape: {features.shape}"

    # -------------------------------------------------------------------------
    # 5. Pipeline Execution (DatasetBuilder)
    # -------------------------------------------------------------------------
    print("\n[Demo] Dataset Building Pipeline")
    builder = DatasetBuilder()

    # Generate datasets (this handles Volumetrics, CNN, PCA, and Tabular encoding)
    # We set load_cached_data=False to ensure the code runs fully.
    datasets = builder.generate_datasets(load_cached_data=False)

    train_data = datasets["train"]
    val_data = datasets["val"]
    test_data = datasets["test"]

    print("Dataset Shapes:")
    print(f"  Train X_fvc: {train_data['X_fvc'].shape}, y: {train_data['y'].shape}")
    print(f"  Val   X_fvc: {val_data['X_fvc'].shape}, y: {val_data['y'].shape}")
    print(f"  Test  X_fvc: {test_data['X_fvc'].shape}")

    # Verification
    # Check that Train and Val have consistent feature dimensions
    assert train_data["X_fvc"].shape[1] == val_data["X_fvc"].shape[1]
    assert train_data["X_unc"].shape[1] == val_data["X_unc"].shape[1]
    assert len(train_data["y"]) == len(mini_train)

    # -------------------------------------------------------------------------
    # 6. Model Training & Inference
    # -------------------------------------------------------------------------
    print("\n[Demo] Model Training")

    # --- A. Median FVC Prediction ---
    # We use a Quantile Regressor for the median (0.5)
    fvc_model = QuantileMedianRegressor(alpha=0.01)  # Small regularization
    fvc_model.fit(train_data["X_fvc"], train_data["y"])

    train_preds = fvc_model.predict(train_data["X_fvc"])
    val_preds = fvc_model.predict(val_data["X_fvc"])

    print(f"  Sample Val Preds: {val_preds[:3]}")

    # --- B. Uncertainty Prediction ---
    # Calculate residuals from the training set
    train_residuals = train_data["y"] - train_preds

    # Train Uncertainty model on the residuals
    unc_model = ResidualUncertaintyRegressor(alpha=0.1)
    unc_model.fit(train_data["X_unc"], train_residuals)

    val_sigma = unc_model.predict(val_data["X_unc"])
    print(f"  Sample Val Sigma: {val_sigma[:3]}")

    # -------------------------------------------------------------------------
    # 7. Evaluation
    # -------------------------------------------------------------------------
    print("\n[Demo] Evaluation")
    metric_score = laplace_log_likelihood(val_data["y"], val_preds, val_sigma)
    print(f"  Validation Laplace Log Likelihood: {metric_score:.4f}")

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Demo] Generating Submission")

    # Predict on Test Set
    test_preds = fvc_model.predict(test_data["X_fvc"])
    test_sigma = unc_model.predict(test_data["X_unc"])

    # Construct Submission DataFrame
    submission = test_data["meta"].copy()
    submission["FVC"] = test_preds
    submission["Confidence"] = test_sigma

    # Filter to required columns
    submission = submission[["Patient_Week", "FVC", "Confidence"]]

    # Display and Save
    print(submission.head())

    sub_path = os.path.join(DEMO_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved to: {sub_path}")

    print("\nDemo Complete!")


if __name__ == "__main__":
    run_demo()
