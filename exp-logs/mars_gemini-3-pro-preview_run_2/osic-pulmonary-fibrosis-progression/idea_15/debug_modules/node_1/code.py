import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
import library.config as config
import library.utils as utils
import library.model_pipeline as pipeline_module
from library.dicom_processor import DicomProcessor
from library.feature_generator import FeatureGenerator
from library.model_pipeline import LungFunctionPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Lung Function Prediction Demo ===")

    # 1. Setup and Reproducibility
    utils.seed_everything(config.SEED)
    working_dir = config.WORKING_DIR
    os.makedirs(working_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Metric Verification
    # ---------------------------------------------------------
    print("\n[1/5] Verifying Metric Function...")
    y_true = np.array([2000, 3000])
    y_pred = np.array([2100, 2900])  # Delta = 100 for both
    sigma = np.array([100, 50])  # Sigma clipped: 100, 70 (min is 70)

    # Calculation:
    # Sample 1: Delta=100, Sigma=100. Metric = -sqrt(2)*100/100 - ln(sqrt(2)*100)
    #           = -1.414 - ln(141.4) = -1.414 - 4.95 = -6.36
    # Sample 2: Delta=100, Sigma=70.  Metric = -sqrt(2)*100/70 - ln(sqrt(2)*70)
    #           = -2.02 - ln(98.99) = -2.02 - 4.59 = -6.61
    # Mean approx -6.48

    score = utils.laplace_log_likelihood(y_true, y_pred, sigma)
    print(f"   Calculated Score: {score:.4f}")
    assert score < 0, "Metric should be negative"
    assert np.isfinite(score), "Metric should be finite"
    print("   Metric verification passed.")

    # ---------------------------------------------------------
    # 3. Create Mini Datasets for Speed
    # ---------------------------------------------------------
    print("\n[2/5] Creating Mini Datasets (Subset)...")

    # Load original metadata
    df_train_orig = pd.read_csv(config.TRAIN_META_PATH)
    df_val_orig = pd.read_csv(config.VAL_META_PATH)
    df_test_orig = pd.read_csv(config.TEST_META_PATH)

    # Select small subset of patients
    # Train: 5 patients, Val: 2 patients, Test: 2 patients
    train_patients = df_train_orig["Patient"].unique()[:5]
    val_patients = df_val_orig["Patient"].unique()[:2]
    test_patients = df_test_orig["Patient"].unique()[:2]

    df_train_mini = df_train_orig[df_train_orig["Patient"].isin(train_patients)].copy()
    df_val_mini = df_val_orig[df_val_orig["Patient"].isin(val_patients)].copy()
    df_test_mini = df_test_orig[df_test_orig["Patient"].isin(test_patients)].copy()

    # Save to working directory
    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_val_path = os.path.join(working_dir, "mini_val.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    print(f"   Mini Train Samples: {len(df_train_mini)}")
    print(f"   Mini Val Samples:   {len(df_val_mini)}")
    print(f"   Mini Test Samples:  {len(df_test_mini)}")

    # ---------------------------------------------------------
    # 4. Patch Library Paths
    # ---------------------------------------------------------
    # We need to redirect the library to use our mini datasets.
    # Since the paths are imported as constants in model_pipeline, we patch them there.
    pipeline_module.TRAIN_META_PATH = mini_train_path
    pipeline_module.VAL_META_PATH = mini_val_path
    pipeline_module.TEST_META_PATH = mini_test_path

    # Also patch config just in case other modules reference it directly later
    config.TRAIN_META_PATH = mini_train_path
    config.VAL_META_PATH = mini_val_path
    config.TEST_META_PATH = mini_test_path

    print("   Library paths patched to use mini datasets.")

    # ---------------------------------------------------------
    # 5. Component Testing: DicomProcessor & FeatureGenerator
    # ---------------------------------------------------------
    print("\n[3/5] Testing Image Processing Components...")

    # Pick a sample patient from training
    sample_patient = train_patients[0]
    sample_row = df_train_mini[df_train_mini["Patient"] == sample_patient].iloc[0]
    dcm_path = sample_row["dcm_path"]

    print(f"   Processing Patient: {sample_patient}")

    # A. DicomProcessor
    dicom_proc = DicomProcessor()
    # Force load_cached_data=False to test processing logic
    images, histogram = dicom_proc.process_patient(
        sample_patient, dcm_path, load_cached_data=False
    )

    print(f"   Image Shape: {images.shape}")
    print(f"   Histogram: {histogram}")

    # Assertions
    expected_slices = config.SLICES_PER_AXIS * 2
    assert images.shape == (
        expected_slices,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), f"Expected image shape {(expected_slices, config.IMAGE_SIZE, config.IMAGE_SIZE)}, got {images.shape}"
    assert histogram.shape == (
        4,
    ), f"Expected histogram shape (4,), got {histogram.shape}"
    assert np.all(
        (images >= 0) & (images <= 1)
    ), "Images should be normalized to [0, 1]"

    # B. FeatureGenerator
    feat_gen = FeatureGenerator()
    # Force load_cached_data=False
    features = feat_gen.generate_patient_features(
        sample_patient, dcm_path, sample_row, load_cached_data=False
    )

    print(f"   Feature Vector Shape: {features.shape}")

    # Expected dimensions:
    # EfficientNet-B0 (1280) * Num Slices + Histogram (4) + Clinical (7)
    num_slices = config.SLICES_PER_AXIS * 2
    expected_dim = (1280 * num_slices) + 4 + 7
    assert features.shape == (
        expected_dim,
    ), f"Expected feature dim {expected_dim}, got {features.shape}"
    print("   Component verification passed.")

    # ---------------------------------------------------------
    # 6. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n[4/5] Running Full Model Pipeline...")

    predictor = LungFunctionPredictor()

    # Run data preparation (generates features for all mini-dataset patients)
    # We disable cache loading to ensure the code runs through the generation logic
    data = predictor.prepare_data(load_cached_data=False)

    # Verify Data Matrices
    X_train = data["X_fvc_train"]
    print(f"   Train Matrix Shape: {X_train.shape}")
    # PCA (40) + Weeks (1) + Interactions (40) = 81 columns for FVC model
    assert (
        X_train.shape[1] == config.PCA_COMPONENTS * 2 + 1
    ), "Incorrect FVC feature matrix dimensions"

    # Fit Models
    predictor.fit(data)

    # Predict on Test
    sub_df = predictor.predict(data)

    # ---------------------------------------------------------
    # 7. Submission Validation
    # ---------------------------------------------------------
    print("\n[5/5] Validating Submission...")

    # Check file existence
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    # Check content
    print(sub_df.head())

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission"

    # Check row count matches mini test set
    assert len(sub_df) == len(
        df_test_mini
    ), f"Submission rows {len(sub_df)} != Test rows {len(df_test_mini)}"

    # Check value validity
    assert sub_df["FVC"].notna().all(), "NaN found in FVC predictions"
    assert sub_df["Confidence"].notna().all(), "NaN found in Confidence predictions"
    assert (sub_df["Confidence"] >= 0).all(), "Confidence cannot be negative"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
