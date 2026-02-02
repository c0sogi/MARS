import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, score_function, save_results
from library.image_processing import process_patient, load_scan, segment_lung
from library.feature_extractor import EfficientNetExtractor
from library.data_manager import DataManager
from library.model_pipeline import DecoupledQuantileModel, run_training_and_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # ------------------------------------------------------------------------
    # 0. Configuration & Setup
    # ------------------------------------------------------------------------
    print("--- Step 0: Configuration Setup ---")

    # Override Config for the demonstration to ensure speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10  # Only process 10 patients per split
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    Config.mkdirs()

    # Set seeds
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Cache Directory: {Config.CACHE_DIR}")
    print("Configuration initialized.\n")

    # ------------------------------------------------------------------------
    # 1. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("--- Step 1: Verifying Utility Functions ---")

    # Test score_function
    # Case 1: Perfect prediction
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    sigma = np.array([100, 100])  # > 70

    # Metric = - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100)
    #        = 0 - ln(141.42) ~= -4.95
    score = score_function(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Score function mismatch. Got {score}, expected {expected_score}"
    print("score_function: Verified.")

    # Test save_results
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    dummy_path = os.path.join(Config.SUBMISSION_DIR, "test_save.csv")
    save_results(dummy_df, dummy_path)
    assert os.path.exists(dummy_path), "save_results failed to create file."
    print("save_results: Verified.\n")

    # ------------------------------------------------------------------------
    # 2. Verify Image Processing
    # ------------------------------------------------------------------------
    print("--- Step 2: Verifying Image Processing ---")

    # Load metadata to find a valid patient
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    sample_patient = train_meta.iloc[0]["Patient"]
    sample_dcm_path = train_meta.iloc[0]["dcm_path"]

    print(f"Processing sample patient: {sample_patient}")

    # Test process_patient (this handles loading, masking, MIP, Zonal selection, Histogram)
    # We force load_cached_data=False to ensure the processing logic runs
    processed_data = process_patient(
        sample_patient, sample_dcm_path, load_cached_data=False
    )

    # Verify keys
    expected_keys = ["mip", "axial_1", "axial_2", "axial_3", "histogram"]
    for key in expected_keys:
        assert key in processed_data, f"Missing key {key} in processed data."

    # Verify Shapes and Value Ranges
    # Images should be (IMAGE_SIZE, IMAGE_SIZE) and normalized [0, 1]
    img_shape = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    for key in ["mip", "axial_1"]:
        img = processed_data[key]
        assert img.shape == img_shape, f"{key} shape mismatch: {img.shape}"
        assert (
            img.min() >= 0.0 and img.max() <= 1.0 + 1e-6
        ), f"{key} values out of range [0, 1]. Range: [{img.min()}, {img.max()}]"

    # Histogram should be (HISTOGRAM_BINS,)
    hist = processed_data["histogram"]
    assert hist.shape == (
        Config.HISTOGRAM_BINS,
    ), f"Histogram shape mismatch: {hist.shape}"
    assert np.isclose(hist.sum(), 1.0, atol=1e-4) or np.isclose(
        hist.sum(), 0.0
    ), "Histogram should sum to 1 (density) or 0 (if empty)."

    print("Image Processing: Verified.\n")

    # ------------------------------------------------------------------------
    # 3. Verify Feature Extraction
    # ------------------------------------------------------------------------
    print("--- Step 3: Verifying Feature Extraction ---")

    extractor = EfficientNetExtractor()

    # Extract features from the processed data dictionary
    features = extractor.extract_features(processed_data)

    # Expected shape: 4 images * 1280 features (EfficientNetB0) = 5120
    expected_dim = 4 * 1280
    assert features.shape == (
        expected_dim,
    ), f"Feature shape mismatch. Got {features.shape}, expected ({expected_dim},)"

    # Check for NaNs
    assert not np.isnan(features).any(), "Extracted features contain NaNs."

    print("Feature Extraction: Verified.\n")

    # ------------------------------------------------------------------------
    # 4. Verify Data Manager
    # ------------------------------------------------------------------------
    print("--- Step 4: Verifying Data Manager ---")

    dm = DataManager()

    # Prepare Training Set (Debug size)
    print("Generating Training Data...")
    train_dataset = dm.prepare_dataset("train", load_cached_data=False)

    # Verify dataset structure
    required_keys = ["X_static", "weeks", "y", "patient_ids", "base_weeks"]
    for k in required_keys:
        assert k in train_dataset, f"Dataset missing key: {k}"

    X_train = train_dataset["X_static"]
    y_train = train_dataset["y"]

    # Check dimensions
    # X_static = 5120 (Visual) + 4 (Hist) + ~7 (Tabular OneHot/Scaled)
    # Exact tabular dim depends on OneHotEncoder categories, usually Sex(2)+Smoking(3)+Age(1)+Percent(1) = 7
    # So approx 5131.
    assert X_train.ndim == 2, "X_static should be 2D."
    assert len(X_train) == len(y_train), "Mismatch between X and y length."
    assert (
        len(X_train) <= Config.DEBUG_SIZE * 100
    ), "Dataset size seems too large for debug mode (checking upper bound loosely due to multiple visits)."

    print(f"Train X shape: {X_train.shape}")

    # Prepare Validation Set
    print("Generating Validation Data...")
    val_dataset = dm.prepare_dataset("val", load_cached_data=False)
    assert len(val_dataset["y"]) > 0, "Validation set is empty."

    # Prepare Test Set
    print("Generating Test Data...")
    test_dataset = dm.prepare_dataset("test", load_cached_data=False)
    assert len(test_dataset["patient_ids"]) > 0, "Test set is empty."

    print("Data Manager: Verified.\n")

    # ------------------------------------------------------------------------
    # 5. Verify Model Pipeline
    # ------------------------------------------------------------------------
    print("--- Step 5: Verifying Model Pipeline ---")

    model = DecoupledQuantileModel()

    # Fit Model
    print("Fitting model...")
    model.fit(
        train_dataset["X_static"],
        train_dataset["weeks"],
        train_dataset["y"],
        train_dataset["base_weeks"],
    )

    # Predict on Validation
    print("Predicting on validation...")
    val_preds, val_sigma = model.predict(
        val_dataset["X_static"], val_dataset["weeks"], val_dataset["base_weeks"]
    )

    # Assertions on predictions
    assert val_preds.shape == val_dataset["y"].shape, "Prediction shape mismatch."
    assert val_sigma.shape == val_dataset["y"].shape, "Confidence shape mismatch."
    assert (
        val_sigma >= Config.MIN_CONFIDENCE
    ).all(), f"Confidence values below minimum threshold {Config.MIN_CONFIDENCE}."

    # Calculate Score
    metric = score_function(val_dataset["y"], val_preds, val_sigma)
    print(f"Validation Metric (Debug): {metric:.4f}")

    print("Model Pipeline: Verified.\n")

    # ------------------------------------------------------------------------
    # 6. End-to-End Execution
    # ------------------------------------------------------------------------
    print("--- Step 6: Running Full Pipeline Wrapper ---")

    # This function orchestrates the whole flow and creates the submission file
    # We expect it to run without error using the cached data we just generated
    try:
        run_training_and_inference()
        print("run_training_and_inference executed successfully.")
    except Exception as e:
        print(f"run_training_and_inference failed: {e}")
        raise e

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_FILE):
        sub_df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission file created with {len(sub_df)} rows.")
        assert "Patient_Week" in sub_df.columns
        assert "FVC" in sub_df.columns
        assert "Confidence" in sub_df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
