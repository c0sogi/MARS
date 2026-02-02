import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import calculate_metric, format_submission
from library.image_loader import (
    load_patient_scan,
    get_variance_slices,
    preprocess_image,
    get_patient_images,
)
from library.feature_extractor import VisualEncoder, extract_patient_embedding
from library.data_manager import DataProcessor
from library.modeling import FVCPredictor, UncertaintyPredictor


def run_demonstration():
    print("=== Starting OSIC Pulmonary Fibrosis Progression Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment for Fast Execution...")
    # Override Config for speed and compatibility with small sample size
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 patients/samples
    Config.PCA_COMPONENTS = 5  # Reduce PCA components to fit small sample size

    # Set seeds for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"PCA Components: {Config.PCA_COMPONENTS}")

    # ------------------------------------------------------------------------
    # 2. Image Processing Demonstration
    # ------------------------------------------------------------------------
    print("\n[Step 2] Demonstrating Image Loading & Preprocessing...")

    # Load metadata to find a valid patient
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_patient = train_meta.iloc[0]
    pid = sample_patient["Patient"]
    dcm_path = sample_patient["dcm_path"]

    print(f"Selected Patient: {pid}")
    print(f"DICOM Path: {dcm_path}")

    # A. Load Raw Scans
    scans = load_patient_scan(pid, dcm_path)
    print(f"Loaded {len(scans)} raw DICOM slices.")

    if len(scans) > 0:
        # B. Select High Variance Slices
        selected_scans = get_variance_slices(scans, num_slices=Config.NUM_SLICES)
        assert len(selected_scans) == Config.NUM_SLICES, "Slice selection failed."

        # C. Preprocess Single Slice
        tensor_slice = preprocess_image(selected_scans[0])
        # Shape should be (3, 224, 224)
        assert tensor_slice.shape == (
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Preprocessing shape mismatch."

        # D. Get Full Patient Tensor (Pipeline Function)
        # Force load_cached_data=False to ensure code execution
        patient_tensor = get_patient_images(pid, dcm_path, load_cached_data=False)
        expected_shape = (Config.NUM_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE)
        assert (
            patient_tensor.shape == expected_shape
        ), f"Patient tensor shape mismatch. Got {patient_tensor.shape}"
        print("Image processing verified successfully.")
    else:
        print(
            "Warning: No scans found for this patient. Skipping specific image assertions."
        )

    # ------------------------------------------------------------------------
    # 3. Feature Extraction Demonstration
    # ------------------------------------------------------------------------
    print("\n[Step 3] Demonstrating Neural Feature Extraction...")

    if len(scans) > 0:
        encoder = VisualEncoder()

        # Extract features from the tensor
        features = encoder(patient_tensor)
        # EfficientNet-B0 outputs 1280 features per slice
        assert features.shape == (
            Config.NUM_SLICES,
            1280,
        ), "Feature extractor output shape mismatch."

        # Test the wrapper function which averages slices
        embedding = extract_patient_embedding(encoder, pid, dcm_path)
        assert embedding.shape == (1280,), "Patient embedding shape mismatch."
        print("Feature extraction verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Data Pipeline Demonstration
    # ------------------------------------------------------------------------
    print("\n[Step 4] Running Full Data Pipeline (DataProcessor)...")

    processor = DataProcessor()

    # Run processing (Metadata -> Images -> Embeddings -> PCA -> Tabular -> Interactions)
    # load_cached_data=False forces re-computation
    data_dict = processor.process_data(load_cached_data=False)

    # Verify contents
    expected_keys = [
        "X_static_train",
        "X_inter_train",
        "y_train",
        "X_static_val",
        "X_inter_val",
        "y_val",
        "X_static_test",
        "X_inter_test",
        "test_ids",
    ]

    for key in expected_keys:
        assert key in data_dict, f"Missing key '{key}' in processed data."
        assert isinstance(
            data_dict[key], np.ndarray
        ), f"Data '{key}' is not a numpy array."

    X_train = data_dict["X_inter_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_inter_val"]
    y_val = data_dict["y_val"]

    print(f"Training Features Shape: {X_train.shape}")
    print(f"Training Targets Shape: {y_train.shape}")
    print("Data pipeline verified successfully.")

    # ------------------------------------------------------------------------
    # 5. Modeling & Prediction Demonstration
    # ------------------------------------------------------------------------
    print("\n[Step 5] Training Models and Predicting...")

    # A. FVC Predictor (Median Regression)
    print("Training FVC Predictor (Quantile Regression)...")
    fvc_model = FVCPredictor()
    fvc_model.fit(X_train, y_train)

    # Predict on Validation
    fvc_pred_val = fvc_model.predict(X_val)
    assert fvc_pred_val.shape == y_val.shape, "FVC prediction shape mismatch."

    # B. Uncertainty Predictor (Elastic Net)
    print("Training Uncertainty Predictor (Elastic Net)...")
    # Calculate absolute residuals from training data
    train_preds = fvc_model.predict(X_train)
    abs_residuals = np.abs(y_train - train_preds)

    # Use static features for uncertainty prediction (as per design)
    X_static_train = data_dict["X_static_train"]
    X_static_val = data_dict["X_static_val"]

    unc_model = UncertaintyPredictor()
    unc_model.fit(X_static_train, abs_residuals)

    # Predict Uncertainty (Sigma)
    sigma_pred_val = unc_model.predict(X_static_val)
    assert sigma_pred_val.shape == y_val.shape, "Uncertainty prediction shape mismatch."

    print("Modeling verified successfully.")

    # ------------------------------------------------------------------------
    # 6. Evaluation & Submission Demonstration
    # ------------------------------------------------------------------------
    print("\n[Step 6] Evaluating and Formatting Submission...")

    # Calculate Metric
    metric_score = calculate_metric(y_val, fvc_pred_val, sigma_pred_val)
    print(f"Validation Laplace Log Likelihood: {metric_score:.5f}")
    assert isinstance(metric_score, float), "Metric calculation returned non-float."

    # Generate Submission for Test Set
    X_inter_test = data_dict["X_inter_test"]
    X_static_test = data_dict["X_static_test"]
    test_ids = data_dict["test_ids"]

    # Predict
    test_fvc = fvc_model.predict(X_inter_test)
    test_sigma = unc_model.predict(X_static_test)

    # Format
    test_df_dummy = pd.DataFrame({"Patient_Week": test_ids})
    format_submission(test_df_dummy, test_fvc, test_sigma, Config.SUBMISSION_PATH)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == len(test_ids), "Submission row count mismatch."
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Filter warnings to keep output clean
    warnings.filterwarnings("ignore")
    run_demonstration()
