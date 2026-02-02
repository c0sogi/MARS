import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.image_processing import process_patient
from library.feature_extraction import FeaturePipeline
from library.data_preparation import DataPreparation
from library.model_factory import QuantileGLMSystem

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_image_processing(df_train):
    """
    Demonstrates the low-level image processing function.
    """
    print("\n=== Demo: Image Processing ===")

    # Pick the first patient from the metadata
    sample_patient = df_train.iloc[0]
    pid = sample_patient["Patient"]
    dcm_path = sample_patient["dcm_path"]

    print(f"Processing patient: {pid}")
    print(f"DICOM Path: {dcm_path}")

    # Process patient (loads DICOMs, computes histogram, selects slices)
    # We force load_cached_data=False to ensure the code actually runs
    data = process_patient(pid, dcm_path, load_cached_data=False)

    slices = data["slices"]
    histogram = data["histogram"]

    print(f"Slices Shape: {slices.shape}")
    print(f"Histogram: {histogram}")

    # Validation
    assert slices.shape == (
        3,
        224,
        224,
    ), f"Expected slices shape (3, 224, 224), got {slices.shape}"
    assert len(histogram) == 5, f"Expected histogram length 5, got {len(histogram)}"

    print("Image processing verification passed.")


def demo_pipeline_and_training():
    """
    Demonstrates the full Data Prep -> Training -> Inference pipeline.
    """
    print("\n=== Demo: Pipeline & Training ===")

    # Initialize DataPreparation
    data_prep = DataPreparation()

    # 1. Get Training and Validation Data
    # This triggers FeatureExtraction pipeline internally
    print("Generating Training/Validation Data...")
    X_fvc_train, X_unc_train, y_train, X_fvc_val, X_unc_val, y_val = (
        data_prep.get_train_val_data(load_cached_data=False)
    )

    print(f"Train FVC Matrix Shape: {X_fvc_train.shape}")
    print(f"Train Uncertainty Matrix Shape: {X_unc_train.shape}")
    print(f"Train Target Shape: {y_train.shape}")

    # Validation: Check consistency
    assert X_fvc_train.shape[0] == y_train.shape[0], "Mismatch in Train samples"
    assert X_fvc_val.shape[0] == y_val.shape[0], "Mismatch in Val samples"
    # Check feature dimensions (Tabular + PCA + Hist + Time interactions)
    # Exact number depends on PCA components (30) + Hist (5) + Tabular (encoded) + Interactions
    # Just ensuring it's not empty
    assert X_fvc_train.shape[1] > 0, "Feature matrix is empty"

    # 2. Model Training
    print("\nInitializing QuantileGLMSystem...")
    model = QuantileGLMSystem()

    print("Fitting model...")
    model.fit(X_fvc_train, X_unc_train, y_train)

    # 3. Evaluation
    print("Evaluating on Validation Set...")
    score = model.evaluate(X_fvc_val, X_unc_val, y_val)
    print(f"Evaluation Score: {score}")

    # Validation
    assert np.isfinite(score), "Evaluation score is not finite"

    return model, data_prep


def demo_inference(model, data_prep):
    """
    Demonstrates inference on the test set and submission file generation.
    """
    print("\n=== Demo: Inference & Submission ===")

    # 1. Get Test Data
    print("Generating Test Data...")
    X_fvc_test, X_unc_test, df_ids = data_prep.get_test_data(load_cached_data=False)

    print(f"Test FVC Matrix Shape: {X_fvc_test.shape}")

    # 2. Predict
    print("Predicting...")
    pred_fvc, pred_delta = model.predict(X_fvc_test, X_unc_test)

    # Convert Delta (MAE) to Confidence (Sigma)
    # Sigma = Delta * sqrt(2)
    pred_sigma = pred_delta * np.sqrt(2)

    # 3. Create Submission DataFrame
    df_sub = df_ids.copy()
    df_sub["FVC"] = pred_fvc
    df_sub["Confidence"] = pred_sigma

    print("Sample Predictions:")
    print(df_sub.head())

    # 4. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(loaded_sub.columns) == [
        "Patient_Week",
        "FVC",
        "Confidence",
    ], "Submission columns mismatch"
    assert len(loaded_sub) == len(df_ids), "Submission row count mismatch"

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(Config.SEED)

    # OPTIMIZATION: Limit data size for speed
    print(f"Setting Config.DEBUG_DATA_SIZE to 5 for quick demonstration.")
    Config.DEBUG_DATA_SIZE = 5

    # Ensure working directory is clean-ish or just rely on the script overwriting
    Config.setup()

    # Load metadata just for the image processing demo
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA)

    # 2. Run Demos
    try:
        # A. Image Processing
        demo_image_processing(df_train_meta)

        # B. Pipeline & Training
        trained_model, data_prep_instance = demo_pipeline_and_training()

        # C. Inference
        demo_inference(trained_model, data_prep_instance)

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
