import os
import sys
import numpy as np
import pandas as pd
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    SUBMISSION_PATH,
    INPUT_DIR,
    RADIOMICS_FEATURES,
)
from library.data_handler import FeatureEngineer
from library.model_architecture import train_model, generate_submission, DualModel
from library.metrics import laplace_log_likelihood
from library.image_processing import extract_global_stats


def main():
    print("=== Starting Demonstration Script ===")

    # ----------------------------------------------------------------
    # 1. Data Loading and Feature Engineering
    # ----------------------------------------------------------------
    print("\n[Step 1] Initializing FeatureEngineer and processing data...")
    engineer = FeatureEngineer()

    # We force load_cached_data=False to demonstrate the full processing pipeline
    # including radiomics extraction and tabular feature preprocessing.
    X_train, y_train, X_val, y_val, X_test, test_df = engineer.load_datasets(
        load_cached_data=False
    )

    # Validation: Check data shapes
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   X_test shape:  {X_test.shape}")

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Mismatch in training samples and targets."
    assert (
        X_val.shape[0] == y_val.shape[0]
    ), "Mismatch in validation samples and targets."
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Mismatch in feature dimensions between train and test."
    print("   Data shapes validated successfully.")

    # ----------------------------------------------------------------
    # 2. Verify Radiomics Extraction Logic (Unit Test)
    # ----------------------------------------------------------------
    print("\n[Step 2] Verifying radiomics extraction logic...")
    # Load raw train metadata to get a sample path
    raw_train = pd.read_csv(TRAIN_METADATA_PATH)
    sample_patient_path = os.path.join(INPUT_DIR, raw_train.iloc[0]["dcm_path"])

    # Extract stats for one patient
    stats = extract_global_stats(sample_patient_path)
    print(f"   Sample radiomics stats: {stats}")

    # Validation: Ensure required keys exist and values are numeric
    for feat in RADIOMICS_FEATURES:
        assert feat in stats, f"Missing radiomics feature: {feat}"
        assert isinstance(stats[feat], (int, float)), f"Feature {feat} is not numeric."

    # Since pydicom might not be installed, we expect Lung_Volume to be > 0
    # (either voxel count or slice count fallback)
    assert stats["Lung_Volume"] > 0, "Lung_Volume should be positive."
    print("   Radiomics extraction verified.")

    # ----------------------------------------------------------------
    # 3. Model Training
    # ----------------------------------------------------------------
    print("\n[Step 3] Training DualModel (FVC + Uncertainty)...")
    # train_model handles instantiation, fitting, and initial evaluation
    model = train_model(X_train, y_train, X_val, y_val)

    # Validation: Check if model components are fitted
    # ElasticNet stores coefficients in .coef_ after fitting
    assert hasattr(model.fvc_model, "coef_"), "FVC model is not fitted."
    assert hasattr(model.sigma_model, "coef_"), "Sigma model is not fitted."
    print("   Model training complete and verified.")

    # ----------------------------------------------------------------
    # 4. Evaluation
    # ----------------------------------------------------------------
    print("\n[Step 4] Evaluating model on validation set...")
    # Predict manually to verify metric calculation
    val_fvc_pred, val_sigma_pred = model.predict(X_val)

    # Compute metric
    metric_score = laplace_log_likelihood(y_val, val_fvc_pred, val_sigma_pred)
    print(f"   Calculated Metric Score: {metric_score:.4f}")

    # Validation: Metric should be negative (log likelihood) and sigma should be clipped >= 70
    assert metric_score < 0, "Metric score should be negative."
    assert np.all(val_sigma_pred >= 70), "Confidence values must be >= 70."
    print("   Evaluation logic verified.")

    # ----------------------------------------------------------------
    # 5. Submission Generation
    # ----------------------------------------------------------------
    print("\n[Step 5] Generating submission file...")
    submission_df = generate_submission(model, X_test, test_df)

    # Validation: Check file existence and format
    assert os.path.exists(
        SUBMISSION_PATH
    ), f"Submission file not found at {SUBMISSION_PATH}"

    loaded_sub = pd.read_csv(SUBMISSION_PATH)
    expected_cols = ["Patient_Week", "FVC", "Confidence"]

    assert (
        list(loaded_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert len(loaded_sub) == len(test_df), "Submission row count mismatch."

    # Check for nulls
    assert not loaded_sub.isnull().values.any(), "Submission contains null values."

    print(f"   Submission saved to {SUBMISSION_PATH}")
    print("   First 5 rows of submission:")
    print(loaded_sub.head().to_string())
    print("   Submission format verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
