import os
import numpy as np
import pandas as pd
import warnings
import sys

# Import library components
import library.config as config
from library.data_loader import load_data
from library.preprocessing import FeaturePreprocessor, get_preprocessed_data
from library.modeling import HybridLDAEnsemble, run_modeling
from library.utils import calculate_log_loss, save_submission


def main():
    # 0. Setup
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    # Set seed for reproducibility in this script
    np.random.seed(42)

    print("=== Starting Library Demonstration and Verification ===\n")

    # ---------------------------------------------------------
    # 1. Data Loading Verification
    # ---------------------------------------------------------
    print("1. Testing Data Loading (library.data_loader)...")

    # Force load from CSVs to verify parsing logic
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(load_cached_data=False)

    # Assertions to verify data structure
    print(f"   Loaded Train shape: {X_train.shape}")
    print(f"   Loaded Val shape:   {X_val.shape}")

    # Check dimensions (192 features expected)
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"
    assert X_val.shape[1] == 192, "Validation feature count mismatch"
    assert X_test.shape[1] == 192, "Test feature count mismatch"

    # Check alignment
    assert len(X_train) == len(y_train), "Train features and labels length mismatch"
    assert len(X_val) == len(y_val), "Val features and labels length mismatch"

    # Check data types
    assert isinstance(
        y_train[0], str
    ), "Target labels should be strings (species names)"
    assert np.issubdtype(X_train.dtype, np.number), "Features should be numeric"

    print("   [PASS] Data loaded and verified.\n")

    # ---------------------------------------------------------
    # 2. Preprocessing Verification
    # ---------------------------------------------------------
    print("2. Testing Feature Preprocessing (library.preprocessing)...")

    # Instantiate the preprocessor class directly
    preprocessor = FeaturePreprocessor()

    # Test fit_transform on a subset
    subset_size = 100
    X_subset = X_train[:subset_size]
    X_transformed = preprocessor.fit_transform(X_subset)

    # Check shape preservation
    assert (
        X_transformed.shape == X_subset.shape
    ), "Preprocessing changed feature dimensions"

    # Check if scaling worked (StandardScaler should make mean ~0 and std ~1)
    # Note: Using a small subset might have slight variance, using loose tolerance
    feat_mean = np.mean(X_transformed, axis=0)
    feat_std = np.std(X_transformed, axis=0)

    # We check the first few features
    assert np.all(np.abs(feat_mean[:5]) < 0.5), "Transformed features are not centered"
    assert np.all(
        np.abs(feat_std[:5] - 1.0) < 0.5
    ), "Transformed features are not scaled"

    # Test the caching wrapper function
    # This should compute transformations and save .npy files to WORKING_DIR
    X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, test_ids_p = (
        get_preprocessed_data(load_cached_data=False)
    )

    assert X_train_p.shape == X_train.shape, "Preprocessed data shape mismatch"
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "X_train_transformed.npy")
    ), "Cache file not created"

    print("   [PASS] Preprocessing logic and caching verified.\n")

    # ---------------------------------------------------------
    # 3. Modeling Verification
    # ---------------------------------------------------------
    print("3. Testing Hybrid LDA Ensemble (library.modeling)...")

    model = HybridLDAEnsemble()

    # Fit the model
    print("   Fitting model (this may take a few seconds)...")
    model.fit(X_train_p, y_train_p)

    # Check if classes were discovered
    assert model.classes_ is not None, "Model did not populate classes_"
    assert len(model.classes_) > 1, "Model should detect multiple classes"

    # Predict on validation set
    val_probs = model.predict_proba(X_val_p)

    # Verify prediction shape and properties
    assert val_probs.shape == (
        len(X_val_p),
        len(model.classes_),
    ), "Prediction shape mismatch"

    # Verify probabilities sum to 1 (approx)
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    print("   [PASS] Model training and prediction verified.\n")

    # ---------------------------------------------------------
    # 4. Utility Verification
    # ---------------------------------------------------------
    print("4. Testing Utilities (library.utils)...")

    # Test Log Loss Calculation
    # We use the predictions from step 3
    loss = calculate_log_loss(y_val_p, val_probs, model.classes_)
    print(f"   Calculated Validation Log Loss: {loss:.5f}")
    assert isinstance(loss, float), "Log loss should be a float"
    assert loss >= 0, "Log loss cannot be negative"

    # Test Submission Saving
    dummy_sub_path = os.path.join(config.WORKING_DIR, "test_submission.csv")

    # Generate dummy predictions for test set
    test_probs = model.predict_proba(X_test_p)

    save_submission(test_ids_p, model.classes_, test_probs, dummy_sub_path)

    assert os.path.exists(dummy_sub_path), "Submission file was not created"

    # Verify file content format
    df_check = pd.read_csv(dummy_sub_path)
    assert "id" in df_check.columns, "Submission missing 'id' column"
    assert len(df_check) == len(test_ids_p), "Submission row count mismatch"
    assert (
        df_check.shape[1] == len(model.classes_) + 1
    ), "Submission column count mismatch"

    print("   [PASS] Utilities verified.\n")

    # ---------------------------------------------------------
    # 5. Full Pipeline Integration Test
    # ---------------------------------------------------------
    print("5. Running Full Modeling Pipeline (Integration Test)...")

    # This function orchestrates the entire process:
    # Load (cached) -> Train -> Val Score -> Predict Test -> Save Submission
    run_modeling(load_cached_data=True)

    # Check final output
    final_submission_path = config.SUBMISSION_FILE_PATH
    assert os.path.exists(final_submission_path), "Final submission file missing"

    print(f"   Final submission generated at: {final_submission_path}")
    print("   [PASS] Full pipeline executed successfully.\n")

    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
