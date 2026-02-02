import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import library modules
from library.config import PATHS, MODEL_PARAMS, FEATURE_PARAMS
from library.data_loader import build_dataset
from library.model_handler import train_model, predict_model


def run_demo():
    # Set seeds for reproducibility
    np.random.seed(42)

    print("--- Starting Library Demonstration ---")

    # ---------------------------------------------------------
    # 1. Optimize Configuration for Speed
    # ---------------------------------------------------------
    print("\n[Step 1] Optimizing hyperparameters for fast demonstration...")
    # Override global model parameters to run a tiny, fast model
    MODEL_PARAMS.LGBM_PARAMS["n_estimators"] = 20
    MODEL_PARAMS.LGBM_PARAMS["num_leaves"] = 8
    MODEL_PARAMS.LGBM_PARAMS["learning_rate"] = 0.1
    MODEL_PARAMS.LGBM_PARAMS["early_stopping_rounds"] = 5
    MODEL_PARAMS.LGBM_PARAMS["min_child_samples"] = (
        1  # Allow splits on small debug data
    )

    # Ensure working directory exists (as defined in config)
    os.makedirs(PATHS.WORKING_DIR, exist_ok=True)
    print(f"Working directory: {PATHS.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading & Feature Extraction (Train/Val)
    # ---------------------------------------------------------
    print("\n[Step 2] Generating features for Training and Validation subsets...")

    # Load a small subset of training data (20 samples)
    # This triggers library.features.generate_features -> parallel processing
    X_train, y_train = build_dataset(
        split="train", debug_size=20, load_cached_data=False
    )

    # Load a small subset of validation data (10 samples)
    X_val, y_val = build_dataset(split="val", debug_size=10, load_cached_data=False)

    print(f"Train Features Shape: {X_train.shape}")
    print(f"Val Features Shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 3. Logic Verification (Assertions)
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying data integrity...")

    # Verify sizes
    assert len(X_train) == 20, f"Expected 20 training samples, got {len(X_train)}"
    assert len(X_val) == 10, f"Expected 10 validation samples, got {len(X_val)}"
    assert len(y_train) == 20, "Mismatch in training target length"

    # Verify feature columns exist
    # Check for a specific feature from each 'View' (Trend, Texture, Spectral)
    # Note: Sensor 1 is guaranteed to be in the list
    sample_cols = X_train.columns.tolist()

    # View A: Trend Quantile
    assert any(
        "trend_q50" in col for col in sample_cols
    ), "Missing Trend features (View A)"
    # View B: Texture Energy
    assert any(
        "txt_energy" in col for col in sample_cols
    ), "Missing Texture features (View B)"
    # View C: Spectral Band
    assert any(
        "spec_low" in col for col in sample_cols
    ), "Missing Spectral features (View C)"

    # Verify no NaNs in features (clean_signal should have handled inputs, extraction should be robust)
    assert not X_train.isnull().values.any(), "Feature matrix contains NaNs"

    print("Data integrity checks passed.")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 4] Training LightGBM model...")

    # Train model using the handler
    # Note: The handler uses the modified MODEL_PARAMS internally
    model = train_model(X_train, y_train, X_val, y_val)

    # Verify model object
    assert isinstance(
        model, lgb.LGBMRegressor
    ), "train_model did not return an LGBMRegressor"
    print("Model training complete.")

    # ---------------------------------------------------------
    # 5. Inference and Evaluation
    # ---------------------------------------------------------
    print("\n[Step 5] Evaluating model on validation set...")

    val_preds = predict_model(model, X_val)

    # Calculate MAE manually to verify
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Manual Validation MAE check: {mae:.4f}")

    assert len(val_preds) == len(y_val), "Prediction length mismatch"
    assert not np.isnan(val_preds).any(), "Predictions contain NaNs"

    # ---------------------------------------------------------
    # 6. Test Set Prediction & Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 6] Generating submission for Test subset...")

    # Load test data (features only, no target)
    # We use debug_size=10 for speed
    X_test, y_test = build_dataset(split="test", debug_size=10, load_cached_data=False)

    assert y_test is None, "Test set should not have target values"

    # Predict
    test_preds = predict_model(model, X_test)

    # To create a valid submission file, we need the segment_ids.
    # The build_dataset function drops them, so we need to retrieve them from the metadata
    # used by the debug process.
    # The debug metadata file is saved in working dir by data_loader.py
    debug_test_meta_path = os.path.join(PATHS.WORKING_DIR, "test_debug_10_metadata.csv")
    test_meta_df = pd.read_csv(debug_test_meta_path)

    # Construct submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": test_meta_df["segment_id"], "time_to_eruption": test_preds}
    )

    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())

    # Save submission
    # We save to a demo file to avoid overwriting the main submission file if needed,
    # though the config defines SUBMISSION_FILE.
    save_path = os.path.join(PATHS.SUBMISSION_DIR, "demo_submission.csv")
    submission_df.to_csv(save_path, index=False)

    assert os.path.exists(save_path), "Submission file was not created"
    print(f"Submission saved to {save_path}")

    print("\n--- Demonstration Finished Successfully ---")


if __name__ == "__main__":
    run_demo()
