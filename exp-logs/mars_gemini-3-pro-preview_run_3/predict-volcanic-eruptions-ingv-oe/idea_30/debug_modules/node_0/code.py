import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

# Import provided library modules
from library.config import Config
from library.data_loader import build_feature_dataset
from library.model_engine import run_cross_validation, predict_ensemble
from library.utils import save_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Demonstration...")

    # Set seeds for reproducibility
    np.random.seed(42)

    # Monkey-patch Config for speed optimization (Demo Mode)
    print("Overriding Config parameters for fast execution...")
    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.LGBM_PARAMS["n_estimators"] = 10  # Reduce trees from 10000 to 10
    Config.LGBM_PARAMS["num_leaves"] = 8  # Reduce complexity
    Config.LGBM_PARAMS["min_child_samples"] = 2  # Allow splits on small debug data
    Config.LGBM_PARAMS["verbosity"] = -1

    # We will use a small debug size to demonstrate feature extraction without waiting
    DEBUG_SIZE = 20

    # ==========================================
    # 2. Data Loading & Feature Extraction
    # ==========================================
    print(f"\nLoading data with debug_size={DEBUG_SIZE}...")

    # Load Train Data
    # We force load_cached_data=False to demonstrate the feature extraction logic runs correctly
    X_train_part, y_train_part = build_feature_dataset(
        mode="train", load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # Load Val Data
    X_val_part, y_val_part = build_feature_dataset(
        mode="val", load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # Load Test Data
    X_test, y_test = build_feature_dataset(
        mode="test", load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # --- Verification Steps ---
    print("Verifying data shapes...")

    # Check Train
    assert not X_train_part.empty, "Training features should not be empty."
    assert y_train_part is not None, "Training target should not be None."
    assert (
        len(X_train_part) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} training samples, got {len(X_train_part)}"
    assert (
        len(y_train_part) == DEBUG_SIZE
    ), "Mismatch between training features and target length."

    # Check Test
    assert not X_test.empty, "Test features should not be empty."
    assert y_test is None, "Test target should be None."
    assert "segment_id" in X_test.columns, "Test data must contain segment_id."

    print("Data loading verification passed.")

    # ==========================================
    # 3. Model Training (Cross-Validation)
    # ==========================================
    print("\nStarting Cross-Validation Training...")

    # Combine partial train and val for the CV engine to split
    # In a real run, we might load the full datasets.
    # Here we concat our debug subsets.
    X_full = pd.concat([X_train_part, X_val_part], ignore_index=True)
    y_full = pd.concat([y_train_part, y_val_part], ignore_index=True)

    # Drop segment_id for training as it's not a feature
    train_cols = [c for c in X_full.columns if c != "segment_id"]
    X_train_ready = X_full[train_cols]

    # Run CV
    models, cv_mae = run_cross_validation(X_train_ready, y_full, save_models=True)

    # --- Verification Steps ---
    print("Verifying training results...")
    assert (
        len(models) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} models, got {len(models)}"
    assert isinstance(cv_mae, float), "CV MAE should be a float."
    assert cv_mae >= 0, "MAE must be non-negative."

    # Check if model files were created
    for i in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"lgbm_model_fold_{i}.txt")
        assert os.path.exists(model_path), f"Model artifact {model_path} was not saved."

    print(f"Training verification passed. CV MAE: {cv_mae:.4f}")

    # ==========================================
    # 4. Inference
    # ==========================================
    print("\nRunning Inference on Test Set...")

    # Prepare Test Features (drop segment_id)
    X_test_ready = X_test[train_cols]

    # Predict
    predictions = predict_ensemble(models, X_test_ready)

    # --- Verification Steps ---
    print("Verifying predictions...")
    assert len(predictions) == len(
        X_test
    ), "Prediction count must match test sample count."
    assert np.all(
        np.isfinite(predictions)
    ), "Predictions contain NaNs or infinite values."

    print("Inference verification passed.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\nGenerating Submission File...")

    submission_filename = "demo_submission.csv"
    save_submission(
        segment_ids=X_test["segment_id"],
        predictions=predictions,
        filename=submission_filename,
    )

    # --- Verification Steps ---
    submission_path = os.path.join(Config.SUBMISSION_DIR, submission_filename)
    assert os.path.exists(submission_path), "Submission file was not created."

    # Load back to check format
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns mismatch."
    assert len(df_sub) == DEBUG_SIZE, "Submission row count mismatch."

    print(f"Submission verification passed. File saved to {submission_path}")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
