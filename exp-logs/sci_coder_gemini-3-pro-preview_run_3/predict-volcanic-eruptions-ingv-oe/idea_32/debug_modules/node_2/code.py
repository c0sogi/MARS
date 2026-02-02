import os
import sys
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb

# Import the provided library modules
import library.config as config
import library.feature_engineering as fe
import library.model_handler as mh
import library.training_manager as tm


def main():
    print("Initializing Seismic Eruption Prediction Demo...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We override global config variables to ensure the demo runs quickly (within seconds/minutes)
    # and uses a separate directory for artifacts.

    # Set seeds for reproducibility
    np.random.seed(42)

    # Reduce dataset size for demonstration
    config.DEBUG_SAMPLE_SIZE = 20  # Only process 20 files for train/val
    config.N_FOLDS = 2  # Use 2 folds for CV
    config.N_JOBS = 2  # Reduce parallelism overhead for small data

    # Configure paths for demo isolation
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Optimize LightGBM for speed (tiny model)
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8
    config.LGBM_PARAMS["learning_rate"] = 0.1
    # Ensure silent execution
    config.LGBM_PARAMS["verbosity"] = -1

    print(f"Configuration updated. Working directory: {config.WORKING_DIR}")

    # ==========================================
    # 2. Feature Engineering Demonstration
    # ==========================================
    print("\n[Step 1] Generating Features...")

    # Generate features for the training set (subset)
    # load_cached_data=False ensures we actually run the processing logic
    train_features = fe.get_train_data(
        load_cached_data=False, debug_size=config.DEBUG_SAMPLE_SIZE
    )

    # Validation
    print(f"Generated Train Features Shape: {train_features.shape}")
    assert not train_features.empty, "Feature DataFrame should not be empty."
    assert "segment_id" in train_features.columns, "Missing segment_id column."
    assert "time_to_eruption" in train_features.columns, "Missing target column."
    # Check for a specific feature to ensure engineering happened (e.g., sensor_1_trend_mean)
    # Note: exact column names depend on feature_engineering.py logic, checking prefix
    assert any(
        c.startswith("sensor_1_") for c in train_features.columns
    ), "Sensor 1 features not generated."

    # Verify cache file creation
    expected_cache = os.path.join(config.WORKING_DIR, "train_features.parquet")
    assert os.path.exists(expected_cache), f"Cache file {expected_cache} not found."

    # ==========================================
    # 3. Model Training Demonstration (Manager)
    # ==========================================
    print("\n[Step 2] Running Cross-Validation...")

    # Run the full CV pipeline
    # This will generate validation features (since they aren't cached yet) and train models
    mae = tm.run_cross_validation(
        load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE
    )

    print(f"Cross-Validation MAE: {mae:.4f}")

    # Validation
    assert isinstance(mae, float), "MAE should be a float."
    assert mae >= 0, "MAE cannot be negative."

    # Verify model files
    for fold in range(config.N_FOLDS):
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_fold_{fold}.txt")
        assert os.path.exists(model_path), f"Model file for fold {fold} missing."

    # ==========================================
    # 4. Direct Model Handler Usage (Component Check)
    # ==========================================
    print("\n[Step 3] Testing Model Handler Component...")

    # Manually prepare data to test train_fold_model directly
    feature_cols = [
        c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X = train_features[feature_cols]
    y = train_features["time_to_eruption"]

    # Simple split
    split_point = len(X) // 2
    X_tr, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_tr, y_val = y.iloc[:split_point], y.iloc[split_point:]

    # Train single model
    booster = mh.train_fold_model(X_tr, y_tr, X_val, y_val, fold_idx=99)

    # Predict
    preds = mh.predict_batch(booster, X_val)

    # Validation
    assert isinstance(
        booster, lgb.Booster
    ), "Returned object is not a LightGBM Booster."
    assert len(preds) == len(X_val), "Prediction count mismatch."
    assert np.isfinite(preds).all(), "Predictions contain NaNs or Infs."
    print("Model Handler direct training and prediction successful.")

    # ==========================================
    # 5. Inference and Submission
    # ==========================================
    print("\n[Step 4] Generating Submission...")

    # Generate predictions for a small subset of test files
    test_debug_size = 5
    tm.generate_test_predictions(load_cached_data=False, debug_size=test_debug_size)

    # Validation
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file not created."

    submission_df = pd.read_csv(config.SUBMISSION_FILE)
    print("Submission Head:")
    print(submission_df.head())

    assert list(submission_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns."
    assert (
        len(submission_df) == test_debug_size
    ), f"Expected {test_debug_size} rows in submission, got {len(submission_df)}."
    assert submission_df["segment_id"].dtype == "int64", "segment_id should be integer."

    print("\nDemo completed successfully. All logic verified.")


if __name__ == "__main__":
    main()
