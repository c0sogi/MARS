import os
import sys
import numpy as np
import pandas as pd
import warnings
import joblib

# Import from the provided library files
import library.config as config
from library.data_loader import TaxiDataLoader
from library.model_trainer import ModelTrainer


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    print("Initializing demonstration...")

    # Set seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Monkey-patch configuration for speed optimization in this demo
    # We reduce the number of estimators to ensure the script completes quickly.
    print("Overriding configuration for fast demonstration...")
    config.XGB_PARAMS["n_estimators"] = 50
    config.LGBM_PARAMS["n_estimators"] = 50

    # Ensure we are using the correct working directory from config
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. DATA LOADING & PROCESSING
    # ==========================================
    print("\n=== Data Loading & Feature Engineering ===")

    # Initialize DataLoader in DEBUG_MODE to use a smaller subset (100k rows)
    # We set load_cached_data=False to demonstrate the processing logic
    loader = TaxiDataLoader(debug_mode=True, load_cached_data=False)

    # Load Training Data
    print("Loading training data...")
    X_train, y_train = loader.get_train_data()

    # Validation: Check shapes and types
    assert not X_train.empty, "Training features should not be empty"
    assert len(X_train) == len(y_train), "Mismatch between training features and target"
    assert (
        "haversine_dist" in X_train.columns
    ), "Feature engineering failed: 'haversine_dist' missing"
    assert (
        "rotated_manhattan_dist" in X_train.columns
    ), "Feature engineering failed: 'rotated_manhattan_dist' missing"
    print(f"Training Data Loaded: {X_train.shape}")

    # Load Validation Data
    print("Loading validation data...")
    X_val, y_val = loader.get_val_data()

    # Validation: Check shapes
    assert not X_val.empty, "Validation features should not be empty"
    print(f"Validation Data Loaded: {X_val.shape}")

    # Load Test Data
    print("Loading test data...")
    X_test, keys = loader.get_test_data()

    # Validation: Check shapes and keys
    assert not X_test.empty, "Test features should not be empty"
    assert len(X_test) == len(keys), "Mismatch between test features and keys"
    # The test set size in metadata is 9914
    assert len(keys) == 9914, f"Expected 9914 test rows, got {len(keys)}"
    print(f"Test Data Loaded: {X_test.shape}")

    # ==========================================
    # 3. MODEL TRAINING
    # ==========================================
    print("\n=== Model Training ===")

    trainer = ModelTrainer()

    # Clear any existing cached models to ensure fresh training for demo
    if os.path.exists(trainer.xgb_model_path):
        os.remove(trainer.xgb_model_path)
    if os.path.exists(trainer.lgbm_model_path):
        os.remove(trainer.lgbm_model_path)

    # Train XGBoost
    print("Training XGBoost...")
    trainer.train_xgboost(X_train, y_train, X_val, y_val, load_cached_model=False)

    # Verify XGBoost model was saved and loaded
    assert trainer.xgb_model is not None, "XGBoost model object is None after training"
    assert os.path.exists(trainer.xgb_model_path), "XGBoost model file was not created"

    # Train LightGBM
    print("Training LightGBM...")
    trainer.train_lgbm(X_train, y_train, X_val, y_val, load_cached_model=False)

    # Verify LightGBM model was saved and loaded
    assert (
        trainer.lgbm_model is not None
    ), "LightGBM model object is None after training"
    assert os.path.exists(
        trainer.lgbm_model_path
    ), "LightGBM model file was not created"

    # ==========================================
    # 4. INFERENCE & SUBMISSION
    # ==========================================
    print("\n=== Inference & Submission ===")

    # Generate Predictions
    predictions = trainer.predict_ensemble(X_test)

    # Validation: Check predictions
    assert len(predictions) == len(keys), "Prediction count does not match key count"
    assert not np.isnan(predictions).any(), "Predictions contain NaN values"

    # Save Submission
    trainer.save_submission(keys, predictions)

    # Verify Submission File
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    # Load submission to verify format
    sub_df = pd.read_csv(config.SUBMISSION_FILE)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns are incorrect"
    assert len(sub_df) == 9914, "Submission row count is incorrect"

    print("\nDemonstration completed successfully.")
    print(f"Submission saved to: {config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
