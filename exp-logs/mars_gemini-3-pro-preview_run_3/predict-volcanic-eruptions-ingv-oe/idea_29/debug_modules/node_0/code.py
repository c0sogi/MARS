import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

# Import from the provided library files
from library.config import Config
from library.data_manager import DataManager
from library.model_engine import ModelEngine


def set_seeds(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # ==========================================
    # 0. Setup and Configuration Overrides
    # ==========================================
    print("Setting up demonstration configuration...")
    set_seeds(Config.SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for speed and silence
    # We want the demo to run in seconds/minutes, not hours
    Config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for demo
    Config.LGBM_PARAMS["learning_rate"] = 0.1  # Faster learning
    Config.LGBM_PARAMS["verbosity"] = -1
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = -1  # Disable logging

    # Define debug sizes
    TRAIN_DEBUG_SIZE = 50
    TEST_DEBUG_SIZE = 20

    # Ensure directories exist (Config.setup handles this, but good to be explicit)
    Config.setup()

    # ==========================================
    # 1. Feature Engineering & Data Loading
    # ==========================================
    print(f"\nStep 1: generating features for {TRAIN_DEBUG_SIZE} training samples...")

    # Force regeneration of features (load_cached_data=False) to test the pipeline
    X_train, y_train = DataManager.get_train_data(
        size=TRAIN_DEBUG_SIZE, load_cached_data=False
    )

    # Validation
    print("Validating training data shape...")
    assert (
        len(X_train) == TRAIN_DEBUG_SIZE
    ), f"Expected {TRAIN_DEBUG_SIZE} samples, got {len(X_train)}"
    assert (
        len(y_train) == TRAIN_DEBUG_SIZE
    ), f"Expected {TRAIN_DEBUG_SIZE} targets, got {len(y_train)}"
    assert X_train.shape[1] > 0, "Feature matrix is empty!"

    # Check for specific feature columns to ensure all streams ran
    # Stream A (Trend), Stream B (Texture), Stream C (PSD)
    cols = X_train.columns.tolist()
    has_trend = any("trend_mean" in c for c in cols)
    has_txt = any("txt_energy" in c for c in cols)
    has_psd = any("psd_low" in c for c in cols)

    assert has_trend, "Stream A features missing."
    assert has_txt, "Stream B features missing."
    assert has_psd, "Stream C features missing."
    print("Feature generation verified successfully.")

    # ==========================================
    # 2. Model Training (Cross-Validation)
    # ==========================================
    print(f"\nStep 2: Training K-Fold Ensemble on debug data...")

    engine = ModelEngine()

    # Train on the subset we just generated
    # load_cached_data=True will pick up the parquet file created in Step 1
    maes = engine.train_kfold_ensemble(size=TRAIN_DEBUG_SIZE, load_cached_data=True)

    # Validation
    print("Validating training results...")
    assert (
        len(maes) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} MAE scores, got {len(maes)}"

    # Check if model files were saved
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.CACHE_DIR, f"lgbm_model_fold_{fold}.txt")
        assert os.path.exists(model_path), f"Model file for fold {fold} missing."

    print(f"Training complete. Average MAE (Debug): {np.mean(maes):.4f}")

    # ==========================================
    # 3. Inference & Submission
    # ==========================================
    print(f"\nStep 3: Generating predictions for {TEST_DEBUG_SIZE} test samples...")

    # Generate test features and predict
    # This creates a submission file at Config.SUBMISSION_PATH
    engine.predict_ensemble(size=TEST_DEBUG_SIZE, load_cached_data=False)

    # Validation
    print("Validating submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    assert (
        len(submission_df) == TEST_DEBUG_SIZE
    ), f"Submission has {len(submission_df)} rows, expected {TEST_DEBUG_SIZE}"

    # Check columns
    expected_cols = ["segment_id", "time_to_eruption"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Check value types
    assert pd.api.types.is_numeric_dtype(
        submission_df["segment_id"]
    ), "segment_id should be numeric"
    assert pd.api.types.is_numeric_dtype(
        submission_df["time_to_eruption"]
    ), "time_to_eruption should be numeric"

    print("Submission file verified successfully.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
