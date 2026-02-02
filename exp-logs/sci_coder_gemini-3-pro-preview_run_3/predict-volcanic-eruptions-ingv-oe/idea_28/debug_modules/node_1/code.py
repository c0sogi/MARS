import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Import from the provided library
from library.config import Config
from library.data_loader import build_feature_dataset
from library.model_trainer import ModelTrainer


def run_demo():
    print("Starting End-to-End Demo...")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    # We modify the Config class attributes directly to optimize for a quick demo run.
    # This ensures we don't process the entire 20GB dataset or train for hours.

    print("Configuring parameters for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 files per dataset
    Config.N_FOLDS = 2  # Use 2-fold CV instead of 5

    # Reduce Model complexity for speed
    Config.MODEL_PARAMS["n_estimators"] = 20
    Config.MODEL_PARAMS["num_leaves"] = 31
    Config.MODEL_PARAMS["learning_rate"] = 0.1

    # Silence training output
    Config.TRAIN_PARAMS["verbose_eval"] = False
    Config.TRAIN_PARAMS["early_stopping_rounds"] = 5

    # Set seeds for reproducibility
    np.random.seed(Config.SEED)

    # ==========================================
    # 2. Feature Extraction Pipeline
    # ==========================================
    print("\n[Step 1] Running Feature Extraction (Debug Mode)...")

    # We force `load_cached_data=False` to verify the signal processing logic
    # actually runs and computes features from the raw CSVs.
    train_df, val_df, test_df = build_feature_dataset(
        load_cached_data=False, debug=True
    )

    # --- Validation ---
    print("Validating extracted features...")

    # Check dimensions
    assert (
        len(train_df) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train DF size exceeds debug limit"
    assert len(test_df) <= Config.DEBUG_SAMPLE_SIZE, "Test DF size exceeds debug limit"

    # Check for specific engineered features (e.g., from Trend or Welch's method)
    expected_cols = ["sensor_1_mean", "sensor_1_trend_std", "sensor_1_spec_low"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing expected feature: {col}"

    # Check for target variable
    assert (
        "time_to_eruption" in train_df.columns
    ), "Target variable missing from Train DF"
    assert (
        "time_to_eruption" not in test_df.columns
    ), "Target variable improperly present in Test DF"

    print(f"Success. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ==========================================
    # 3. Model Training (Stratified CV)
    # ==========================================
    print("\n[Step 2] Training LightGBM Ensemble...")

    trainer = ModelTrainer()

    # Run Cross-Validation
    # This uses the features extracted in the previous step
    mae_score = trainer.run_stratified_cv(train_df)

    # --- Validation ---
    print(f"CV Finished. MAE: {mae_score:.4f}")
    assert mae_score > 0, "MAE should be a positive float"
    assert (
        len(trainer.models) == Config.N_FOLDS
    ), f"Trainer should have {Config.N_FOLDS} trained models"

    # Check if model files were saved to disk
    for i in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"lgbm_model_fold_{i}.txt")
        assert os.path.exists(
            model_path
        ), f"Model file for fold {i} not found at {model_path}"

    # ==========================================
    # 4. Inference and Submission
    # ==========================================
    print("\n[Step 3] Generating Submission...")

    trainer.generate_submission(test_df)

    # --- Validation ---
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Load submission to check format
    sub_df = pd.read_csv(submission_path)

    # Check Header
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns are incorrect"

    # Check Row Count
    assert len(sub_df) == len(
        test_df
    ), f"Submission row count {len(sub_df)} does not match test set {len(test_df)}"

    # Check Data Types
    assert pd.api.types.is_integer_dtype(
        sub_df["segment_id"]
    ), "segment_id should be integer"
    assert pd.api.types.is_numeric_dtype(
        sub_df["time_to_eruption"]
    ), "time_to_eruption should be numeric"

    print("\n[Success] Demo completed successfully. All logic verified.")
    print(f"Final Submission Head:\n{sub_df.head()}")


if __name__ == "__main__":
    # Suppress LightGBM warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    run_demo()
