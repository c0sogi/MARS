import os
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import seed_everything, save_submission, calculate_mae
from library.dataset import generate_dataset
from library.trainer import EnsembleTrainer

if __name__ == "__main__":
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing Demonstration...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Setup directories
    Config.setup()

    # Override Config for Fast Execution (Debug Mode)
    print("Configuring parameters for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 files for speed
    Config.N_FOLDS = 2  # Reduce folds to 2

    # Optimize LightGBM for tiny dataset and speed
    Config.LGBM_PARAMS.update(
        {
            "n_estimators": 10,  # Very few trees
            "num_leaves": 8,  # Simple trees
            "min_child_samples": 2,  # Allow splits on small data
            "verbose": -1,  # Silent mode
        }
    )
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = False

    # Clean up previous models in the working directory to ensure fresh run verification
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    # ==========================================
    # 2. Feature Generation (Training)
    # ==========================================
    print("\n--- Step 1: Generating Training Features ---")
    # Force computation (load_cached_data=False) to verify feature extraction logic
    X_train, y_train = generate_dataset("train", load_cached_data=False, debug=True)

    # Validation
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Training Target Shape: {y_train.shape}")

    assert not X_train.empty, "Training feature DataFrame is empty."
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length."
    assert "segment_id" in X_train.columns, "'segment_id' column missing from X_train."
    # Check that we respected the debug sample size (approximate, as file availability matters)
    assert len(X_train) <= Config.DEBUG_SAMPLE_SIZE, "Debug sample size limit exceeded."

    # Check for NaNs (Features should be imputed by the library)
    # Exclude segment_id from numeric check
    feat_cols = [c for c in X_train.columns if c != "segment_id"]
    assert (
        not X_train[feat_cols].isnull().values.any()
    ), "NaN values found in training features."

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n--- Step 2: Training Ensemble Model ---")
    trainer = EnsembleTrainer()

    # Train
    oof_preds = trainer.train_ensemble(X_train, y_train)

    # Validation
    assert len(oof_preds) == len(y_train), "OOF predictions length mismatch."

    # Check if models were saved
    saved_models = os.listdir(trainer.model_dir)
    print(f"Saved model files: {saved_models}")
    assert (
        len(saved_models) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} saved models, found {len(saved_models)}."

    # Calculate Metric
    mae = calculate_mae(y_train, oof_preds)
    print(f"Training Completed. OOF MAE: {mae:.4f}")

    # ==========================================
    # 4. Feature Generation (Test)
    # ==========================================
    print("\n--- Step 3: Generating Test Features ---")
    # Generate test set
    X_test, y_test = generate_dataset("test", load_cached_data=False, debug=True)

    # Validation
    print(f"Test Features Shape: {X_test.shape}")
    assert not X_test.empty, "Test feature DataFrame is empty."
    assert y_test is None, "y_test should be None for test split."
    assert "segment_id" in X_test.columns, "'segment_id' column missing from X_test."

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\n--- Step 4: Running Inference ---")
    test_preds = trainer.predict_ensemble(X_test)

    # Validation
    assert len(test_preds) == len(X_test), "Test predictions length mismatch."
    assert np.all(
        np.isfinite(test_preds)
    ), "Test predictions contain non-finite values."
    print("Inference successful.")

    # ==========================================
    # 6. Submission
    # ==========================================
    print("\n--- Step 5: Saving Submission ---")
    save_submission(test_preds, X_test)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(submission_df.head())

    # check format
    assert list(submission_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns are incorrect."
    assert len(submission_df) == len(X_test), "Submission row count mismatch."
    assert (
        submission_df["segment_id"].dtype == np.int64
        or submission_df["segment_id"].dtype == np.float64
    ), "segment_id type mismatch"

    print("\n=== Demonstration Completed Successfully ===")
