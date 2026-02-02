import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_mae, save_submission
from library.data_loader import load_train_data, load_val_data, load_test_data
from library.models import StackingManager


def main():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print("[1/6] Setting up environment and overriding configuration for speed...")

    # Ensure reproducibility
    seed_everything(42)

    # Override Config for rapid demonstration
    # We enable debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 30  # Process only 30 files per dataset for speed

    # Reduce model complexity for the demo to ensure it finishes quickly
    Config.N_FOLDS = 2  # Minimum folds for CV
    Config.N_ESTIMATORS = 10  # Very few trees
    Config.EARLY_STOPPING_ROUNDS = 5

    # Update the parameter dictionaries in Config to reflect the reduced estimators
    Config.LGBM_PARAMS["n_estimators"] = Config.N_ESTIMATORS
    Config.XGB_PARAMS["n_estimators"] = Config.N_ESTIMATORS
    Config.CATBOOST_PARAMS["iterations"] = Config.N_ESTIMATORS

    # Ensure working directory exists for our demo output
    demo_output_dir = "./working/demo_run"
    os.makedirs(demo_output_dir, exist_ok=True)

    # Redirect Config working dir to avoid conflicts with existing caches
    Config.WORKING_DIR = demo_output_dir

    # ==========================================
    # 2. Data Loading & Feature Engineering
    # ==========================================
    print("[2/6] Loading and processing data (Feature Engineering)...")

    # Load data. We set load_cached_data=False to force the feature engineering
    # pipeline to run, verifying the signal processing logic.
    X_train, y_train = load_train_data(load_cached_data=False, debug=True)
    X_val, y_val = load_val_data(load_cached_data=False, debug=True)
    X_test, _ = load_test_data(load_cached_data=False, debug=True)

    print(f"    Train Features Shape: {X_train.shape}")
    print(f"    Val Features Shape:   {X_val.shape}")
    print(f"    Test Features Shape:  {X_test.shape}")

    # Assertions to verify data integrity
    assert not X_train.empty, "Training features DataFrame is empty."
    assert not X_val.empty, "Validation features DataFrame is empty."
    assert not X_test.empty, "Test features DataFrame is empty."
    assert y_train is not None, "Training target is None."
    assert y_val is not None, "Validation target is None."
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Mismatch between X_train and y_train rows."

    # Check for NaNs (Feature engineering should handle them, but let's verify)
    assert not X_train.isnull().values.any(), "NaN values found in training features."

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("[3/6] Initializing Stacking Manager...")

    manager = StackingManager()

    # Verify that the configuration overrides propagated to the model manager
    assert (
        manager.base_params["lgbm"]["n_estimators"] == 10
    ), "LGBM n_estimators not updated."
    assert (
        manager.base_params["xgb"]["n_estimators"] == 10
    ), "XGB n_estimators not updated."

    # ==========================================
    # 4. Training Pipeline
    # ==========================================
    print(
        "[4/6] Running Training Pipeline (Level 0 CV -> Level 1 -> Level 0 Retrain)..."
    )

    # This runs Stratified K-Fold CV, trains the Meta Learner, and retrains base models
    manager.fit_pipeline(X_train, y_train)

    # Verify models are stored in memory
    assert manager.level1_model is not None, "Level 1 Meta Learner was not trained."
    assert len(manager.level0_models_full) == 3, "Not all base models were retrained."

    # ==========================================
    # 5. Evaluation
    # ==========================================
    print("[5/6] Evaluating on Validation Set...")

    val_preds = manager.predict(X_val)

    # Compute Metric
    mae = compute_mae(y_val, val_preds)
    print(f"    Validation MAE: {mae:.4f}")

    # Basic sanity check on predictions
    assert len(val_preds) == len(y_val), "Prediction length mismatch."
    assert not np.isnan(val_preds).any(), "NaNs found in predictions."

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("[6/6] Generating Submission for Test Set...")

    test_preds = manager.predict(X_test)

    submission_path = os.path.join(demo_output_dir, "submission_demo.csv")
    save_submission(X_test.index, test_preds, output_path=submission_path)

    # Verify file creation
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify file content format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns."
    assert len(sub_df) == len(X_test), "Submission row count mismatch."

    print(f"Demo completed successfully. Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
