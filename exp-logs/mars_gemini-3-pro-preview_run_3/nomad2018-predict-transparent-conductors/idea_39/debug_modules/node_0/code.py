import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library import data_manager, model_engine


def run_pipeline_demo():
    print("Starting pipeline demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We enable debug mode to process only a small subset of data (50 samples).
    # We also reduce the number of boosting rounds for XGBoost to ensure fast training.
    print("Configuring for fast execution...")
    Config.DEBUG_MODE = True
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["learning_rate"] = 0.1

    # Ensure output directories exist
    Config.setup()

    # -------------------------------------------------------------------------
    # 2. Data Loading and Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Data Manager] Loading and processing datasets...")
    # We set load_cached_data=False to force the feature extraction pipeline
    # to run on the raw .xyz files, demonstrating the logic in feature_extraction.py.
    (X_train, y_train_log), (X_val, y_val_log), (X_test, test_ids) = (
        data_manager.get_datasets(load_cached_data=False)
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # Validation Checks
    assert X_train.shape[0] > 0, "Training set is empty."
    assert X_val.shape[0] > 0, "Validation set is empty."
    assert X_test.shape[0] > 0, "Test set is empty."
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature count mismatch between train and test."
    # Check that targets are log-transformed (should not be negative if original values >= 0)
    # Formation energy can be negative, but bandgap is usually positive.
    # The transformation is log1p, so if original > -1, result is real.
    assert not y_train_log.isnull().values.any(), "NaNs found in training targets."

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\n[Model Engine] Training XGBoost models...")
    # This trains one model per target column defined in y_train_log
    models = model_engine.train_xgboost(
        X_train, y_train_log, X_val, y_val_log, verbose=True
    )

    expected_targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    for t in expected_targets:
        assert t in models, f"Model for target '{t}' was not trained."

    # -------------------------------------------------------------------------
    # 4. Evaluation
    # -------------------------------------------------------------------------
    print("\n[Model Engine] Evaluating on validation set...")
    # Predict on validation set (predictions are in log scale)
    val_preds_log = model_engine.predict_xgboost(models, X_val)

    # Calculate metrics (RMSE on log data = RMSLE on original data)
    metrics = model_engine.evaluate_model(y_val_log, val_preds_log)

    print("Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Model Engine] Generating test predictions...")
    # Predict on test set (log scale)
    test_preds_log = model_engine.predict_xgboost(models, X_test)

    # Inverse transform to get original scale
    test_preds = data_manager.inverse_transform_targets(test_preds_log)

    # Construct submission DataFrame
    submission = pd.DataFrame()
    submission["id"] = test_ids
    # Assign predicted columns
    for col in expected_targets:
        submission[col] = test_preds[col]

    # Verify submission format
    assert submission.shape[0] == len(test_ids), "Submission row count mismatch."
    assert (
        list(submission.columns) == ["id"] + expected_targets
    ), "Submission columns mismatch."

    # Save submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    if os.path.exists(Config.SUBMISSION_PATH):
        print("Submission file successfully created.")
        print("Head of submission file:")
        print(submission.head())
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    run_pipeline_demo()
