import os
import sys
import shutil
import pandas as pd
import numpy as np

# 1. Configuration Override
# We must modify the config module variables *before* importing other modules
# that might use them at module level.
import library.config as config

# Override configuration for speed and demonstration purposes
config.SAMPLE_SIZE = 50  # Use only 50 samples for quick execution
config.XGB_PARAMS["n_estimators"] = 10  # Reduce trees for speed
config.WORKING_DIR = "./working/demo_execution"  # Separate working dir
config.SUBMISSION_PATH = (
    "./working/demo_submission.csv"  # Output submission to working dir
)

# Ensure the working directory exists
os.makedirs(config.WORKING_DIR, exist_ok=True)

# 2. Import Library Modules
from library.data import load_dataset
from library.model import DualTargetRegressor, save_submission


def main():
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Data Loading and Feature Generation
    # -------------------------------------------------------------------------
    print("\n[1] Loading Datasets and Generating Features...")

    # Load Train
    # load_cached_data=False ensures we actually run the feature generation logic
    # for this demonstration, proving it works.
    print("Loading Training Data...")
    X_train_full, y_train = load_dataset("train", load_cached_data=False)

    # Load Validation
    print("Loading Validation Data...")
    X_val_full, y_val = load_dataset("val", load_cached_data=False)

    # Load Test
    print("Loading Test Data...")
    X_test_full, _ = load_dataset("test", load_cached_data=False)

    # Verification of shapes
    print(f"Train shapes: X={X_train_full.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val_full.shape}, y={y_val.shape}")
    print(f"Test shapes:  X={X_test_full.shape}")

    assert (
        len(X_train_full) == config.SAMPLE_SIZE
    ), f"Expected {config.SAMPLE_SIZE} training samples, got {len(X_train_full)}"
    assert (
        len(X_val_full) == config.SAMPLE_SIZE
    ), f"Expected {config.SAMPLE_SIZE} validation samples, got {len(X_val_full)}"
    # Test set might be smaller than SAMPLE_SIZE if the file has fewer rows, but here 240 > 50.
    assert (
        len(X_test_full) == config.SAMPLE_SIZE
    ), f"Expected {config.SAMPLE_SIZE} test samples, got {len(X_test_full)}"

    # -------------------------------------------------------------------------
    # 2. Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2] Preprocessing...")

    # The 'id' column is included in the features but should not be used for training.
    # We extract it for the test set submission later.
    test_ids = X_test_full["id"].copy()

    # Drop 'id' from features
    drop_cols = ["id"]

    X_train = X_train_full.drop(columns=drop_cols, errors="ignore")
    X_val = X_val_full.drop(columns=drop_cols, errors="ignore")
    X_test = X_test_full.drop(columns=drop_cols, errors="ignore")

    print(f"Features used for training: {X_train.shape[1]}")

    # Verify no non-numeric columns remain
    assert (
        X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1]
    ), "Non-numeric columns found in X_train"

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\n[3] Training Model...")

    regressor = DualTargetRegressor()

    # Train the model
    metrics = regressor.train(X_train, y_train, X_val, y_val)

    print("\nTraining Metrics:")
    for target, mse in metrics.items():
        print(f"  {target}: MSE={mse:.6f}")
        assert mse >= 0, "MSE cannot be negative"

    # -------------------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------------------
    print("\n[4] Running Inference on Test Set...")

    # Predict (returns log-transformed values)
    preds_log = regressor.predict(X_test)

    print("Prediction sample (Log Scale):")
    print(preds_log.head())

    assert preds_log.shape == (len(X_test), 2), "Prediction shape mismatch"
    assert not preds_log.isnull().values.any(), "NaNs in predictions"

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    save_submission(test_ids, preds_log, output_path=config.SUBMISSION_PATH)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file created at {config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("Submission head:")
        print(df_sub.head())

        # Check columns
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

        # Check values are valid (inverse transform of log1p produces values > -1)
        assert (df_sub["bandgap_energy_ev"] > -1.0).all(), "Bandgap energy invalid"

    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
