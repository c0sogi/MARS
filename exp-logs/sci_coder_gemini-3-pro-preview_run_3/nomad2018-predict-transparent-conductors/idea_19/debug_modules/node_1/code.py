import os
import sys
import numpy as np
import pandas as pd
import warnings
import random

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import library components
from library.config import get_xgb_params, TARGET_COLS, RANDOM_SEED, SUBMISSION_DIR
from library.utils import inverse_log_transform, compute_rmsle, save_submission
from library.data import FeatureLoader
from library.model import DualTargetRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting demonstration of the pipeline...")

    # 1. Initialize FeatureLoader with debug=True
    # This ensures we only process a small subset of data (100 samples) for speed.
    print("\n[1] Initializing FeatureLoader (debug=True)...")
    loader = FeatureLoader(debug=True)

    # 2. Load Training and Validation Data
    # We set load_cached_data=False to force the feature extraction logic to execute
    # on the debug subset, verifying the feature engineering code path.
    print("[2] Loading Training and Validation Data...")
    X_train, y_train_log, X_val, y_val_log = loader.load_train_val(
        load_cached_data=False
    )

    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train_log shape: {y_train_log.shape}")
    print(f"    X_val shape: {X_val.shape}")
    print(f"    y_val_log shape: {y_val_log.shape}")

    # Verify that data was loaded and shapes align
    assert not X_train.empty, "X_train should not be empty"
    assert not y_train_log.empty, "y_train_log should not be empty"
    assert X_train.shape[0] == y_train_log.shape[0], "Mismatch in training samples"
    assert X_val.shape[0] == y_val_log.shape[0], "Mismatch in validation samples"
    # Ensure targets match configuration
    assert (
        list(y_train_log.columns) == TARGET_COLS
    ), "Target columns do not match config"

    # 3. Load Test Data
    print("\n[3] Loading Test Data...")
    X_test, ids_test = loader.load_test(load_cached_data=False)

    print(f"    X_test shape: {X_test.shape}")
    print(f"    ids_test shape: {ids_test.shape}")

    # Verify test data
    assert not X_test.empty, "X_test should not be empty"
    assert len(ids_test) == len(X_test), "Mismatch in test samples"
    # Ensure feature consistency: Test set should have same columns as Train (handled by loader)
    assert list(X_test.columns) == list(
        X_train.columns
    ), "Feature mismatch between Train and Test"

    # 4. Initialize Model
    # Get debug parameters for faster training (fewer estimators, limited depth)
    print("\n[4] Initializing DualTargetRegressor...")
    xgb_params = get_xgb_params(debug=True)
    model = DualTargetRegressor(params=xgb_params)

    # 5. Train Model
    print("[5] Training Model...")
    # The model handles two internal XGBoost regressors for the two targets
    model.fit(
        X_train,
        y_train_log,
        X_val=X_val,
        y_val=y_val_log,
        early_stopping_rounds=10,
        verbose=False,
    )
    print("    Training complete.")

    # 6. Evaluate on Validation Set
    print("\n[6] Evaluating on Validation Set...")
    # Predict returns a DataFrame with columns matching TARGET_COLS (log scale)
    val_preds_log = model.predict(X_val)

    # Inverse transform to get original scale for metric calculation
    val_preds_original = inverse_log_transform(val_preds_log)
    y_val_original = inverse_log_transform(y_val_log)

    # Compute RMSLE using the utility function
    rmsle_score = compute_rmsle(y_val_original.values, val_preds_original.values)
    print(f"    Validation RMSLE: {rmsle_score:.4f}")

    # Sanity checks
    assert val_preds_original.shape == y_val_original.shape
    assert not np.isnan(rmsle_score), "RMSLE is NaN"
    assert rmsle_score >= 0, "RMSLE cannot be negative"

    # 7. Generate Test Predictions
    print("\n[7] Generating Test Predictions...")
    test_preds_log = model.predict(X_test)
    test_preds_original = inverse_log_transform(test_preds_log)

    print(f"    Test Predictions shape: {test_preds_original.shape}")
    print(f"    Sample predictions:\n{test_preds_original.head()}")

    # 8. Save Submission
    print("\n[8] Saving Submission...")
    submission_filename = "demo_submission.csv"
    save_submission(
        ids=ids_test,
        formation_energy=test_preds_original[TARGET_COLS[0]],
        bandgap_energy=test_preds_original[TARGET_COLS[1]],
        filename=submission_filename,
    )

    # Verify file creation
    submission_path = os.path.join(SUBMISSION_DIR, submission_filename)
    if os.path.exists(submission_path):
        print(f"    Verified: {submission_path} exists.")

        # Verify submission format
        df_sub = pd.read_csv(submission_path)
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        assert len(df_sub) == len(ids_test), "Submission row count mismatch"
        print("    Submission format verified.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    # Set global seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    run_demo()
