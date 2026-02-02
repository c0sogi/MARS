import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# Import library components
from library.config import Config
from library.data_loader import get_train_val_data, get_test_data
from library.model import CouplingPredictor
from library.utils import save_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment for rapid demonstration...")

    # Override XGBoost parameters for speed
    # We reduce the number of estimators significantly to ensure the script completes quickly.
    Config.XGB_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["learning_rate"] = 0.1
    Config.XGB_PARAMS["max_depth"] = 6
    Config.XGB_PARAMS["early_stopping_rounds"] = 10

    # Ensure we utilize the available GPU
    Config.XGB_PARAMS["device"] = "cuda"
    Config.XGB_PARAMS["tree_method"] = "hist"

    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading (Training & Validation)
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading and processing training data subset...")

    # Load a subset (50,000 rows) to demonstrate the pipeline without long wait times.
    # We set load_cached_data=False to explicitly run the feature engineering logic.
    X_train, y_train, X_val, y_val = get_train_val_data(
        load_cached_data=False, debug_nrows=50000
    )

    print(f"   Training Features Shape: {X_train.shape}")
    print(f"   Training Target Shape:   {y_train.shape}")
    print(f"   Validation Features Shape: {X_val.shape}")

    # Verification: Ensure data is loaded correctly
    assert not X_train.empty, "Training data should not be empty."
    assert not X_val.empty, "Validation data should not be empty."
    assert len(X_train) == len(
        y_train
    ), "Mismatch between training features and target."

    # Verification: Check if critical features exist
    required_features = ["dist", "type_enc", "atom_0_enc", "atom_1_enc"]
    for feat in required_features:
        assert feat in X_train.columns, f"Missing required feature: {feat}"

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\n[3/6] Initializing and training XGBoost model...")

    predictor = CouplingPredictor()

    # Fit the model
    # This will also trigger the internal _evaluate_log_mae method on the validation set
    predictor.fit(X_train, y_train, X_val, y_val)

    # Verification: Check if model is actually trained by inspecting the underlying booster
    assert (
        predictor.model.get_booster() is not None
    ), "Model failed to initialize booster."

    # -------------------------------------------------------------------------
    # 4. Validation Logic Check
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying model predictions on validation set...")

    val_preds = predictor.predict(X_val)

    # Basic sanity checks on predictions
    assert len(val_preds) == len(y_val), "Prediction length mismatch."
    assert not np.isnan(val_preds).any(), "Predictions contain NaNs."
    assert np.isfinite(val_preds).all(), "Predictions contain infinite values."

    # Check if the model is better than a dummy predictor (mean baseline)
    # This is a loose check, but ensures the model learned *something*
    mae_model = np.mean(np.abs(y_val - val_preds))
    mae_baseline = np.mean(np.abs(y_val - y_train.mean()))

    print(f"   Model MAE: {mae_model:.4f}")
    print(f"   Baseline MAE: {mae_baseline:.4f}")

    if mae_model >= mae_baseline:
        print("   Warning: Model did not beat baseline in this small subset run.")
    else:
        print("   Success: Model outperformed mean baseline.")

    # -------------------------------------------------------------------------
    # 5. Test Data Processing & Prediction
    # -------------------------------------------------------------------------
    print("\n[5/6] Processing test data and generating predictions...")

    # Load subset of test data
    X_test, test_ids = get_test_data(load_cached_data=False, debug_nrows=10000)

    print(f"   Test Features Shape: {X_test.shape}")

    # Generate predictions
    test_preds = predictor.predict(X_test)

    # Verification
    assert len(test_preds) == len(test_ids), "Test prediction count mismatch."

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6/6] Saving submission file...")

    save_submission(test_ids, test_preds)

    # Verify file existence and format
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission saved to: {Config.SUBMISSION_PATH}")
        print(f"   Submission shape: {df_sub.shape}")

        expected_cols = ["id", "scalar_coupling_constant"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"
        assert (
            len(df_sub) == 10000
        ), "Submission file row count does not match input subset."
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
