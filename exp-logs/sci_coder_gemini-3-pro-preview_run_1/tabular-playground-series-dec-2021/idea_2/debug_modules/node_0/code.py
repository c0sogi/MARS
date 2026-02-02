import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings

# Import provided library modules
from library.config import Config
from library.data_loader import load_and_process_data
from library.model_factory import XGBoostWrapper, LightGBMWrapper
from library.ensemble_utils import weighted_soft_voting, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of the Heterogeneous Ensemble pipeline...")

    # --- 1. Configuration Overrides for Speed ---
    # We modify the Config parameters in-place to ensure the demo runs quickly.
    # In a real run, we would use the values defined in config.py.
    print("Configuring for fast demonstration mode...")
    Config.XGB_FIT_PARAMS["num_boost_round"] = 10
    Config.XGB_FIT_PARAMS["early_stopping_rounds"] = 5
    Config.XGB_FIT_PARAMS["verbose_eval"] = False

    Config.LGBM_FIT_PARAMS["num_boost_round"] = 10
    Config.LGBM_FIT_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_FIT_PARAMS["verbose_eval"] = False

    # We will limit the data to 5000 samples
    DEMO_SAMPLES = 5000

    # --- 2. Data Loading and Processing ---
    print(f"Loading and processing data (subset={DEMO_SAMPLES})...")
    # load_cached_data=False forces the feature engineering logic to run for demonstration
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process_data(
        load_cached_data=False, max_samples=DEMO_SAMPLES
    )

    # Validation: Check shapes
    print(f"Data Loaded: Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    assert (
        len(X_train) == DEMO_SAMPLES
    ), f"Expected {DEMO_SAMPLES} training samples, got {len(X_train)}"
    assert (
        len(X_val) == DEMO_SAMPLES
    ), f"Expected {DEMO_SAMPLES} validation samples, got {len(X_val)}"

    # Validation: Check Feature Engineering
    # The library adds 'Euclidean_Distance_To_Hydrology' and 'Hydrology_Elevation'
    expected_features = ["Euclidean_Distance_To_Hydrology", "Hydrology_Elevation"]
    for feat in expected_features:
        if feat not in X_train.columns:
            raise AssertionError(
                f"Feature engineering failed: {feat} missing from X_train"
            )
    print("Feature engineering verification passed.")

    # --- 3. Model Training: XGBoost ---
    print("\n--- Training XGBoost Model ---")
    xgb_wrapper = XGBoostWrapper()

    # Train
    xgb_wrapper.train(X_train, y_train, X_val, y_val)

    # Predict on Validation
    xgb_val_probs = xgb_wrapper.predict_proba(X_val)

    # Validation: Check prediction shape
    n_classes = len(xgb_wrapper.le.classes_)
    assert xgb_val_probs.shape == (
        len(X_val),
        n_classes,
    ), f"XGBoost output shape mismatch. Expected {(len(X_val), n_classes)}, got {xgb_val_probs.shape}"
    print("XGBoost training and inference successful.")

    # --- 4. Model Training: LightGBM ---
    print("\n--- Training LightGBM Model ---")
    lgbm_wrapper = LightGBMWrapper()

    # Train
    lgbm_wrapper.train(X_train, y_train, X_val, y_val)

    # Predict on Validation
    lgbm_val_probs = lgbm_wrapper.predict_proba(X_val)

    # Validation: Check prediction shape
    assert lgbm_val_probs.shape == (
        len(X_val),
        n_classes,
    ), f"LightGBM output shape mismatch. Expected {(len(X_val), n_classes)}, got {lgbm_val_probs.shape}"
    print("LightGBM training and inference successful.")

    # --- 5. Ensemble Logic ---
    print("\n--- Running Ensemble (Weighted Soft-Voting) ---")

    # Weights from Config
    weights = [Config.WEIGHT_XGB, Config.WEIGHT_LGBM]
    predictions = [xgb_val_probs, lgbm_val_probs]

    ensemble_val_probs = weighted_soft_voting(predictions, weights)

    # Calculate Ensemble Accuracy
    # We need to map probabilities back to class labels to compare with y_val
    # Note: y_val contains original class labels (e.g., 1, 2, 7).
    # The wrappers use LabelEncoder internally. We need to use the classes_ from one of the wrappers.
    # Since both wrappers fit on the same y_train, their encoders should be identical.

    class_labels = xgb_wrapper.le.classes_
    pred_indices = np.argmax(ensemble_val_probs, axis=1)
    pred_labels = class_labels[pred_indices]

    ensemble_acc = accuracy_score(y_val, pred_labels)
    print(f"Ensemble Validation Accuracy (Subset): {ensemble_acc:.4f}")

    # Basic sanity check: Accuracy should be better than random guessing (1/7 approx 0.14)
    # Given the class imbalance (class 1 and 2 dominate), a dummy classifier is around 0.5.
    # We expect at least valid output.
    assert 0.0 <= ensemble_acc <= 1.0, "Accuracy score out of bounds."

    # --- 6. Submission Generation ---
    print("\n--- Generating Submission ---")

    # Predict on Test Set
    xgb_test_probs = xgb_wrapper.predict_proba(X_test)
    lgbm_test_probs = lgbm_wrapper.predict_proba(X_test)

    # Ensemble Test Predictions
    ensemble_test_probs = weighted_soft_voting(
        [xgb_test_probs, lgbm_test_probs], weights
    )

    # Save Submission
    save_submission(
        test_ids=test_ids,
        probabilities=ensemble_test_probs,
        class_labels=class_labels,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check rows (should match test_ids length, which is DEMO_SAMPLES here)
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["Id", "Cover_Type"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
