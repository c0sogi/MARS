import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb

# Import from the provided library files
import library.config as config
from library.data_loader import load_metadata
from library.features import process_data
from library.model import train_model, predict_model
from library.utils import calculate_rmsle


def main():
    print("Starting demonstration of the material property prediction pipeline...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    # We reduce the number of estimators to ensure this demo script finishes quickly.
    # In a real run, we would use the values in config.py.
    print("Adjusting hyperparameters for quick demonstration...")
    config.XGB_PARAMS["n_estimators"] = 10
    config.EARLY_STOPPING_ROUNDS = 5

    # We will use a small subset of data for this demo
    DEMO_TRAIN_SIZE = 50
    DEMO_VAL_SIZE = 20
    DEMO_TEST_SIZE = 20

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Loading Metadata ---")
    # Load subsets of metadata
    train_meta = load_metadata(split="train", limit=DEMO_TRAIN_SIZE)
    val_meta = load_metadata(split="val", limit=DEMO_VAL_SIZE)
    test_meta = load_metadata(split="test", limit=DEMO_TEST_SIZE)

    print(f"Loaded {len(train_meta)} training samples.")
    print(f"Loaded {len(val_meta)} validation samples.")
    print(f"Loaded {len(test_meta)} test samples.")

    # Verify loaded data
    assert not train_meta.empty, "Training metadata is empty."
    assert "file_path" in train_meta.columns, "Metadata missing 'file_path' column."
    assert (
        "formation_energy_ev_natom" in train_meta.columns
    ), "Metadata missing target column."

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n--- Processing Features ---")
    # process_data handles loading geometry files and computing descriptors
    # It also caches results to disk in WORK_DIR

    # Note: process_data caches based on ID hash. Since we are using subsets,
    # it might re-compute if the specific subset hash isn't cached, or load if it is.
    # We force computation or load existing logic handled by the function.

    train_df = process_data(train_meta)
    val_df = process_data(val_meta)
    test_df = process_data(test_meta)

    # Verify feature generation
    # We expect more columns than metadata (features added)
    assert train_df.shape[1] > train_meta.shape[1], "No new features were generated."

    # Check for specific generated features (e.g., Volume, RDF)
    expected_feature_prefixes = ["vol_ang3", "density_amu_ang3", "RDF_", "Al_CN_p50"]
    cols = train_df.columns.tolist()
    for prefix in expected_feature_prefixes:
        match = any(c.startswith(prefix) for c in cols)
        if not match:
            # It's possible some specific element features (like Al_CN) aren't generated
            # if the subset doesn't contain Al, but volume/density/RDF should exist.
            if prefix in ["vol_ang3", "density_amu_ang3"]:
                raise AssertionError(
                    f"Expected feature {prefix} not found in processed data."
                )
            else:
                print(
                    f"Warning: Feature prefix {prefix} not found in subset (might be expected)."
                )

    print(f"Training data shape after feature extraction: {train_df.shape}")

    # -------------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------------
    print("\n--- Training Models ---")
    # train_model handles log-transform of targets and training separate models
    models = train_model(train_df, val_df)

    # Verify models dictionary
    for target in config.TARGET_COLS:
        assert target in models, f"Model for {target} was not trained."
        assert isinstance(
            models[target], xgb.XGBRegressor
        ), f"Model for {target} is not an XGBRegressor."

    # -------------------------------------------------------------------------
    # 5. Prediction
    # -------------------------------------------------------------------------
    print("\n--- Generating Predictions ---")
    submission_df = predict_model(models, test_df)

    # Verify submission structure
    expected_cols = ["id"] + config.TARGET_COLS
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"
    assert len(submission_df) == len(
        test_df
    ), "Submission row count does not match test set size."

    # Check for non-negative predictions (physical constraint)
    for target in config.TARGET_COLS:
        if (submission_df[target] < 0).any():
            print(
                f"Warning: Negative predictions found for {target}. Clipping to 0 for validity check."
            )
            submission_df[target] = submission_df[target].clip(lower=0)

    print("\nSample Predictions:")
    print(submission_df.head())

    # -------------------------------------------------------------------------
    # 6. Saving Output
    # -------------------------------------------------------------------------
    output_path = os.path.join(config.WORK_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"\nDemo submission saved to: {output_path}")

    # -------------------------------------------------------------------------
    # 7. Metric Verification (Self-Check)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Metric Calculation ---")
    # Create dummy ground truth and predictions to test calculate_rmsle
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # Manual calculation:
    # log(1+1)=0.693, log(1+1.1)=0.742 -> diff=0.049 -> sq=0.0024
    # log(1+2)=1.099, log(1+1.9)=1.065 -> diff=0.034 -> sq=0.0011
    # log(1+3)=1.386, log(1+3.2)=1.435 -> diff=0.049 -> sq=0.0024
    # mean_sq = 0.00196 -> sqrt = 0.044

    score = calculate_rmsle(y_true, y_pred)
    print(f"Test RMSLE Score: {score:.5f}")
    assert score < 0.1, "RMSLE calculation seems incorrect."

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
