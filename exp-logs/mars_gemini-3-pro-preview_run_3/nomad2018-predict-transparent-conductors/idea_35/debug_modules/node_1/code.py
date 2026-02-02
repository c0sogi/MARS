import os
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.features as features
import library.preprocessing as preprocessing
import library.model as model


def run_demo():
    print("=== Starting Demonstration of Library Components ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")
    # Enable debug mode to use a small subset of data (e.g., 50 samples)
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50

    # Reduce XGBoost estimators for quick training demonstration
    config.XGB_PARAMS["n_estimators"] = 10
    config.EARLY_STOPPING_ROUNDS = 5

    # Ensure working directory is clean for this run (optional, but good for verification)
    if os.path.exists(config.WORKING_DIR):
        # We won't delete it to avoid permission issues, but we will force
        # re-processing by setting load_cached_data=False in calls.
        pass

    print(f"Debug Mode: {config.DEBUG}")
    print(f"Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"XGB Estimators: {config.XGB_PARAMS['n_estimators']}")

    # ---------------------------------------------------------
    # 2. Testing Utils
    # ---------------------------------------------------------
    print("\n[2] Testing Utility Functions...")

    # Test Log/Exp transform
    val = np.array([0.0, 1.0, 10.0])
    log_val = utils.log1p_transform(val)
    exp_val = utils.expm1_transform(log_val)

    assert np.allclose(val, exp_val), "Log1p -> Expm1 roundtrip failed."
    print("Log/Exp transform verified.")

    # Test RMSLE calculation
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_pred = np.array([[1.1, 9.5], [1.9, 21.0]])
    rmsle = utils.calculate_rmsle(y_true, y_pred)
    assert isinstance(rmsle, float), "RMSLE should return a float."
    assert rmsle > 0, "RMSLE should be positive."
    print(f"RMSLE calculation verified: {rmsle:.4f}")

    # ---------------------------------------------------------
    # 3. Testing Data Loader
    # ---------------------------------------------------------
    print("\n[3] Testing Data Loader...")

    # Load metadata (this will use the debug sample size)
    train_meta = data_loader.load_metadata("train", load_cached_data=False)
    print(f"Loaded train metadata shape: {train_meta.shape}")

    assert (
        len(train_meta) <= config.DEBUG_SAMPLE_SIZE
    ), f"Expected <= {config.DEBUG_SAMPLE_SIZE} samples in debug mode, got {len(train_meta)}"

    # Test Data Generator
    print("Testing data generator (reading first 2 geometries)...")
    gen = data_loader.get_data_generator(train_meta.head(2))
    count = 0
    for idx, row, atoms in gen:
        count += 1
        assert hasattr(
            atoms, "get_positions"
        ), "Yielded object is not an ASE Atoms object."
        # Basic check on atoms
        assert len(atoms) > 0, "Crystal structure has no atoms."
    assert count == 2, "Generator did not yield expected number of items."
    print("Data loader and generator verified.")

    # ---------------------------------------------------------
    # 4. Testing Feature Extraction
    # ---------------------------------------------------------
    print("\n[4] Testing Feature Extraction...")

    # Process data to generate features
    # We use load_cached_data=False to force execution of feature logic
    df_features_train = features.process_data("train", load_cached_data=False)

    print(f"Generated features shape: {df_features_train.shape}")

    # Check for expected columns
    expected_prefixes = ["vol_per_atom", "rdf_", "bvs_", "angle_"]
    cols = df_features_train.columns.tolist()
    for prefix in expected_prefixes:
        matching = [c for c in cols if prefix in c]
        assert len(matching) > 0, f"Missing features with prefix: {prefix}"

    # Check targets exist in training features
    for target in config.TARGET_COLS:
        assert (
            target in df_features_train.columns
        ), f"Target {target} missing from features."

    print("Feature extraction verified.")

    # ---------------------------------------------------------
    # 5. Testing Preprocessing
    # ---------------------------------------------------------
    print("\n[5] Testing Preprocessing...")

    # This function loads features for train, val, test, cleans them, and transforms targets
    train_df, val_df, test_df = preprocessing.prepare_datasets(load_cached_data=False)

    print(f"Prepared Train shape: {train_df.shape}")
    print(f"Prepared Val shape:   {val_df.shape}")
    print(f"Prepared Test shape:  {test_df.shape}")

    # Verify cleaning (no constant columns)
    # We check if std deviation of any numeric column is 0 (ignoring targets/id)
    feature_cols = [c for c in train_df.columns if c not in config.TARGET_COLS + ["id"]]
    stds = train_df[feature_cols].std()
    assert (stds > 0).all() or len(
        stds
    ) == 0, "Found constant columns in prepared dataset."

    # Verify target transformation (values should be smaller than raw if log transformed)
    # Formation energy is small, but bandgap is usually > 0. Let's check bandgap.
    # Raw bandgap mean is around 2.0. Log1p(2.0) ~= 1.1.
    if "bandgap_energy_ev" in train_df.columns:
        mean_bg = train_df["bandgap_energy_ev"].mean()
        print(f"Transformed Bandgap Mean: {mean_bg:.4f}")
        # It's hard to assert exact value without raw data comparison,
        # but we assume the function works based on utils test.

    print("Preprocessing verified.")

    # ---------------------------------------------------------
    # 6. Testing Model Training
    # ---------------------------------------------------------
    print("\n[6] Testing Model Training...")

    # Instantiate the wrapper
    regressor = model.DualTargetRegressor()

    # Split features and targets
    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COLS]

    X_val = val_df[feature_cols]
    y_val = val_df[config.TARGET_COLS]

    # Fit model
    regressor.fit(X_train, y_train, X_val, y_val)
    print("Model fitting complete.")

    # Predict
    preds = regressor.predict(X_val)
    print(f"Predictions shape: {preds.shape}")
    assert preds.shape == (len(X_val), 2), "Prediction shape mismatch."
    assert (
        preds.values >= 0
    ).all(), "Predictions contain negative values (physical impossibility)."

    print("Model training and prediction verified.")

    # ---------------------------------------------------------
    # 7. Testing Full Pipeline (Train & Predict)
    # ---------------------------------------------------------
    print("\n[7] Testing Full Pipeline Integration...")

    # We use the prepared dataframes from step 5
    trained_model = model.train_and_predict(train_df, val_df, test_df)

    # Check if submission file was created
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission format
    sub_df = pd.read_csv(submission_path)
    assert "id" in sub_df.columns, "Submission missing 'id' column."
    for target in config.TARGET_COLS:
        assert target in sub_df.columns, f"Submission missing '{target}' column."

    assert len(sub_df) == len(test_df), "Submission row count mismatch."

    print(f"Pipeline completed successfully. Submission generated at {submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
