import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import log_transform, inverse_log_transform, calculate_rmse
from library.data_loader import load_and_process_data
from library.model_trainer import XGBoostTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting NYC Taxi Fare Prediction Demo ===")

    # 1. Setup and Configuration Overrides
    # We modify Config attributes to run a fast, isolated demo.
    print("\n[1] Configuring environment...")
    set_seed(Config.RANDOM_SEED)

    # Use a specific subdirectory for this demo to avoid overwriting main work
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce complexity for speed
    Config.N_CLUSTERS = 20  # Fewer clusters for faster KMeans

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Clusters: {Config.N_CLUSTERS}")

    # 2. Data Loading and Processing
    # We use debug=True to load a small slice of data and process it on the fly.
    # load_cached_data=False ensures we test the processing logic, not just file I/O.
    print("\n[2] Loading and processing data (Debug Mode)...")
    SAMPLE_SIZE = 5000

    train_df, val_df, test_df = load_and_process_data(
        load_cached_data=False, debug=True, sample_size=SAMPLE_SIZE
    )

    # 3. Validation of Data Processing
    print("\n[3] Validating processed data...")

    # Check shapes
    assert len(train_df) <= SAMPLE_SIZE, "Train set size exceeds sample limit"
    assert len(test_df) <= SAMPLE_SIZE, "Test set size exceeds sample limit"

    # Check for engineered features
    expected_features = [
        "pickup_cluster",
        "dropoff_cluster",
        "haversine_dist",
        "manhattan_dist",
        "bearing",
        "hour",
        "weekday",
        "year",
    ]
    for feat in expected_features:
        assert feat in train_df.columns, f"Feature {feat} missing from training data"
        assert feat in test_df.columns, f"Feature {feat} missing from test data"

    print(f"Verified existence of features: {expected_features}")

    # Check Target Transformation
    # The raw data has some negative fares. log1p(negative) -> NaN.
    # We must clean this for the model to train correctly.
    initial_len = len(train_df)
    train_df = train_df.dropna(subset=["fare_amount"])
    val_df = val_df.dropna(subset=["fare_amount"])
    dropped_count = initial_len - len(train_df)

    if dropped_count > 0:
        print(
            f"Dropped {dropped_count} rows with invalid target values (NaN after log transform)."
        )

    # Verify target is roughly in log scale (max fare shouldn't be huge)
    # Raw max is ~90k, log(90k) ~ 11.4. If we see values > 20, something is wrong.
    max_log_fare = train_df["fare_amount"].max()
    assert (
        max_log_fare < 20
    ), f"Target variable appears not log-transformed. Max value: {max_log_fare}"
    print("Target variable validation passed (Log Scale).")

    # 4. Model Training
    print("\n[4] Training XGBoost Model...")

    # Initialize trainer with reduced estimators for speed
    trainer = XGBoostTrainer(n_estimators=10)

    # Train
    model = trainer.train(train_df, val_df, target_col="fare_amount", key_col="key")

    assert model is not None, "Model training failed to return a model object."
    print("Model training complete.")

    # 5. Prediction and Inverse Transformation
    print("\n[5] Generating Predictions...")

    # Predict on validation set first to check RMSE manually
    val_preds_log = trainer.predict(val_df)
    val_preds_dollar = inverse_log_transform(val_preds_log)
    val_actual_log = val_df["fare_amount"].values
    val_actual_dollar = inverse_log_transform(val_actual_log)

    # Calculate RMSE on dollar scale
    rmse_dollar = calculate_rmse(val_actual_dollar, val_preds_dollar)
    print(f"Manual Validation RMSE ($): {rmse_dollar:.4f}")

    # Predict on Test Set
    test_preds_log = trainer.predict(test_df)
    test_preds_dollar = inverse_log_transform(test_preds_log)

    assert len(test_preds_dollar) == len(test_df), "Prediction length mismatch"
    assert np.all(test_preds_dollar >= -1.0), "Predictions contain invalid values (<-1)"

    # 6. Model Persistence
    print("\n[6] Testing Model Persistence...")
    model_filename = "demo_xgb_model.json"
    trainer.save_model(model_filename)

    # Verify file exists
    saved_path = os.path.join(Config.WORKING_DIR, model_filename)
    assert os.path.exists(saved_path), "Model file was not saved correctly."

    # Load model back
    new_trainer = XGBoostTrainer()
    new_trainer.load_model(model_filename)
    # Quick check if loaded model predicts same as original
    loaded_preds = new_trainer.predict(test_df)
    # Allow small float tolerance
    np.testing.assert_allclose(
        test_preds_log,
        loaded_preds,
        rtol=1e-5,
        err_msg="Loaded model predictions differ from original.",
    )
    print("Model save/load cycle verified.")

    # 7. Submission Generation
    print("\n[7] Generating Submission File...")
    submission_df = pd.DataFrame(
        {"key": test_df["key"], "fare_amount": test_preds_dollar}
    )

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify content
    saved_sub = pd.read_csv(submission_path)
    assert list(saved_sub.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert len(saved_sub) == len(test_df), "Submission row count mismatch"

    print(f"Submission saved to {submission_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
