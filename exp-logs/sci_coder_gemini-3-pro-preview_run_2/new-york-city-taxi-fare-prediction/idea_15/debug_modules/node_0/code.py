import os
import sys
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.geometry_utils import DistanceCalculator, GridIndexer
from library.data_loader import TaxiDataLoader
from library.feature_factory import FactorizedEncoder, process_data
from library.model_trainer import XGBTrainer, train_and_predict


def run_demo():
    print("=== Starting Demonstration and Verification Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Redirect working directory to avoid messing with existing runs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_OUTPUT_PATH = os.path.join(Config.WORKING_DIR, "xgb_model.json")
    Config.CACHE_PROCESSED_TRAIN = os.path.join(
        Config.WORKING_DIR, "processed_train.parquet"
    )
    Config.CACHE_PROCESSED_VAL = os.path.join(
        Config.WORKING_DIR, "processed_val.parquet"
    )
    Config.CACHE_PROCESSED_TEST = os.path.join(
        Config.WORKING_DIR, "processed_test.parquet"
    )

    # Reduce sample sizes for speed
    Config.LEARNER_SAMPLE_SIZE = 5000  # Small subset for training
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset for wisdom/val/test in debug mode

    # Reduce Model complexity
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 2
    Config.XGB_PARAMS["max_depth"] = 3

    # Ensure directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Unit Testing Geometry Utilities
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Geometry Utilities...")

    # Test Haversine: Distance between (0,0) and (0,1) is approx 111.19 km
    d = DistanceCalculator.haversine(
        np.array([0]), np.array([0]), np.array([1]), np.array([0])
    )
    expected_d = 111.19
    assert np.isclose(
        d[0], expected_d, atol=1.0
    ), f"Haversine calc failed. Got {d[0]}, expected ~{expected_d}"

    # Test Grid Indexer
    # L5 is 0.045. Coordinate 0.05 should fall into index 1 (floor(0.05/0.045) = 1)
    # Coordinate -0.05 should fall into index -2 (floor(-0.05/0.045) = -2)
    lat_arr = np.array([0.05, -0.05])
    lon_arr = np.array([0.00, 0.00])
    keys = GridIndexer.get_grid_key(lat_arr, lon_arr, "L5")

    assert (
        keys[0] == "1_0"
    ), f"Grid key generation failed for positive coord. Got {keys[0]}"
    assert (
        keys[1] == "-2_0"
    ), f"Grid key generation failed for negative coord. Got {keys[1]}"

    print("Geometry utilities verified.")

    # -------------------------------------------------------------------------
    # 3. Unit Testing Factorized Encoder Logic (Synthetic)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying FactorizedEncoder Logic (Synthetic Data)...")

    # Create synthetic Wisdom data
    # Route A: 2 trips, fares 10 and 20. Global Sum=30, Count=2, Mean=15.
    wisdom_data = pd.DataFrame(
        {
            "pickup_latitude": [40.75, 40.75],
            "pickup_longitude": [-73.98, -73.98],
            "dropoff_latitude": [40.76, 40.76],
            "dropoff_longitude": [-73.99, -73.99],
            "pickup_datetime": ["2015-01-01 10:00:00", "2015-01-01 10:00:00"],
            "fare_amount": [10.0, 20.0],
            "passenger_count": [1, 1],
        }
    )

    # Create synthetic Learner data (Subset of wisdom effectively, or new data)
    # Let's say this learner row corresponds to the first wisdom row (fare 10).
    # Ideally, we want to see if the encoder subtracts this row from the global stats.
    # Note: FactorizedEncoder.transform_learner assumes the learner rows are part of the distribution
    # but we are simulating 'Rest of World'.
    # If we pass a row with fare 12 that maps to the same key.
    learner_data = pd.DataFrame(
        {
            "pickup_latitude": [40.75],
            "pickup_longitude": [-73.98],
            "dropoff_latitude": [40.76],
            "dropoff_longitude": [-73.99],
            "pickup_datetime": ["2015-01-01 10:00:00"],
            "fare_amount": [12.0],  # This is the target for this row
            "passenger_count": [1],
        }
    )

    encoder = FactorizedEncoder(n_splits=2)

    # Fit Wisdom
    encoder.fit_wisdom(wisdom_data)

    # Check if global stats recorded correctly
    # Key generation: 40.75/0.00135 (L7) ~ 30185.
    # We just check if the key exists in stats.
    l7_stats = encoder.global_stats["spatial_L7"]
    assert l7_stats["sum"].sum() == 30.0, "Wisdom stats sum incorrect."
    assert l7_stats["count"].sum() == 2, "Wisdom stats count incorrect."

    # Transform Learner
    # We expect the logic: (Global_Sum - Fold_Sum) / (Global_Count - Fold_Count)
    # Since n_splits=2 and we have 1 row, that row is in the val set of one fold.
    # Global Sum for this key is 30. Fold Sum (this row) is 12.
    # Rest Sum = 18. Rest Count = 2 - 1 = 1.
    # Expected Mean Fare = 18 / 1 = 18.0.

    # Note: The KFold is random. With 1 row, it will be in the test set of one fold.
    # The code iterates all folds.

    transformed_learner = encoder.transform_learner(learner_data)
    result_mean = transformed_learner["mean_fare_L7"].iloc[0]

    # If the logic works, it should be 18.0.
    # If it didn't subtract, it would be 30/2 = 15.0 (Global Mean).
    # If it fell back to global average (due to key mismatch), it would be (10+20)/2 = 15.0.

    print(f"Synthetic Test Result: {result_mean}")
    assert np.isclose(
        result_mean, 18.0
    ), f"Vectorized subtraction logic failed. Expected 18.0, got {result_mean}"

    print("FactorizedEncoder logic verified.")

    # -------------------------------------------------------------------------
    # 4. Integration Test: Data Loading & Processing
    # -------------------------------------------------------------------------
    print(
        "\n[Step 4] Integration Test - Loading & Processing Real Data (Debug Mode)..."
    )

    # Use the process_data wrapper which uses TaxiDataLoader internally
    # debug=True ensures we use subsamples and don't load the full 55M rows
    X_train, y_train, X_val, y_val, X_test, test_keys = process_data(
        load_cached_data=False, debug=True
    )

    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Val Shape: {X_val.shape}")

    # Validation Checks
    assert not X_train.isnull().values.any(), "X_train contains NaNs after processing."
    assert not X_val.isnull().values.any(), "X_val contains NaNs after processing."
    assert "mean_fare_L7" in X_train.columns, "Feature 'mean_fare_L7' missing."
    assert "temporal_fare" in X_train.columns, "Feature 'temporal_fare' missing."

    # Check value ranges
    assert (
        y_train.min() >= Config.LOOSE_FILTER["min_fare"]
    ), "Target contains values below min_fare."

    print("Data processing pipeline verified.")

    # -------------------------------------------------------------------------
    # 5. Integration Test: Model Training & Prediction
    # -------------------------------------------------------------------------
    print("\n[Step 5] Integration Test - Training & Prediction...")

    trainer = XGBTrainer()

    # Train
    rmse = trainer.train(X_train, y_train, X_val, y_val)
    print(f"Training complete. RMSE: {rmse:.4f}")

    assert rmse > 0, "RMSE should be positive."
    assert os.path.exists(Config.MODEL_OUTPUT_PATH), "Model artifact not found."

    # Predict
    preds = trainer.predict(X_test)

    assert len(preds) == len(X_test), "Prediction length mismatch."
    assert (
        preds >= 2.5
    ).all(), "Predictions contain values below minimum fare ($2.50)."

    print("Model training and prediction verified.")

    # -------------------------------------------------------------------------
    # 6. Full Pipeline Execution (Submission Generation)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Full Pipeline Wrapper...")

    # This generates the submission csv
    train_and_predict(X_train, y_train, X_val, y_val, X_test, test_keys)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect."
    assert len(sub_df) == len(X_test), "Submission row count mismatch."

    print("Full pipeline verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
