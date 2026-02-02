import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_processing as dp
import library.feature_engineering as fe
import library.model_training as mt


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_synthetic_data(output_dir):
    """
    Creates small synthetic parquet files matching the schema of the real dataset
    to allow for rapid testing of the pipeline components.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Common schema generation
    n_rows = 1000

    # Generate random data within a reasonable range for NYC
    data = {
        "key": [f"id_{i}" for i in range(n_rows)],
        "pickup_datetime": pd.date_range(
            start="2015-01-01", periods=n_rows, freq="h"
        ).astype(str),
        "pickup_longitude": np.random.uniform(-74.0, -73.9, n_rows),
        "pickup_latitude": np.random.uniform(40.7, 40.8, n_rows),
        "dropoff_longitude": np.random.uniform(-74.0, -73.9, n_rows),
        "dropoff_latitude": np.random.uniform(40.7, 40.8, n_rows),
        "passenger_count": np.random.randint(1, 6, n_rows),
        "fare_amount": np.random.uniform(5.0, 50.0, n_rows),
    }

    df = pd.DataFrame(data)

    # Save Train
    df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)

    # Save Val (same structure)
    df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)

    # Save Test (drop fare_amount)
    df_test = df.drop(columns=["fare_amount"])
    df_test.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    print(f"Synthetic data created in {output_dir}")
    return df


def test_utils():
    print("\n=== Testing Library: utils.py ===")

    # 1. Test Haversine
    # Distance between (0,0) and (1,0) degrees latitude is approx 111.19 km
    lat1, lon1 = np.array([0.0]), np.array([0.0])
    lat2, lon2 = np.array([1.0]), np.array([0.0])
    dist = utils.haversine_array(lat1, lon1, lat2, lon2)

    print(f"Haversine calculation (1 deg lat): {dist[0]:.4f} km")
    assert np.isclose(dist[0], 111.19, atol=1.0), "Haversine calculation incorrect"

    # 2. Test Clamp Coordinates
    # Create data with outliers
    df_outlier = pd.DataFrame(
        {
            "pickup_longitude": [-100.0, -74.0],  # -100 is outlier
            "pickup_latitude": [40.75, 40.75],
            "dropoff_longitude": [-73.9, -73.9],
            "dropoff_latitude": [40.75, 40.75],
        }
    )

    # Config bounds: lon_min: -74.50
    clamped_df = utils.clamp_coordinates(df_outlier)
    val = clamped_df.iloc[0]["pickup_longitude"]
    expected = config.NYC_BOUNDING_BOX["lon_min"]

    print(f"Clamping check: Input -100.0 -> Output {val}")
    assert val == expected, f"Clamping failed. Expected {expected}, got {val}"
    print("Utils verification passed.")


def test_data_processing():
    print("\n=== Testing Library: data_processing.py ===")

    # We rely on the monkey-patched paths set in main()
    # Test processing of 'train' split
    # force load_cached_data=False to ensure logic runs
    processed_df = dp.process_data("train", load_cached_data=False)

    # Verify Feature Engineering
    expected_cols = ["year", "hour", "dist_haversine", "bearing"]
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing feature: {col}"

    print(f"Processed dataframe shape: {processed_df.shape}")
    print("Data processing verification passed.")
    return processed_df


def test_feature_engineering(train_df):
    print("\n=== Testing Library: feature_engineering.py ===")

    # Test SpatialTargetEncoder
    encoder = fe.SpatialTargetEncoder(
        smoothing_params={"k_folds": 2, "smoothing": 10, "min_samples_leaf": 1}
    )

    # Fit Transform on Train
    # We need to ensure we don't use the cached file from previous runs if we want to test logic
    # But for this demo, we can just run the encoder directly on the df
    encoded_values = encoder.fit_transform(train_df, target_col="fare_amount")

    assert len(encoded_values) == len(train_df), "Encoded values length mismatch"
    assert not np.isnan(encoded_values).any(), "NaNs found in target encoding"

    print(f"Target Encoding Mean: {encoded_values.mean():.4f}")

    # Verify it was added to the dataframe in the pipeline wrapper
    # We call the high-level function
    df_encoded = fe.get_target_encoded_data("train", load_cached_data=False)
    assert (
        "route_avg_fare" in df_encoded.columns
    ), "route_avg_fare column missing after pipeline"

    print("Feature engineering verification passed.")


def test_model_training():
    print("\n=== Testing Library: model_training.py ===")

    # Load data
    train_df = fe.get_target_encoded_data("train", load_cached_data=True)
    val_df = fe.get_target_encoded_data("val", load_cached_data=True)

    # Define fast hyperparameters
    fast_params = config.XGB_PARAMS.copy()
    fast_params["n_estimators"] = 10  # Very few trees for speed
    fast_params["max_depth"] = 3

    # Train
    print("Training XGBoost model (fast mode)...")
    model, features = mt.train_model(train_df, val_df, params=fast_params)

    # Evaluate
    rmse = mt.evaluate_model(model, val_df, features)
    assert rmse >= 0, "RMSE should be non-negative"

    # Predict
    test_df = fe.get_target_encoded_data("test", load_cached_data=False)
    submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    mt.predict_and_submit(model, test_df, features, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Validate submission format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print("Model training and submission verification passed.")


def main():
    # 0. Setup
    set_seed(42)
    warnings.filterwarnings("ignore")

    # Define temporary directories for the demo
    demo_base = "./working/demo_execution"
    demo_metadata = os.path.join(demo_base, "metadata")
    demo_working = os.path.join(demo_base, "working")

    # Clean up previous runs if any
    if os.path.exists(demo_base):
        shutil.rmtree(demo_base)

    # 1. Create Synthetic Data
    create_synthetic_data(demo_metadata)

    # 2. Monkey-Patch Configuration
    # This redirects the library to use our synthetic data and temp working dir
    print(f"Redirecting config.METADATA_DIR to {demo_metadata}")
    config.METADATA_DIR = demo_metadata

    print(f"Redirecting config.WORKING_DIR to {demo_working}")
    config.WORKING_DIR = demo_working
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Also update the internal stats paths in feature_engineering class instances
    # Note: The class uses WORKING_DIR at instantiation or method call time usually,
    # but the paths are defined in __init__. We need to ensure new instances pick up the change.
    # Since we imported the module 'fe', and the class uses 'WORKING_DIR' imported from config,
    # changing config.WORKING_DIR affects future usages if they reference config.WORKING_DIR directly.
    # However, `fe.SpatialTargetEncoder` sets `self.stats_path` in `__init__` using `WORKING_DIR`.
    # We must ensure we instantiate the class AFTER patching.

    # 3. Run Tests
    try:
        test_utils()
        processed_train = test_data_processing()
        test_feature_engineering(processed_train)
        test_model_training()

        print("\n=== All Demonstrations Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Verification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
