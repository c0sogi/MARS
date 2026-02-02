import os
import sys
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

# Import library modules
from library.utils import (
    clamp_coordinates,
    haversine_array,
    manhattan_distance,
    rotate_coordinates,
)
from library.features import FeatureEngineer
from library.encoders import GlobalRouteEncoder
from library.data_pipeline import TaxiDataLoader
from library.model import FarePredictor
import library.config as config

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_dummy_data(n_rows=100):
    """Creates a synthetic dataframe mimicking the taxi dataset schema."""
    # Generate random coordinates around NYC
    base_lat = 40.75
    base_lon = -73.98

    data = {
        "key": [f"id_{i}" for i in range(n_rows)],
        "pickup_datetime": [
            (datetime(2015, 1, 1) + timedelta(hours=i)).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            for i in range(n_rows)
        ],
        "pickup_longitude": np.random.normal(base_lon, 0.01, n_rows),
        "pickup_latitude": np.random.normal(base_lat, 0.01, n_rows),
        "dropoff_longitude": np.random.normal(base_lon, 0.01, n_rows),
        "dropoff_latitude": np.random.normal(base_lat, 0.01, n_rows),
        "passenger_count": np.random.randint(1, 6, n_rows),
        "fare_amount": np.random.uniform(5.0, 50.0, n_rows),  # Target
    }
    return pd.DataFrame(data)


# ==========================================
# DEMONSTRATION FUNCTIONS
# ==========================================


def demonstrate_utils():
    print("\n=== Demonstrating library.utils ===")

    # Create outliers to test clamping
    df = pd.DataFrame(
        {
            "pickup_longitude": [
                -75.0,
                -73.98,
                -70.0,
            ],  # -75 and -70 are outside typical NYC box
            "pickup_latitude": [40.75, 40.75, 45.0],
            "dropoff_longitude": [-73.98, -73.98, -73.98],
            "dropoff_latitude": [40.75, 40.75, 40.75],
        }
    )

    # 1. Test Clamp Coordinates
    print("Testing clamp_coordinates...")
    clamped_df = clamp_coordinates(df, bounding_box=config.NYC_BOUNDING_BOX)

    # Check if values are within bounds
    assert clamped_df["pickup_longitude"].min() >= config.NYC_BOUNDING_BOX["min_lon"]
    assert clamped_df["pickup_latitude"].max() <= config.NYC_BOUNDING_BOX["max_lat"]
    print("-> Clamping verified.")

    # 2. Test Distance Calculations
    print("Testing distance functions...")
    # Distance between same points should be 0
    d_hav = haversine_array(40.75, -73.98, 40.75, -73.98)
    d_man = manhattan_distance(40.75, -73.98, 40.75, -73.98)

    assert np.isclose(d_hav, 0.0), "Haversine distance for same point should be 0"
    assert np.isclose(d_man, 0.0), "Manhattan distance for same point should be 0"

    # Distance between 1 degree lat (approx 111km)
    d_hav_1deg = haversine_array(40.0, -73.0, 41.0, -73.0)
    assert 100 < d_hav_1deg < 120, f"Haversine calculation seems off: {d_hav_1deg}"
    print("-> Distance calculations verified.")

    # 3. Test Rotation
    print("Testing rotate_coordinates...")
    rot_df = rotate_coordinates(df, angle=45)
    expected_cols = [
        "pickup_lon_rot",
        "pickup_lat_rot",
        "dropoff_lon_rot",
        "dropoff_lat_rot",
    ]
    for col in expected_cols:
        assert col in rot_df.columns, f"Missing rotated column: {col}"
    print("-> Rotation verified.")


def demonstrate_features():
    print("\n=== Demonstrating library.features ===")

    df = create_dummy_data(10)

    # Initialize FeatureEngineer
    fe = FeatureEngineer(rotation_angle=45, clamp_input=True)

    print("Transforming data with FeatureEngineer...")
    df_transformed = fe.transform(df)

    # Verify Temporal Features
    assert "hour" in df_transformed.columns
    assert "year" in df_transformed.columns
    assert "dayofweek" in df_transformed.columns

    # Verify Spatial Features
    assert "dist_km" in df_transformed.columns
    assert "dist_manhattan" in df_transformed.columns
    assert "abs_diff_lon" in df_transformed.columns

    # Verify Rotation
    assert "pickup_lon_rot" in df_transformed.columns

    print("-> Feature engineering verified.")


def demonstrate_encoders():
    print("\n=== Demonstrating library.encoders ===")

    # Need enough rows for K-Fold (n_splits=5)
    train_df = create_dummy_data(20)
    test_df = create_dummy_data(5)
    # Remove target from test to simulate real scenario
    test_df = test_df.drop(columns=["fare_amount"])

    encoder = GlobalRouteEncoder(grid_precision=3, n_splits=5, random_state=42)

    # 1. Fit and Transform OOF on Train
    print("Running fit_transform_oof...")
    train_encoded = encoder.fit_transform_oof(train_df)

    assert "oof_fare" in train_encoded.columns
    assert not train_encoded["oof_fare"].isnull().all()
    print("-> OOF Encoding successful.")

    # 2. Transform Test (Global Map)
    print("Running transform_global...")
    test_encoded = encoder.transform_global(test_df)

    assert "oof_fare" in test_encoded.columns
    # Should use global mean for unknown routes (which these random ones likely are)
    assert not test_encoded["oof_fare"].isnull().any()
    print("-> Global transformation successful.")


def demonstrate_pipeline_and_model():
    print("\n=== Demonstrating library.data_pipeline & library.model ===")

    # 1. Setup Data Pipeline
    loader = TaxiDataLoader()

    # MOCKING: Override load_raw_data to use dummy data instead of reading 55M rows
    # This ensures the demo runs instantly while still exercising the pipeline logic.
    print("Generating synthetic datasets for pipeline demonstration...")
    dummy_train = create_dummy_data(500)
    dummy_val = create_dummy_data(100)
    dummy_test = create_dummy_data(50)
    dummy_test = dummy_test.drop(columns=["fare_amount"])  # Test set has no target

    # Monkeypatch the instance method
    loader.load_raw_data = lambda: (dummy_train, dummy_val, dummy_test)

    # Override cache directory to a temporary location to verify processing logic
    loader.cache_dir = "./working/demo_pipeline_cache/"
    if os.path.exists(loader.cache_dir):
        shutil.rmtree(loader.cache_dir)
    os.makedirs(loader.cache_dir, exist_ok=True)

    # Override subsample size to ensure we use all our dummy data
    loader.subsample_size = 500

    # Execute Pipeline
    print("Executing TaxiDataLoader.get_processed_data()...")
    X_train, y_train, X_val, y_val, X_test, test_keys = loader.get_processed_data(
        load_cached_data=False
    )

    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Val Shape: {X_val.shape}")
    print(f"Processed Test Shape: {X_test.shape}")

    assert X_train.shape[0] == 500
    assert X_val.shape[0] == 100
    assert X_test.shape[0] == 50
    assert "oof_fare" in X_train.columns
    assert "dist_km" in X_train.columns

    # 2. Setup and Train Model
    print("Initializing FarePredictor...")
    # Modify params for speed (reduce depth, use CPU if GPU not available, though GPU is present)
    fast_params = config.XGB_PARAMS.copy()
    fast_params["max_depth"] = 3

    predictor = FarePredictor(params=fast_params)

    print("Training model (fast mode)...")
    # Train for very few rounds just to verify the loop
    predictor.train(
        X_train, y_train, X_val, y_val, num_boost_round=10, early_stopping_rounds=5
    )

    # 3. Prediction
    print("Generating predictions...")
    preds = predictor.predict(X_test)

    assert len(preds) == 50
    assert (preds >= 2.50).all(), "Predictions must be floored at $2.50"
    print(f"Sample predictions: {preds[:5]}")

    # 4. Save Submission
    submission_path = "./working/demo_submission.csv"
    predictor.save_submission(test_keys, preds, output_path=submission_path)

    assert os.path.exists(submission_path)
    print("-> Pipeline and Model flow verified.")


def main():
    set_seed(42)

    try:
        demonstrate_utils()
        demonstrate_features()
        demonstrate_encoders()
        demonstrate_pipeline_and_model()
        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\nValidation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
