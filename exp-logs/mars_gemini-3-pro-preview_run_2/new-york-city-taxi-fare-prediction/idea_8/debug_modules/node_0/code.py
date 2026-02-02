import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
# Note: We import config to patch mutable parameters for the demo
import library.config
from library.utils import haversine_distance, manhattan_distance, filter_within_bbox
from library.local_features import LocalFeatureGenerator
from library.global_features import SpatialTargetEncoder
from library.data_manager import DataManager
from library.model import FarePredictor


def create_demo_data():
    """
    Creates small subsets of the training and validation data for demonstration purposes.
    """
    print("Creating demo datasets (subset of metadata)...")

    # Define paths
    real_train_path = "./metadata/train.parquet"
    real_val_path = "./metadata/val.parquet"

    demo_train_path = "./working/demo_train.parquet"
    demo_val_path = "./working/demo_val.parquet"

    # Load first 10k rows
    # We use pyarrow engine implicitly via pandas
    df_train = pd.read_parquet(real_train_path).head(10000)
    df_val = pd.read_parquet(real_val_path).head(5000)

    # Save to working dir
    df_train.to_parquet(demo_train_path, index=False)
    df_val.to_parquet(demo_val_path, index=False)

    return demo_train_path, demo_val_path


def verify_utils():
    """
    Verifies utility functions.
    """
    print("Verifying utils.py...")

    # Test Haversine
    # Distance between (0,0) and (0,1) deg is approx 111.19 km
    d = haversine_distance(0, 0, 0, 1)
    assert np.isclose(d, 111.19, atol=0.1), f"Haversine calculation incorrect: {d}"

    # Test BBox Filter
    df = pd.DataFrame(
        {
            "pickup_longitude": [-74.0, -200.0],  # 2nd is out
            "pickup_latitude": [40.7, 40.7],
            "dropoff_longitude": [-74.0, -74.0],
            "dropoff_latitude": [40.7, 40.7],
        }
    )
    bbox = {"min_long": -75, "max_long": -73, "min_lat": 40, "max_lat": 41}
    filtered = filter_within_bbox(df, bbox)
    assert len(filtered) == 1, "BBox filtering failed to remove outlier"

    print("Utils verification passed.")


def verify_local_features():
    """
    Verifies local feature generation.
    """
    print("Verifying local_features.py...")

    df = pd.DataFrame(
        {
            "key": ["k1", "k2"],
            "pickup_datetime": ["2015-01-27 13:08:24 UTC", "2015-01-27 13:08:24 UTC"],
            "pickup_longitude": [-74.00, -73.99],
            "pickup_latitude": [40.73, 40.74],
            "dropoff_longitude": [-73.99, -73.98],
            "dropoff_latitude": [40.74, 40.75],
            "passenger_count": [1, 2],
        }
    )

    gen = LocalFeatureGenerator()
    res = gen.process(df)

    expected_cols = ["hour", "year", "dist_haversine", "abs_diff_lon"]
    for col in expected_cols:
        assert col in res.columns, f"Missing local feature: {col}"

    # Check if original datetime is removed (as per implementation it is not in cols_to_keep)
    assert "pickup_datetime" not in res.columns, "pickup_datetime should be dropped"

    print("Local features verification passed.")


def verify_global_features():
    """
    Verifies global feature generation (SpatialTargetEncoder).
    """
    print("Verifying global_features.py...")

    # Create simple data where route A->B has specific cost
    df_train = pd.DataFrame(
        {
            "key": ["k1", "k2"],
            "pickup_latitude": [40.750, 40.750],
            "pickup_longitude": [-74.000, -74.000],
            "dropoff_latitude": [40.800, 40.800],
            "dropoff_longitude": [-73.950, -73.950],
            "fare_amount": [10.0, 20.0],
        }
    )

    df_test = pd.DataFrame(
        {
            "key": ["k3"],
            "pickup_latitude": [40.750],
            "pickup_longitude": [-74.000],
            "dropoff_latitude": [40.800],
            "dropoff_longitude": [-73.950],
        }
    )

    encoder = SpatialTargetEncoder(grid_rounding=3, k_folds=2)

    # Test Fit (Inference mode)
    encoder.fit(df_train)
    res = encoder.transform(df_test)

    # Mean of 10 and 20 is 15
    assert np.isclose(
        res[0], 15.0
    ), f"Global feature encoding incorrect. Expected 15.0, got {res[0]}"

    print("Global features verification passed.")


def run_pipeline_demo(train_path, val_path):
    """
    Runs the full pipeline using DataManager and FarePredictor.
    """
    print("\n=== Running Full Pipeline Demo ===")

    # 1. Configure DataManager
    dm = DataManager()
    # Override paths to use our small demo data
    dm.train_path = train_path
    dm.val_path = val_path
    # Test path remains default (from metadata)

    # 2. Load and Prepare Data
    # We set load_cached_data=False to ensure we process our new demo data
    # and don't accidentally load cache from a full run.
    print("DataManager: Loading and preparing data...")
    X_train, y_train, X_val, y_val, X_test, test_keys = dm.load_and_prepare_data(
        load_cached_data=False
    )

    print(
        f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    assert not X_train.empty, "X_train is empty"
    assert len(X_train) == len(y_train), "Mismatch in training features/targets"
    assert "route_avg_fare" in X_train.columns, "Global feature missing from X_train"
    assert "dist_haversine" in X_train.columns, "Local feature missing from X_train"

    # 3. Configure Model
    # Patch XGB_PARAMS for speed (Modifying the dictionary in-place)
    library.config.XGB_PARAMS.update(
        {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5,
        }
    )

    predictor = FarePredictor()

    # 4. Train
    print("FarePredictor: Training model...")
    predictor.train(X_train, y_train, X_val, y_val)

    # 5. Predict
    print("FarePredictor: Generating predictions...")
    preds = predictor.predict(X_test)

    assert len(preds) == len(X_test), "Prediction length mismatch"
    assert (preds >= 2.50).all(), "Predictions contain values below minimum fare floor"

    # Create submission dataframe (in memory check)
    submission = pd.DataFrame({"key": test_keys, "fare_amount": preds})

    print("Sample predictions:")
    print(submission.head())

    print("\nPipeline demo completed successfully.")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs("./working", exist_ok=True)

    # 1. Create Data
    demo_train_path, demo_val_path = create_demo_data()

    # 2. Run Unit Verifications
    verify_utils()
    verify_local_features()
    verify_global_features()

    # 3. Run Integration Demo
    run_pipeline_demo(demo_train_path, demo_val_path)
