import os
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import gc

# Import from the provided library
from library.config import ProjectConfig
from library.utils import (
    haversine_distance,
    manhattan_distance,
    bearing,
    calculate_geohash,
    rotate_coordinates,
    clamp_coordinates,
)
from library.stats_manager import StatsManager
from library.data_pipeline import DataPipeline
from library.feature_builder import FeatureBuilder
from library.model_trainer import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_dummy_data(path, num_rows=1000, is_test=False):
    """Generates synthetic data adhering to the schema expected by the pipeline."""
    # NYC Coordinates approx center
    base_lat = 40.75
    base_lon = -73.98

    data = {
        "key": [f"id_{i}" for i in range(num_rows)],
        "pickup_datetime": pd.date_range(
            start="2015-01-01", periods=num_rows, freq="min"
        ).astype(str),
        "pickup_longitude": np.random.normal(base_lon, 0.01, num_rows),
        "pickup_latitude": np.random.normal(base_lat, 0.01, num_rows),
        "dropoff_longitude": np.random.normal(base_lon, 0.01, num_rows),
        "dropoff_latitude": np.random.normal(base_lat, 0.01, num_rows),
        "passenger_count": np.random.randint(1, 6, num_rows),
    }

    # Add target for train/val
    if not is_test:
        # Simple linear relation for sanity check
        dist = np.sqrt(
            (data["pickup_latitude"] - data["dropoff_latitude"]) ** 2
            + (data["pickup_longitude"] - data["dropoff_longitude"]) ** 2
        )
        data["fare_amount"] = 5.0 + dist * 1000.0 + np.random.normal(0, 1, num_rows)
        # Ensure some valid wisdom data (fare > 2.5, dist > 0.2km approx)
        # 0.002 degrees is roughly 0.2km
        mask = dist > 0.002
        # Force some rows to be valid for wisdom filter
        data["fare_amount"] = np.maximum(data["fare_amount"], 3.0)

    df = pd.DataFrame(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def setup_demo_environment():
    """Overrides ProjectConfig to use a temporary directory and small data."""
    print("Setting up demo environment...")

    # Define paths in working directory
    demo_dir = "./working/demo_env"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    demo_input = os.path.join(demo_dir, "input")
    demo_cache = os.path.join(demo_dir, "cache")
    demo_submission = os.path.join(demo_dir, "submission")

    os.makedirs(demo_input, exist_ok=True)
    os.makedirs(demo_cache, exist_ok=True)
    os.makedirs(demo_submission, exist_ok=True)

    # Generate dummy data
    train_path = os.path.join(demo_input, "train.parquet")
    val_path = os.path.join(demo_input, "val.parquet")
    test_path = os.path.join(demo_input, "test.parquet")

    generate_dummy_data(train_path, num_rows=2000, is_test=False)
    generate_dummy_data(val_path, num_rows=500, is_test=False)
    generate_dummy_data(test_path, num_rows=100, is_test=True)

    # Monkeypatch ProjectConfig
    ProjectConfig.INPUT_DIR = demo_input
    ProjectConfig.METADATA_DIR = demo_input
    ProjectConfig.WORKING_DIR = demo_dir
    ProjectConfig.CACHE_DIR = demo_cache
    ProjectConfig.SUBMISSION_DIR = demo_submission

    ProjectConfig.TRAIN_PATH = train_path
    ProjectConfig.VAL_PATH = val_path
    ProjectConfig.TEST_PATH = test_path

    # Speed up training
    ProjectConfig.TRAIN_SUBSAMPLE_SIZE = 1500  # Use most of the dummy data
    ProjectConfig.XGB_PARAMS["n_estimators"] = 10
    ProjectConfig.XGB_PARAMS["n_jobs"] = 2
    ProjectConfig.EARLY_STOPPING_ROUNDS = 5
    ProjectConfig.VERBOSE_EVAL = 1

    print("Environment setup complete.")


def test_utils():
    print("\n=== Testing Library Utils ===")

    # 1. Haversine
    # Distance between (0,0) and (0,1) degree is approx 111km
    d = haversine_distance(0, 0, 0, 1)
    assert np.isclose(d, 111.19, atol=1.0), f"Haversine calculation incorrect: {d}"
    print("Haversine Distance: OK")

    # 2. Geohash
    lats = np.array([40.75])
    lons = np.array([-73.98])
    gh = calculate_geohash(lats, lons, precision=5)
    assert isinstance(gh, np.ndarray), "Geohash should return numpy array"
    assert gh.dtype == np.int32, "Geohash should be int32"
    print("Geohash Calculation: OK")

    # 3. Rotate Coordinates
    lat_rot, lon_rot = rotate_coordinates(lats, lons)
    assert len(lat_rot) == 1, "Rotation output shape mismatch"
    print("Coordinate Rotation: OK")


def test_stats_manager():
    print("\n=== Testing StatsManager ===")
    sm = StatsManager()

    # Force re-computation
    stats = sm.compute_global_moments(load_cached=False)

    # Verify structure of returned stats
    expected_keys = (
        [f"global_L{l}" for l in ProjectConfig.GEOHASH_LEVELS]
        + [f"fold_L{l}" for l in ProjectConfig.GEOHASH_LEVELS]
        + ["global_L5_hour"]
    )

    for k in expected_keys:
        assert k in stats, f"Missing stat key: {k}"
        assert not stats[k].empty, f"Stat DataFrame {k} is empty"

    print(f"Stats computed successfully. Keys: {list(stats.keys())}")
    return stats


def test_feature_builder(stats):
    print("\n=== Testing FeatureBuilder ===")
    fb = FeatureBuilder()

    # Create a small dataframe
    df = pd.DataFrame(
        {
            "key": ["test_1"],
            "pickup_datetime": ["2015-01-01 12:00:00"],
            "pickup_latitude": [40.75],
            "pickup_longitude": [-73.98],
            "dropoff_latitude": [40.76],
            "dropoff_longitude": [-73.99],
            "passenger_count": [1],
        }
    )

    # 1. Geometric Features
    df_geo = fb.add_geometric_features(df)
    expected_cols = ["distance_haversine", "bearing", "geohash_5", "hour"]
    for c in expected_cols:
        assert c in df_geo.columns, f"Missing geometric feature: {c}"

    # 2. Stats Enrichment (Inference Mode)
    df_enriched = fb.enrich_with_stats(df_geo, stats)
    expected_stats = ["mean_fare_L5", "std_fare_L5"]
    for c in expected_stats:
        assert c in df_enriched.columns, f"Missing statistical feature: {c}"

    print("Feature Builder: OK")


def test_data_pipeline():
    print("\n=== Testing DataPipeline ===")
    dp = DataPipeline()

    # Force processing from scratch
    train, val, test = dp.get_data(load_cached=False)

    assert not train.empty, "Train set is empty"
    assert not val.empty, "Val set is empty"
    assert not test.empty, "Test set is empty"

    # Check for leakage prevention features in Train (should have fold subtraction applied)
    # Ideally we check if 'mean_fare_L5' exists.
    assert "mean_fare_L5" in train.columns
    assert "fare_amount" in train.columns
    assert "fare_amount" not in test.columns

    print(
        f"Data Pipeline loaded: Train={train.shape}, Val={val.shape}, Test={test.shape}"
    )


def test_model_training():
    print("\n=== Testing ModelTrainer ===")
    trainer = ModelTrainer()

    # Train
    trainer.train_model(load_cached_data=True)  # Use data cached by pipeline test

    # Check Model Artifact
    model_path = os.path.join(ProjectConfig.CACHE_DIR, "xgb_model.json")
    assert os.path.exists(model_path), "Model file was not saved."

    # Check Submission
    sub_path = os.path.join(ProjectConfig.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    sub_df = pd.read_csv(sub_path)
    assert "key" in sub_df.columns and "fare_amount" in sub_df.columns
    assert (
        len(sub_df) == 100
    ), f"Submission length mismatch. Expected 100, got {len(sub_df)}"

    print("Model Training and Prediction: OK")


def main():
    set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Unit Tests
    test_utils()

    # 3. Integration Tests
    # We pass stats from test_stats_manager to feature_builder test to save time
    stats = test_stats_manager()
    test_feature_builder(stats)

    # 4. Pipeline Execution
    test_data_pipeline()

    # 5. Full Training Loop
    test_model_training()

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
