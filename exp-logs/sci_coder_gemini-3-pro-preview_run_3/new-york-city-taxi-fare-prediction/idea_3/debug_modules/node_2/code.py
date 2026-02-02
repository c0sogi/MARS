import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import random

# Import provided library modules
import library.config as config
import library.utils as utils
from library.spatial_encoder import SpatialTargetEncoder
from library.data_pipeline import load_and_process
from library.trainers import train_xgboost_model, train_lightgbm_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration Overrides
    print(">>> Setting up environment and patching configuration for speed...")
    set_seed(42)
    warnings.filterwarnings("ignore")

    # Override config constants to ensure the demo runs quickly
    # Reduce clustering complexity
    config.N_CLUSTERS = 20

    # Reduce Model complexity for demo purposes
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = 5
    # Ensure GPU is used if available, otherwise it might fallback or error depending on env.
    # The prompt says A100 is available.

    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_rounds"] = 5
    config.LGBM_PARAMS["num_leaves"] = 31  # Reduce from 512 for small data

    # Redirect cache to a demo folder to avoid conflicts
    DEMO_CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_cache")
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    config.CACHE_DIR = DEMO_CACHE_DIR
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print("Configuration patched successfully.")

    # 2. Verify Utility Functions
    print("\n>>> Verifying library.utils...")

    # Create dummy data
    dummy_data = {
        "pickup_latitude": [40.7128, 40.7580],
        "pickup_longitude": [-74.0060, -73.9855],
        "dropoff_latitude": [40.7060, 40.7829],
        "dropoff_longitude": [-74.0088, -73.9654],
    }
    df_dummy = pd.DataFrame(dummy_data)

    # Test Haversine
    dists = utils.haversine_distance(
        df_dummy["pickup_latitude"],
        df_dummy["pickup_longitude"],
        df_dummy["dropoff_latitude"],
        df_dummy["dropoff_longitude"],
    )
    assert len(dists) == 2
    assert np.all(dists > 0), "Distances must be positive"
    print(" - Haversine distance calculation verified.")

    # Test Rotation
    df_rot = utils.rotate_coordinates(df_dummy.copy())
    expected_rot_cols = [
        "pickup_x_rot",
        "pickup_y_rot",
        "dropoff_x_rot",
        "dropoff_y_rot",
    ]
    for col in expected_rot_cols:
        assert col in df_rot.columns, f"Missing rotated column: {col}"
    print(" - Coordinate rotation verified.")

    # Test Landmarks
    df_land = utils.add_landmark_features(df_dummy.copy())
    # Check for a few expected landmark columns
    assert "dist_pickup_JFK" in df_land.columns
    assert "dist_dropoff_WTC" in df_land.columns
    print(" - Landmark feature generation verified.")

    # 3. Verify Spatial Encoder
    print("\n>>> Verifying library.spatial_encoder...")

    # Create slightly larger dummy data for clustering
    # Generate 100 random points around NYC
    lat_center, lon_center = 40.7128, -74.0060
    n_samples = 100

    df_spatial = pd.DataFrame(
        {
            "pickup_latitude": np.random.normal(lat_center, 0.01, n_samples),
            "pickup_longitude": np.random.normal(lon_center, 0.01, n_samples),
            "dropoff_latitude": np.random.normal(lat_center, 0.01, n_samples),
            "dropoff_longitude": np.random.normal(lon_center, 0.01, n_samples),
        }
    )
    y_spatial = np.random.uniform(5, 50, n_samples)

    encoder = SpatialTargetEncoder(n_clusters=5, random_state=42)

    # Test fit_transform (Training mode)
    df_encoded_train = encoder.fit_transform(df_spatial, y_spatial, n_splits=3)
    assert "pickup_cluster_fare" in df_encoded_train.columns
    assert "dropoff_cluster_fare" in df_encoded_train.columns
    assert not df_encoded_train["pickup_cluster_fare"].isnull().any()
    print(" - SpatialTargetEncoder fit_transform verified.")

    # Test transform (Inference mode)
    df_encoded_test = encoder.transform(df_spatial)
    assert "pickup_cluster_fare" in df_encoded_test.columns
    assert not df_encoded_test["pickup_cluster_fare"].isnull().any()
    print(" - SpatialTargetEncoder transform verified.")

    # 4. Verify Data Pipeline
    print("\n>>> Verifying library.data_pipeline...")
    print("Running load_and_process with sampling (n=5000)...")

    # We use debug_sample_size to limit data loading and processing time
    # load_cached_data=False forces the pipeline to run logic instead of loading old files
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process(
        load_cached_data=False, debug_sample_size=5000
    )

    # Assertions
    print(f" - Train shape: {X_train.shape}")
    print(f" - Val shape: {X_val.shape}")
    print(f" - Test shape: {X_test.shape}")

    assert len(X_train) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Check for engineered features
    expected_features = [
        "hour",
        "year",
        "haversine_dist",
        "pickup_cluster_fare",
        "dist_pickup_JFK",
    ]
    # Note: haversine_dist is not explicitly added in data_pipeline, but let's check what IS added.
    # data_pipeline calls apply_geometric_features -> add_landmark_features.
    # It does NOT explicitly call haversine for point-to-point distance in the pipeline provided in the prompt text
    # (it was in the analysis script, but let's check library/data_pipeline.py content).
    # library/data_pipeline.py calls `apply_geometric_features` -> `add_landmark_features` & `rotate_coordinates`.
    # It does NOT add a generic 'haversine_dist' column between pickup and dropoff in the provided `apply_geometric_features`.
    # However, `SpatialTargetEncoder` adds `pickup_cluster_fare`.
    # `extract_temporal_features` adds `hour`, `year`.

    for feat in [
        "hour",
        "year",
        "pickup_cluster_fare",
        "dist_pickup_JFK",
        "pickup_x_rot",
    ]:
        assert feat in X_train.columns, f"Expected feature {feat} missing in X_train"

    print(" - Data pipeline output structure verified.")

    # 5. Verify Trainers
    print("\n>>> Verifying library.trainers...")

    # XGBoost
    print("Testing XGBoost training...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # LightGBM
    print("Testing LightGBM training...")
    lgb_model = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # Verify predictions
    print("Verifying predictions...")
    xgb_preds = xgb_model.predict(X_val)
    lgb_preds = lgb_model.predict(X_val)

    assert len(xgb_preds) == len(X_val)
    assert len(lgb_preds) == len(X_val)
    assert not np.isnan(xgb_preds).any()
    assert not np.isnan(lgb_preds).any()

    print(" - Model training and prediction verified.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
