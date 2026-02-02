import os
import sys
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Import from the provided library
from library.config import NYC_BB, GEOHASH_LEVELS, WORKING_DIR, SUBMISSION_PATH
from library.utils import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    vectorized_geohash,
)
from library.data_processor import get_processed_data
from library.feature_engineering import InteractionStatsEngine
from library.model_trainer import XGBoostTrainer, run_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_utils():
    print("\n=== Verifying Utilities ===")

    # 1. Test Distance Metrics
    # Points: NYC (approx) to a point 1 degree lat away
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 41.7128, -74.0060

    h_dist = haversine_distance(lat1, lon1, lat2, lon2)
    m_dist = manhattan_distance(lat1, lon1, lat2, lon2)

    # 1 degree lat is approx 111km
    print(f"Haversine Distance (1 deg lat): {h_dist:.4f} km")
    print(f"Manhattan Distance (1 deg lat): {m_dist:.4f} km")

    assert 110.0 < h_dist < 112.0, "Haversine distance calculation incorrect"
    assert 110.0 < m_dist < 112.0, "Manhattan distance calculation incorrect"

    # 2. Test Geohash
    lats = np.array([40.7128, 40.7580])
    lons = np.array([-74.0060, -73.9855])
    precision = 5
    hashes = vectorized_geohash(lats, lons, precision)

    print(f"Geohashes (L{precision}): {hashes}")
    assert len(hashes) == 2
    assert len(hashes[0]) == precision
    # Known prefix for NYC is 'dr5' or similar
    assert str(hashes[0]).startswith(
        "dr"
    ), "Geohash generation seems incorrect for NYC coordinates"

    # 3. Test Clamping
    df_clamp = pd.DataFrame(
        {
            "pickup_latitude": [40.0, 45.0, 40.75],  # 40.0 < min_lat, 45.0 > max_lat
            "pickup_longitude": [-75.0, -70.0, -73.98],  # -75 < min_lon, -70 > max_lon
            "dropoff_latitude": [40.75, 40.75, 40.75],
            "dropoff_longitude": [-73.98, -73.98, -73.98],
        }
    )

    clamped = clamp_coordinates(df_clamp.copy())

    assert clamped["pickup_latitude"].min() >= NYC_BB["min_lat"]
    assert clamped["pickup_latitude"].max() <= NYC_BB["max_lat"]
    assert clamped["pickup_longitude"].min() >= NYC_BB["min_lon"]
    assert clamped["pickup_longitude"].max() <= NYC_BB["max_lon"]
    print("Coordinate clamping verified.")


def verify_data_processor():
    print("\n=== Verifying Data Processor ===")

    # Use a unique small subsample size to force fresh processing (avoid existing cache collisions)
    demo_subsample = 2000

    # Test loading and processing training data (Loose mode)
    df = get_processed_data(
        "train", mode="loose", subsample_size=demo_subsample, load_cached_data=False
    )

    print(f"Processed Train Shape: {df.shape}")
    assert len(df) > 0
    assert len(df) <= demo_subsample

    # Check for enriched features
    expected_cols = ["haversine_dist", "manhattan_dist", "abs_diff_lon", "abs_diff_lat"]
    for col in expected_cols:
        assert col in df.columns, f"Missing enriched feature: {col}"

    # Check geohash columns
    for level in GEOHASH_LEVELS:
        assert f"pickup_geohash_{level}" in df.columns
        assert f"dropoff_geohash_{level}" in df.columns

    print("Data Processor output verified.")
    return df


def verify_feature_engineering(df_train_sample):
    print("\n=== Verifying Feature Engineering (InteractionStatsEngine) ===")

    # 1. Simulate "Strict" data for Wisdom phase (using the sample for speed)
    # In reality, this would be the full dataset, but we use the sample here.
    df_strict = df_train_sample.copy()

    engine = InteractionStatsEngine(working_dir=WORKING_DIR)

    # Fit the engine
    print("Fitting Stats Engine...")
    engine.fit(df_strict, load_cached=False)

    # Check if stats files were created
    for level in GEOHASH_LEVELS:
        stats_path = engine._get_stats_path(level)
        assert os.path.exists(stats_path), f"Stats file for level {level} not found"

    # 2. Transform Training Data (Learner Phase - Vectorized Subtraction)
    print("Transforming Training Data...")
    df_trans_train = engine.transform_train(
        df_train_sample, num_folds=3
    )  # Reduced folds for speed

    # Check for interaction features
    for level in GEOHASH_LEVELS:
        col = f"mean_fare_L{level}"
        assert (
            col in df_trans_train.columns
        ), f"Interaction feature {col} missing in train"
        # Check that we don't have all NaNs (unless dataset is tiny and disjoint)
        # With 2000 rows, we should have some overlap
        valid_count = df_trans_train[col].notna().sum()
        print(
            f"  Level {level}: {valid_count}/{len(df_trans_train)} valid interaction values"
        )

    # 3. Transform Test/Val Data (Direct Mapping)
    print("Transforming Validation Data...")
    # Just reuse the sample as 'val' for structure testing
    df_trans_val = engine.transform_test(df_train_sample)

    for level in GEOHASH_LEVELS:
        col = f"mean_fare_L{level}"
        assert col in df_trans_val.columns, f"Interaction feature {col} missing in val"

    print("Feature Engineering verified.")
    return df_trans_train, df_trans_val


def verify_model_training(df_train, df_val):
    print("\n=== Verifying Model Training ===")

    # Prepare features
    exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold_id"}
    exclude_cols.update([c for c in df_train.columns if "geohash" in c])

    features = [c for c in df_train.columns if c not in exclude_cols]
    target = "fare_amount"

    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]

    # Initialize Trainer with fast parameters
    fast_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "n_jobs": 4,
        "device": "cpu",  # Use CPU for small demo to avoid overhead/compatibility if GPU busy
    }

    trainer = XGBoostTrainer(
        params=fast_params, model_path=os.path.join(WORKING_DIR, "demo_xgb.json")
    )

    # Train
    print("Training XGBoost...")
    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=10,
        early_stopping_rounds=5,
        verbose_eval=5,
    )

    assert trainer.model is not None

    # Predict
    print("Predicting...")
    preds = trainer.predict(X_val)
    assert len(preds) == len(X_val)
    assert preds.dtype == np.float32

    # Save/Load
    trainer.save_model()
    assert os.path.exists(trainer.model_path)

    trainer.model = None  # Force unload
    trainer.load_model()
    assert trainer.model is not None

    print("Model Training and Persistence verified.")


def verify_full_pipeline():
    print("\n=== Verifying Full Pipeline Execution ===")

    # Run the provided pipeline function with minimal parameters
    # This ensures the integration logic in library/model_trainer.py is correct

    try:
        run_pipeline(
            subsample_size=3000,
            num_boost_round=5,
            early_stopping_rounds=2,
            load_cached_data=False,
        )

        # Check submission file
        assert os.path.exists(SUBMISSION_PATH)
        sub_df = pd.read_csv(SUBMISSION_PATH)
        assert "key" in sub_df.columns
        assert "fare_amount" in sub_df.columns
        assert len(sub_df) > 0
        print(f"Pipeline finished successfully. Submission shape: {sub_df.shape}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise e


def main():
    set_seed(42)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("Starting Library Verification...")

    # 1. Utils
    verify_utils()

    # 2. Data Processor
    df_sample = verify_data_processor()

    # 3. Feature Engineering
    df_train_eng, df_val_eng = verify_feature_engineering(df_sample)

    # 4. Model Training
    verify_model_training(df_train_eng, df_val_eng)

    # 5. Full Pipeline
    verify_full_pipeline()

    print("\nAll verifications passed successfully!")


if __name__ == "__main__":
    main()
