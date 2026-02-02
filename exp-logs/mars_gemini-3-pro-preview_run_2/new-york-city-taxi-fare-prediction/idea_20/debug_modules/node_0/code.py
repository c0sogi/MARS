import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings

# Import the provided library modules
from library import config
from library import utils
from library import data_manager
from library.stats_computer import StatsEngine
from library.feature_pipeline import FeatureGenerator
from library.model_handler import TaxiFareRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("=== Starting Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring for Demo Run...")

    # Set a custom cache directory for this demo to avoid conflicts
    demo_cache_dir = os.path.join(config.WORKING_DIR, "demo_run")
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Override config constants
    config.CACHE_DIR = demo_cache_dir
    config.TRAIN_SUBSAMPLE_SIZE = 10_000  # Use only 10k rows for training
    config.SEED = 42

    # Override XGBoost params for fast training
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 3
    config.XGB_PARAMS["learning_rate"] = 0.1
    # Ensure we use a compatible device (CPU is safer for tiny datasets/demos,
    # but A100 is available so 'cuda' is fine. We stick to config default 'cuda'
    # but reduce complexity).

    print(f"Cache Directory: {config.CACHE_DIR}")
    print(f"Train Subsample Size: {config.TRAIN_SUBSAMPLE_SIZE}")
    print(f"XGB Estimators: {config.XGB_PARAMS['n_estimators']}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utils...")

    # Test Haversine
    lat1, lon1 = 40.7128, -74.0060  # NYC
    lat2, lon2 = 51.5074, -0.1278  # London
    dist = utils.calculate_haversine(lat1, lon1, lat2, lon2)
    print(f"  Haversine Distance (NYC -> London): {dist:.2f} km")
    assert dist > 5500 and dist < 5600, "Haversine calculation seems off"

    # Test Geohash Encoding
    gh = utils.encode_geohash(lat1, lon1, precision=5)
    print(f"  Geohash (NYC, p=5): {gh}")
    assert isinstance(gh, (str, np.str_)), "Geohash should be a string"
    assert len(str(gh)) == 5, "Geohash precision mismatch"

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading Data via DataManager...")

    # Force load from scratch (load_cached_data=False) to demonstrate processing logic
    # This reads the metadata parquets, clamps coordinates, and splits/subsamples.
    learner_df, wisdom_df, val_df, test_df = data_manager.load_dataset(
        load_cached_data=False
    )

    # Assertions
    assert (
        len(learner_df) == config.TRAIN_SUBSAMPLE_SIZE
    ), f"Learner set size mismatch. Expected {config.TRAIN_SUBSAMPLE_SIZE}, got {len(learner_df)}"
    assert not wisdom_df.empty, "Wisdom set is empty"
    assert not val_df.empty, "Validation set is empty"
    assert not test_df.empty, "Test set is empty"

    # Verify clamping (NYC BBox)
    bbox = config.NYC_BBOX  # [lon_min, lat_min, lon_max, lat_max]
    assert (
        learner_df["pickup_longitude"].min() >= bbox[0]
    ), "Longitude clamping failed (min)"
    assert (
        learner_df["pickup_latitude"].max() <= bbox[3]
    ), "Latitude clamping failed (max)"

    print("  Data Loaded and Verified.")

    # -------------------------------------------------------------------------
    # 4. Statistical Engine (StatsEngine)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Testing StatsEngine...")

    stats_engine = StatsEngine()

    # Compute global stats on wisdom set
    print("  Computing Global Stats...")
    global_stats = stats_engine.compute_global_stats(wisdom_df, load_cached_data=False)

    # Verify structure of global stats
    levels = config.GEOHASH_LEVELS
    for lvl in levels:
        key_route = f"L{lvl}_route"
        key_rate = f"L{lvl}_rate"
        assert key_route in global_stats, f"Missing route stats for level {lvl}"
        assert key_rate in global_stats, f"Missing rate stats for level {lvl}"
        assert not global_stats[key_route].empty, f"Route stats empty for level {lvl}"

    # Test Enrichment on a small slice
    print("  Enriching a sample slice...")
    sample_slice = learner_df.head(100).copy()
    enriched_slice = stats_engine.enrich_data(sample_slice, global_stats, mode="train")

    # Check for new columns
    expected_col = f"mean_fare_L{levels[0]}"
    assert (
        expected_col in enriched_slice.columns
    ), f"Enrichment failed to add {expected_col}"
    print(f"  Enrichment successful. Added columns like {expected_col}.")

    # -------------------------------------------------------------------------
    # 5. Feature Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Feature Pipeline...")

    feature_gen = FeatureGenerator()

    # Process all datasets
    # Note: We pass load_cached_data=False to ensure it runs the logic
    X_train, y_train, X_val, y_val, X_test, test_keys = feature_gen.process(
        learner_df, wisdom_df, val_df, test_df, load_cached_data=False
    )

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Verify features
    assert "dist_haversine" in X_train.columns, "Base feature 'dist_haversine' missing"
    assert "hour" in X_train.columns, "Base feature 'hour' missing"
    assert X_train.shape[0] == len(y_train), "X_train and y_train length mismatch"
    assert X_test.shape[0] == len(test_keys), "X_test and test_keys length mismatch"

    # -------------------------------------------------------------------------
    # 6. Model Training & Prediction
    # -------------------------------------------------------------------------
    print("\n[Step 6] Training Model...")

    model_handler = TaxiFareRegressor()

    # Train
    model_handler.train(X_train, y_train, X_val, y_val)

    # Verify Model State
    assert model_handler.model is not None, "Model object is None after training"
    assert model_handler.best_score is not None, "Best score not recorded"

    # Predict
    print("  Generating Predictions...")
    preds = model_handler.predict(X_test)

    # Verify Predictions
    assert len(preds) == len(X_test), "Prediction length mismatch"
    assert (preds >= 2.50).all(), "Predictions violated minimum fare floor ($2.50)"
    print(f"  Predictions generated. Mean Fare: ${preds.mean():.2f}")

    # Save/Load Check
    print("  Testing Model I/O...")
    model_handler.save_model("demo_model.json")
    model_handler.load_model("demo_model.json")
    print("  Model saved and loaded successfully.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission File...")

    submission = pd.DataFrame({"key": test_keys, "fare_amount": preds})

    output_path = config.SUBMISSION_OUTPUT_PATH
    submission.to_csv(output_path, index=False)

    print(f"  Submission saved to {output_path}")
    print("  Head of submission:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    set_seed(42)
    run_demo()
