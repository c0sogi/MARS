import sys
import os
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library import config
from library import utils
from library.data_processor import TaxiDataProcessor
from library.model import FarePredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Verify Utilities
    print("\n[1/5] Verifying Utilities...")
    utils.set_seed(42)

    # Test Haversine: Distance between (0,0) and (1,0) degrees is ~111.19 km
    dist_test = utils.haversine_distance(0, 0, 1, 0)
    print(f"Test Haversine Distance (0,0) -> (1,0): {dist_test:.4f} km")
    assert 111.0 < dist_test < 111.5, f"Haversine calculation incorrect: {dist_test}"
    print("Utility checks passed.")

    # 2. Data Processing
    print("\n[2/5] Processing Data...")
    processor = TaxiDataProcessor()

    # Use a small sample size for the demo to ensure speed
    DEMO_SAMPLE_SIZE = 20000

    # Process Train
    print(f"Processing training data (sample={DEMO_SAMPLE_SIZE})...")
    train_df = processor.process_data(
        "train", load_cached_data=False, debug_sample_size=DEMO_SAMPLE_SIZE
    )

    # Validate Train Data
    assert not train_df.empty, "Training dataframe is empty."
    assert "distance_haversine" in train_df.columns, "Feature engineering failed."
    # Check cleaning logic (fares should be >= 2.5 based on config)
    min_fare = train_df[config.FEATURE_CONFIG["target_col"]].min()
    assert (
        min_fare >= config.FEATURE_CONFIG["bounds"]["fare_min"]
    ), f"Data cleaning failed: Found fare {min_fare} < {config.FEATURE_CONFIG['bounds']['fare_min']}"

    # Process Val
    print(f"Processing validation data (sample={DEMO_SAMPLE_SIZE})...")
    val_df = processor.process_data(
        "val", load_cached_data=False, debug_sample_size=DEMO_SAMPLE_SIZE
    )
    assert not val_df.empty, "Validation dataframe is empty."

    # Process Test (Full size, as it's small ~10k)
    print("Processing test data...")
    test_df = processor.process_data("test", load_cached_data=False)
    assert len(test_df) > 0, "Test dataframe is empty."

    # 3. Model Configuration & Training
    print("\n[3/5] Configuring and Training Model...")
    predictor = FarePredictor()

    # OPTIMIZATION: Override default n_estimators (10000) to 50 for quick demo
    predictor.params["n_estimators"] = 50
    predictor.params["verbosity"] = -1
    predictor.train_config["early_stopping_rounds"] = 5
    predictor.train_config["verbose_eval"] = False

    # Re-initialize the internal LightGBM model with the optimized parameters
    # (Since the class initializes it in __init__ with the old params)
    predictor.model = lgb.LGBMRegressor(**predictor.params)

    print(f"Training LightGBM with {predictor.params['n_estimators']} estimators...")
    predictor.fit(train_df, val_df)
    print("Training complete.")

    # 4. Prediction & Persistence
    print("\n[4/5] Generating Predictions and Saving Model...")

    # Predict
    predictions = predictor.predict(test_df)

    # Validate Predictions
    assert len(predictions) == len(test_df), "Prediction count mismatch."
    assert not np.isnan(predictions).any(), "Predictions contain NaNs."
    print(f"Generated {len(predictions)} predictions.")

    # Test Model Persistence
    model_save_path = os.path.join(config.WORKING_DIR, "demo_model.pkl")
    predictor.save_model(model_save_path)
    assert os.path.exists(model_save_path), "Model file was not created."

    # Reload to verify
    predictor.load_model(model_save_path)
    print("Model persistence verified.")

    # 5. Submission
    print("\n[5/5] Creating Submission File...")
    submission_df = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

    # Ensure submission directory exists (handled by config, but double check path)
    submission_path = config.DATA_PATHS["submission"]
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file missing."
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (len(test_df), 2), "Submission file shape mismatch."
    assert list(saved_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns mismatch."

    print("\n--- Demonstration Complete Success ---")


if __name__ == "__main__":
    main()
