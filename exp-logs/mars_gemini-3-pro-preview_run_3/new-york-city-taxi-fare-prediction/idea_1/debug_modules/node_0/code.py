import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config, utils
from library.data_processor import TaxiDataProcessor
from library.model import FareRegressor
from library.trainer import train_model


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_utils():
    print("\n=== Demonstrating library.utils ===")

    # Test coordinates: NYC (approx) to JFK Airport (approx)
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 40.6413, -73.7781

    # Calculate distance
    dist = utils.haversine_distance(lat1, lon1, lat2, lon2)

    print(f"Calculated Haversine Distance: {dist:.4f} km")

    # Assertion: Distance should be positive and roughly within expected range (approx 20-30km)
    assert dist > 0, "Distance must be positive"
    assert 15 < dist < 35, f"Distance {dist} seems incorrect for NYC->JFK"

    # Test Vectorization
    lats1 = np.array([lat1, lat1])
    lons1 = np.array([lon1, lon1])
    lats2 = np.array([lat2, lat2])
    lons2 = np.array([lon2, lon2])

    dists = utils.haversine_distance(lats1, lons1, lats2, lons2)
    assert len(dists) == 2, "Vectorized calculation failed length check"
    assert np.allclose(dists, dist), "Vectorized calculation values mismatch"
    print("Utils verification passed.")


def demo_data_processor():
    print("\n=== Demonstrating library.data_processor ===")

    processor = TaxiDataProcessor()

    # Use a small sample size for speed
    sample_size = 1000

    # Force processing (load_cached_data=False) to verify logic
    print("Processing data with sampling...")
    train_df, val_df, test_df = processor.process_data(
        load_cached_data=False, train_sample_size=sample_size
    )

    # Verifications
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 1. Check Sampling
    assert (
        len(train_df) == sample_size
    ), f"Expected {sample_size} training rows, got {len(train_df)}"

    # 2. Check Feature Engineering
    expected_features = [
        "haversine_dist",
        "abs_diff_lon",
        "abs_diff_lat",
        "year",
        "hour",
    ]
    for feat in expected_features:
        assert feat in train_df.columns, f"Feature {feat} missing from training set"
        assert feat in test_df.columns, f"Feature {feat} missing from test set"

    # 3. Check Cleaning (Train set should not have negative fares)
    if "fare_amount" in train_df.columns:
        assert (
            train_df["fare_amount"] >= 0
        ).all(), "Found negative fare amounts in training data"

    print("Data Processor verification passed.")
    return train_df, val_df, test_df


def demo_model(train_df, val_df, test_df):
    print("\n=== Demonstrating library.model ===")

    regressor = FareRegressor()

    # Override parameters for speed (very few trees)
    regressor.params["max_iter"] = 5
    regressor.params["verbose"] = 0
    # Re-initialize internal model with new params
    from sklearn.ensemble import HistGradientBoostingRegressor

    regressor.model = HistGradientBoostingRegressor(**regressor.params)

    # Fit
    print("Fitting model (fast mode)...")
    regressor.fit(train_df, val_df)

    # Predict
    print("Predicting on test set...")
    preds = regressor.predict(test_df)

    # Verify Predictions
    assert len(preds) == len(test_df), "Prediction length mismatch"
    assert not np.isnan(preds).any(), "Predictions contain NaNs"

    # Save Submission
    print("Saving submission...")
    regressor.save_submission(test_df, preds)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"

    # Verify Submission Content
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "key" in sub_df.columns and "fare_amount" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print("Model verification passed.")


def demo_trainer():
    print("\n=== Demonstrating library.trainer (End-to-End) ===")

    # Run the full pipeline with reduced complexity
    # We use a slightly different sample size to differentiate from the previous step
    # and max_iter=2 to make it extremely fast
    model = train_model(load_cached_data=False, train_sample_size=500, max_iter=2)

    assert model is not None, "Trainer returned None"
    print("Trainer pipeline verification passed.")


if __name__ == "__main__":
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(42)

    # Clean up working directory to ensure fresh start for demos
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    try:
        # 2. Run Demos
        demo_utils()

        # We pass the dataframes from processor demo to model demo to save time reloading
        train_df, val_df, test_df = demo_data_processor()

        demo_model(train_df, val_df, test_df)

        demo_trainer()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Print traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
