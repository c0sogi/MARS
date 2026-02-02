import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Import library classes
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.knowledge_base import KnowledgeBase
from library.model import TaxiFareModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Taxi Fare Prediction Pipeline Demo ===")

    # 1. Configuration
    # Initialize Config with debug=True to use smaller dataset subsets and fewer trees
    # for a fast demonstration.
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True)
    config.setup_dirs()
    config.set_seed()

    # Verify debug settings
    assert (
        config.BACKGROUND_SIZE == 1_000_000
    ), "Debug mode should set BACKGROUND_SIZE to 1M"
    assert (
        config.XGB_PARAMS["n_estimators"] == 100
    ), "Debug mode should reduce n_estimators"
    print("Configuration initialized successfully (Debug Mode).")

    # 2. Data Loading
    print("\n[2] Loading and Partitioning Data...")
    data_loader = DataLoader(config)

    # We force load_cached_data=False to demonstrate the processing logic from scratch
    bg_df, fg_df, val_df, test_df = data_loader.get_data(load_cached_data=False)

    # Verification
    print(f"Background set shape: {bg_df.shape}")
    print(f"Foreground set shape: {fg_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    assert len(bg_df) > 0, "Background set is empty"
    assert len(fg_df) > 0, "Foreground set is empty"
    assert len(val_df) > 0, "Validation set is empty"
    assert len(test_df) > 0, "Test set is empty"
    # Check strict filtering on background (no negative fares)
    assert (
        bg_df["fare_amount"].min() >= config.BG_MIN_FARE
    ), "Background set contains invalid fares"
    print("Data loading and partitioning verified.")

    # 3. Feature Engineering
    print("\n[3] Feature Engineering...")
    fe = FeatureEngineer(config)

    # Process Background (Minimal features: keys + basic dist)
    # Using a unique cache key for this run to avoid conflicts
    bg_df = fe.process(
        bg_df, cache_key="demo_bg", load_cached_data=False, is_background=True
    )

    # Process Foreground, Val, Test (Full features)
    fg_df = fe.process(
        fg_df, cache_key="demo_fg", load_cached_data=False, is_background=False
    )
    val_df = fe.process(
        val_df, cache_key="demo_val", load_cached_data=False, is_background=False
    )
    test_df = fe.process(
        test_df, cache_key="demo_test", load_cached_data=False, is_background=False
    )

    # Verification
    # Background should have spatial keys
    assert "key_fine" in bg_df.columns, "key_fine missing from background"
    # Foreground should have advanced features
    assert "bearing" in fg_df.columns, "bearing missing from foreground"
    assert (
        "pickup_rot_x" in fg_df.columns
    ), "Rotated coordinates missing from foreground"
    print("Feature engineering verified.")

    # 4. Knowledge Base (Priors)
    print("\n[4] Building Knowledge Base and Enriching Data...")
    kb = KnowledgeBase(config)

    # Compute Priors from Background
    priors = kb.compute_priors(bg_df, load_cached_data=False)

    assert "fine" in priors, "Fine-grained priors missing"
    assert "global" in priors, "Global stats missing"
    print(f"Global Mean Fare: {priors['global']['mean_fare']:.2f}")

    # Enrich Datasets
    fg_df = kb.enrich_dataset(fg_df, priors)
    val_df = kb.enrich_dataset(val_df, priors)
    test_df = kb.enrich_dataset(test_df, priors)

    # Verification
    # Check if 'smart_fare' (the fallback logic feature) exists
    assert "smart_fare" in fg_df.columns, "smart_fare missing from enriched foreground"
    assert "smart_fare" in test_df.columns, "smart_fare missing from enriched test"
    # Check if a specific prior column exists
    assert "fine_fare" in fg_df.columns, "fine_fare missing from enriched foreground"
    print("Knowledge Base enrichment verified.")

    # 5. Model Training
    print("\n[5] Training XGBoost Model...")
    model = TaxiFareModel(config)

    # Train
    model.train(fg_df, val_df)

    # Verification
    model_file = config.get_cache_path("xgb_model.json")
    assert os.path.exists(model_file), f"Model file not found at {model_file}"
    print("Model training completed and saved.")

    # 6. Prediction and Submission
    print("\n[6] Generating Predictions and Submission...")

    # Predict
    predictions = model.predict(test_df)

    # Verify predictions
    assert len(predictions) == len(test_df), "Prediction count mismatch"
    assert np.all(
        predictions >= 2.50
    ), "Predictions contain values below min fare ($2.50)"
    assert not np.isnan(predictions).any(), "Predictions contain NaNs"

    # Generate Submission
    model.generate_submission(test_df, predictions)

    # Verify Submission File
    submission_path = config.FINAL_SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (
        len(test_df),
        2,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"

    print(f"Submission generated successfully at {submission_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
