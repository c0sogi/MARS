import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
import torch

# Import provided library modules
from library import config
from library import utils
from library import data_processor
from library import spatial_encoder
from library import model

# --- Configuration for Demo ---
# We override some config defaults to ensure the demo runs fast (within minutes)
# while still exercising all code paths.
DEMO_TRAIN_FRAC = 0.005  # Use ~0.5% of training data (~220k rows)
DEMO_VAL_FRAC = 0.01  # Use ~1% of validation data (~110k rows)
DEMO_N_CLUSTERS = 50  # Reduced clusters for speed
DEMO_N_FOLDS = 3  # Reduced folds for OOF
DEMO_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 100,  # Few trees for fast demo
    "learning_rate": 0.1,  # Higher LR for fast convergence in demo
    "num_leaves": 31,  # Simpler trees
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 4,  # Restrict threads
    "device": "cpu",  # Use CPU for small demo data to avoid overhead
    "verbose": -1,
    "random_state": config.RANDOM_SEED,
}


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    print("--- Setting up environment ---")
    warnings.filterwarnings("ignore")
    set_seeds(config.RANDOM_SEED)
    config.setup_directories()

    # 2. Data Processing
    print("\n--- Step 1: Data Loading & Processing ---")
    dm = data_processor.TaxiDataManager()

    # Load processed data (with sampling for speed)
    print(f"Loading training data (sample_frac={DEMO_TRAIN_FRAC})...")
    train_df = dm.get_processed_data("train", sample_frac=DEMO_TRAIN_FRAC)

    print(f"Loading validation data (sample_frac={DEMO_VAL_FRAC})...")
    val_df = dm.get_processed_data("val", sample_frac=DEMO_VAL_FRAC)

    print("Loading test data...")
    test_df = dm.get_processed_data("test")
    test_df = test_df.reset_index(drop=True)

    # Validation assertions
    assert not train_df.empty, "Training dataframe is empty"
    assert "dist_haversine" in train_df.columns, "Geometric features missing"
    assert "hour" in train_df.columns, "Temporal features missing"
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # 3. Spatial Encoding
    print("\n--- Step 2: Spatial Feature Engineering ---")
    # Initialize encoder with demo settings
    encoder = spatial_encoder.SpatialRouteEncoder(
        n_clusters=DEMO_N_CLUSTERS,
        n_folds=DEMO_N_FOLDS,
        cache_dir=os.path.join(config.CACHE_DIR, "demo_spatial"),
    )

    print("Fitting spatial clusters...")
    encoder.fit_clusters(train_df, load_cached_data=False)

    print("Generating OOF target encoding for training set...")
    # Get the feature column (returns a DataFrame with index matched to input)
    train_spatial = encoder.get_oof_target_encoding(train_df, load_cached_data=False)
    train_df = train_df.join(train_spatial)

    print("Fitting global route map...")
    encoder.fit_global_map(train_df, load_cached_data=False)

    print("Applying global encoding to validation and test sets...")
    val_spatial = encoder.get_global_target_encoding(
        val_df, "val", load_cached_data=False
    )
    val_df = val_df.join(val_spatial)

    test_spatial = encoder.get_global_target_encoding(
        test_df, "test", load_cached_data=False
    )
    test_df = pd.concat([test_df, test_spatial], axis=1)

    # Validation assertions
    assert "mean_fare_route" in train_df.columns, "Spatial feature missing in train"
    assert "mean_fare_route" in test_df.columns, "Spatial feature missing in test"
    assert (
        train_df["mean_fare_route"].isna().sum() == 0
    ), "NaNs found in spatial features (Train)"

    # 4. Model Training
    print("\n--- Step 3: Model Training ---")
    fare_model = model.FareModel(params=DEMO_LGBM_PARAMS)

    # Train
    # We ignore 'key' and 'pickup_datetime' as they are not features
    fare_model.train(train_df, val_df, target_col="fare_amount")

    # Check importance
    importance = fare_model.get_feature_importance()
    print("\nTop 5 Features:")
    print(importance.head(5).to_string(index=False))

    # Save model
    model_path = os.path.join(config.CACHE_DIR, "demo_model.txt")
    fare_model.save(model_path)

    # 5. Prediction & Submission
    print("\n--- Step 4: Prediction & Submission ---")

    # Verify features
    missing_cols = [c for c in fare_model.feature_names if c not in test_df.columns]
    if missing_cols:
        raise KeyError(f"test_df is missing features: {missing_cols}")

    # Predict
    predictions = fare_model.predict(test_df)

    # Create submission dataframe
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

    # Ensure no negative fares (simple post-processing)
    submission["fare_amount"] = submission["fare_amount"].clip(lower=0)

    # Save submission
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

    # Final Validation
    print("\n--- Final Validation ---")
    assert len(submission) == len(test_df), "Submission row count mismatch"
    assert submission["fare_amount"].isna().sum() == 0, "NaNs in prediction"
    assert (submission["fare_amount"] >= 0).all(), "Negative fare predictions found"

    # Print sample
    print("Sample Submission Rows:")
    print(submission.head())

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
