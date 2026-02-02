import os
import numpy as np
import pandas as pd
import torch
import shutil
import importlib
from library.config import Config
from library import utils, preprocessing, dataset, model, train, predict

# Force reload modules to ensure fixes are picked up in persistent environments (Cite debug_lesson_3)
importlib.reload(model)
importlib.reload(preprocessing)
importlib.reload(dataset)
importlib.reload(train)
importlib.reload(predict)


def main():
    print("Starting Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # Define temporary paths
    DEMO_DIR = "./working/demo"
    os.makedirs(DEMO_DIR, exist_ok=True)

    SUBSET_TRAIN_META = os.path.join(DEMO_DIR, "train_meta_subset.csv")
    SUBSET_VAL_META = os.path.join(DEMO_DIR, "val_meta_subset.csv")
    SUBSET_TEST_META = os.path.join(DEMO_DIR, "test_meta_subset.csv")

    # Override Config paths to use our subsets and a specific cache
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    Config.TRAIN_METADATA_PATH = SUBSET_TRAIN_META
    Config.VAL_METADATA_PATH = SUBSET_VAL_META
    Config.TEST_METADATA_PATH = SUBSET_TEST_META
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16

    print(f"Configured cache directory: {Config.CACHE_DIR}")

    # -------------------------------------------------------------------------
    # 2. Create Data Subsets
    # -------------------------------------------------------------------------
    print("\nCreating data subsets for speed...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/validation_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 1 drive for train, 1 for val, 1 trip for test
    train_drives = orig_train["drive_id"].unique()[:1]
    val_drives = orig_val["drive_id"].unique()[:1]
    test_trips = orig_test["tripId"].unique()[:1]

    df_train_sub = orig_train[orig_train["drive_id"].isin(train_drives)].copy()
    df_val_sub = orig_val[orig_val["drive_id"].isin(val_drives)].copy()
    df_test_sub = orig_test[orig_test["tripId"].isin(test_trips)].copy()

    # Limit rows further to ensure super fast execution
    df_train_sub = df_train_sub.head(200)
    df_val_sub = df_val_sub.head(100)
    df_test_sub = df_test_sub.head(100)

    # Save subsets
    df_train_sub.to_csv(SUBSET_TRAIN_META, index=False)
    df_val_sub.to_csv(SUBSET_VAL_META, index=False)
    df_test_sub.to_csv(SUBSET_TEST_META, index=False)

    print(f"Train subset: {len(df_train_sub)} rows")
    print(f"Val subset: {len(df_val_sub)} rows")
    print(f"Test subset: {len(df_test_sub)} rows")

    # -------------------------------------------------------------------------
    # 3. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\nVerifying Utility Functions...")

    # Test Coordinate Conversion
    lat_ref, lon_ref = 37.0, -122.0
    # Move 1 degree North (approx 111km)
    lat_target, lon_target = 38.0, -122.0

    d_east, d_north = utils.wgs84_to_local_meters(
        lat_ref, lon_ref, lat_target, lon_target
    )

    # 1 deg lat is approx 111,320 meters
    assert np.isclose(
        d_east, 0.0, atol=1e-5
    ), "East distance should be 0 for North movement"
    assert np.isclose(
        d_north, 111320.0, rtol=0.01
    ), f"North distance {d_north} not close to 111320"

    # Test Round Trip
    lat_recon, lon_recon = utils.local_meters_to_wgs84(
        lat_ref, lon_ref, d_east, d_north
    )
    assert np.isclose(
        lat_recon, lat_target, atol=1e-6
    ), "Latitude reconstruction failed"
    assert np.isclose(
        lon_recon, lon_target, atol=1e-6
    ), "Longitude reconstruction failed"

    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Dataset Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\nInitializing Dataset (Train)...")
    # This triggers preprocessing.load_dataset which computes features and saves to cache
    train_ds = dataset.GNSSWindowDataset(mode="train", load_cached_data=False)

    # Check item structure
    x, y = train_ds[0]
    print(f"Sample Input Shape: {x.shape}")  # Should be (WINDOW_SIZE, n_features)
    print(f"Sample Target Shape: {y.shape}")  # Should be (2,) -> [East, North]

    assert x.shape == (
        Config.WINDOW_SIZE,
        len(Config.INPUT_FEATURES),
    ), "Incorrect input shape"
    assert y.shape == (len(Config.TARGET_COLUMNS),), "Incorrect target shape"

    # -------------------------------------------------------------------------
    # 5. Demonstrate Model Instantiation
    # -------------------------------------------------------------------------
    print("\nInstantiating BiLSTM Model...")
    input_dim = x.shape[1]
    output_dim = y.shape[0]

    net = model.BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=32,  # Reduced for demo
        num_layers=1,  # Reduced for demo
        output_dim=output_dim,
    )

    # Forward pass check
    # Add batch dimension: (1, window, features)
    dummy_input = x.unsqueeze(0)
    dummy_out = net(dummy_input)

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (1, output_dim), "Model output shape mismatch"

    # -------------------------------------------------------------------------
    # 6. Run Full Training Pipeline
    # -------------------------------------------------------------------------
    print("\nRunning Training Pipeline...")
    # run_training handles dataloaders, training loop, and submission generation
    # We use load_cached_data=True because we just initialized the dataset above,
    # so the parquet is already in our demo cache.
    train.run_training(
        load_cached_data=True, batch_size=16, epochs=1, learning_rate=0.01
    )

    model_path = os.path.join(Config.CACHE_DIR, "bilstm_model.pth")
    if os.path.exists(model_path):
        print(f"Model successfully saved to {model_path}")
    else:
        raise FileNotFoundError("Model file was not created!")

    # -------------------------------------------------------------------------
    # 7. Run Inference Pipeline
    # -------------------------------------------------------------------------
    print("\nRunning Inference Pipeline...")
    # This will process the test subset we created
    predict.run_inference(
        batch_size=16,
        load_cached_data=False,  # Force processing of test data
        model_path=model_path,
    )

    # -------------------------------------------------------------------------
    # 8. Verify Submission
    # -------------------------------------------------------------------------
    print("\nVerifying Submission...")
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission loaded. Shape: {sub_df.shape}")
        print(sub_df.head())

        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            col in sub_df.columns for col in expected_cols
        ), "Missing columns in submission"
        assert len(sub_df) == len(
            df_test_sub
        ), f"Submission length mismatch. Expected {len(df_test_sub)}, got {len(sub_df)}"
        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
