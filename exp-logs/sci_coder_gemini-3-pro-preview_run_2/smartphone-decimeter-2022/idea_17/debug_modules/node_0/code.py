import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
import library.config as config
import library.utils as utils
import library.preprocessing as preprocessing
import library.dataset as dataset
import library.model as model
import library.trainer as trainer
import library.inference as inference


def main():
    print("Starting demonstration...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration for demo...")

    # Create a working directory for demo files
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Monkey-patch config to use demo directory and settings
    config.CACHE_DIR = demo_dir
    config.SUBMISSION_DIR = demo_dir
    config.EPOCHS = 1
    config.BATCH_SIZE = 16
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Update cache file paths in config to point to new CACHE_DIR
    config.TRAIN_CACHE_FILES = {
        "X_kin": os.path.join(config.CACHE_DIR, "train_X_kinematic.npy"),
        "X_sky": os.path.join(config.CACHE_DIR, "train_X_sky.npy"),
        "y": os.path.join(config.CACHE_DIR, "train_y.npy"),
        "meta": os.path.join(config.CACHE_DIR, "train_meta.parquet"),
    }
    config.VAL_CACHE_FILES = {
        "X_kin": os.path.join(config.CACHE_DIR, "val_X_kinematic.npy"),
        "X_sky": os.path.join(config.CACHE_DIR, "val_X_sky.npy"),
        "y": os.path.join(config.CACHE_DIR, "val_y.npy"),
        "meta": os.path.join(config.CACHE_DIR, "val_meta.parquet"),
    }
    config.TEST_CACHE_FILES = {
        "X_kin": os.path.join(config.CACHE_DIR, "test_X_kinematic.npy"),
        "X_sky": os.path.join(config.CACHE_DIR, "test_X_sky.npy"),
        "meta": os.path.join(config.CACHE_DIR, "test_meta.parquet"),
    }
    config.SCALER_PATH = os.path.join(config.CACHE_DIR, "scaler.joblib")
    config.MODEL_PATH = os.path.join(config.CACHE_DIR, "best_model.pth")

    # ---------------------------------------------------------
    # 2. Prepare Subset Metadata
    # ---------------------------------------------------------
    print("\n[2] Preparing subset metadata...")

    # Load original metadata
    train_meta_orig = pd.read_csv("./metadata/train_metadata.csv")
    val_meta_orig = pd.read_csv("./metadata/validation_metadata.csv")
    test_meta_orig = pd.read_csv("./metadata/test_metadata.csv")

    # Select one trip for each split to keep it fast
    train_trip = train_meta_orig["tripId"].unique()[0]
    val_trip = val_meta_orig["tripId"].unique()[0]
    test_trip = test_meta_orig["tripId"].unique()[0]

    print(f"  Selected Train Trip: {train_trip}")
    print(f"  Selected Val Trip: {val_trip}")
    print(f"  Selected Test Trip: {test_trip}")

    train_subset = train_meta_orig[train_meta_orig["tripId"] == train_trip].copy()
    val_subset = val_meta_orig[val_meta_orig["tripId"] == val_trip].copy()
    test_subset = test_meta_orig[test_meta_orig["tripId"] == test_trip].copy()

    # Save subsets to demo directory
    demo_train_path = os.path.join(demo_dir, "train_metadata_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_metadata_subset.csv")
    demo_test_path = os.path.join(demo_dir, "test_metadata_subset.csv")

    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)
    test_subset.to_csv(demo_test_path, index=False)

    # Patch config paths to point to subsets
    config.TRAIN_METADATA_PATH = demo_train_path
    config.VAL_METADATA_PATH = demo_val_path
    config.TEST_METADATA_PATH = demo_test_path

    # ---------------------------------------------------------
    # 3. Verify Utils
    # ---------------------------------------------------------
    print("\n[3] Verifying utility functions...")

    # Test ECEF to LLA
    # Known point: Approx Googleplex (37.422, -122.084, 0)
    # Converted to ECEF (approx): -2698600, -4296600, 3854800
    x, y, z = -2698600.0, -4296600.0, 3854800.0
    lat, lon, alt = utils.ecef_to_lla(x, y, z)
    print(f"  ECEF({x}, {y}, {z}) -> LLA({lat:.4f}, {lon:.4f}, {alt:.4f})")

    # Basic sanity check range
    assert 37.0 < lat < 38.0, "Latitude calculation seems off"
    assert -123.0 < lon < -121.0, "Longitude calculation seems off"

    # Test Haversine
    d = utils.haversine_distance(37.0, -122.0, 37.001, -122.0)  # ~111 meters
    print(f"  Haversine distance (0.001 deg lat): {d:.4f} meters")
    assert 110.0 < d < 112.0, "Haversine distance calculation seems off"

    # ---------------------------------------------------------
    # 4. Verify Preprocessing & Data Loading
    # ---------------------------------------------------------
    print("\n[4] Verifying preprocessing pipeline...")

    # This will process the subset metadata and save to cache
    # We set load_cached_data=False to force processing
    (train_data, val_data, test_data) = preprocessing.load_data(load_cached_data=False)

    train_X_kin, train_X_sky, train_y, train_meta = train_data

    print(f"  Train Kinematic Shape: {train_X_kin.shape}")  # (N, Window, Feats)
    print(f"  Train Sky Shape: {train_X_sky.shape}")  # (N, Feats)
    print(f"  Train Target Shape: {train_y.shape}")  # (N, 2)

    # Assertions
    assert len(train_X_kin) == len(train_y), "Feature and target lengths mismatch"
    assert (
        train_X_kin.shape[1] == config.WINDOW_SIZE
    ), f"Window size mismatch. Expected {config.WINDOW_SIZE}, got {train_X_kin.shape[1]}"
    assert train_X_kin.shape[2] == len(
        config.KINEMATIC_FEATURES
    ), "Kinematic feature count mismatch"
    assert (
        train_X_sky.shape[1] == 6
    ), "Sky feature count mismatch (expected 6)"  # 3 features * (mean, std)

    # ---------------------------------------------------------
    # 5. Verify Dataset & DataLoader
    # ---------------------------------------------------------
    print("\n[5] Verifying Dataset and DataLoader...")

    # Get dataloaders (this will also fit and save scalers)
    train_loader, val_loader, test_loader, _ = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=0,
        load_cached_data=True,  # Load the cache we just generated
    )

    # Check one batch
    kin_batch, sky_batch, y_batch = next(iter(train_loader))
    print(
        f"  Batch Kinematic Shape: {kin_batch.shape}"
    )  # (Batch, Feats, Window) due to transpose in Dataset
    print(f"  Batch Sky Shape: {sky_batch.shape}")
    print(f"  Batch Target Shape: {y_batch.shape}")

    # Verify transpose for 1D CNN (Batch, Channels, Length)
    assert kin_batch.shape[1] == len(
        config.KINEMATIC_FEATURES
    ), "Dataset did not transpose channels correctly"
    assert kin_batch.shape[2] == config.WINDOW_SIZE, "Dataset window dimension mismatch"

    # ---------------------------------------------------------
    # 6. Verify Model
    # ---------------------------------------------------------
    print("\n[6] Verifying Model Architecture...")

    net = model.SCRCNN()
    # Move to CPU for verification
    net.to("cpu")

    # Forward pass with the batch we fetched
    output = net(kin_batch, sky_batch)
    print(f"  Model Output Shape: {output.shape}")

    assert output.shape == (config.BATCH_SIZE, 2), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    # ---------------------------------------------------------
    # 7. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[7] Verifying Training Loop...")

    # Train for 1 epoch using the subset
    trained_model = trainer.train_model(load_cached_data=True)

    assert os.path.exists(config.MODEL_PATH), "Model checkpoint was not saved"
    print("  Training completed and model saved.")

    # ---------------------------------------------------------
    # 8. Verify Inference
    # ---------------------------------------------------------
    print("\n[8] Verifying Inference Pipeline...")

    inference.generate_submission(load_cached_data=True)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission Rows: {len(df_sub)}")
    print("  Sample Submission:")
    print(df_sub.head())

    # Check for NaNs in submission
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    # Check columns
    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
