import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# Mock tqdm to prevent progress bars
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = noop_tqdm

# Import library modules
from library.config import Config
from library.utils import (
    haversine_distance,
    ecef_to_lla,
    latlon_to_meters,
    meters_to_latlon,
    seed_everything,
)
from library.model import SkyContextualizedCNN
from library.trainer import train_model
from library.inference import generate_submission


def run_demo():
    print("Initializing Demo...")

    # --------------------------------------------------------------------------
    # 1. Setup Directories
    # --------------------------------------------------------------------------
    demo_dir = "./working/demo"
    cache_dir = os.path.join(demo_dir, "cache")
    sub_dir = os.path.join(demo_dir, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Create Subset Metadata (Real Data)
    # --------------------------------------------------------------------------
    print("Creating subset metadata...")
    # Load original metadata
    orig_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select 1 trip for each split to keep it fast
    train_trip = orig_train_meta["tripId"].unique()[0]
    val_trip = orig_val_meta["tripId"].unique()[0]
    test_trip = orig_test_meta["tripId"].unique()[0]

    # Filter and limit rows (enough for a few windows)
    # Window size is 15, so we need at least that many rows per segment
    demo_train_meta = (
        orig_train_meta[orig_train_meta["tripId"] == train_trip].head(200).copy()
    )
    demo_val_meta = orig_val_meta[orig_val_meta["tripId"] == val_trip].head(100).copy()
    demo_test_meta = (
        orig_test_meta[orig_test_meta["tripId"] == test_trip].head(100).copy()
    )

    # Save subset metadata
    demo_train_path = os.path.join(demo_dir, "train_meta_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_meta_subset.csv")
    demo_test_path = os.path.join(demo_dir, "test_meta_subset.csv")

    demo_train_meta.to_csv(demo_train_path, index=False)
    demo_val_meta.to_csv(demo_val_path, index=False)
    demo_test_meta.to_csv(demo_test_path, index=False)

    print(
        f"Created subset metadata: Train={len(demo_train_meta)}, Val={len(demo_val_meta)}, Test={len(demo_test_meta)}"
    )

    # --------------------------------------------------------------------------
    # 3. Override Config for Demo
    # --------------------------------------------------------------------------
    print("Overriding Config parameters...")
    Config.WORKING_DIR = cache_dir
    Config.SUBMISSION_DIR = sub_dir

    # Point to subset metadata
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    # Point to demo cache files
    Config.TRAIN_CACHE_PATH = os.path.join(cache_dir, "train_data.parquet")
    Config.VAL_CACHE_PATH = os.path.join(cache_dir, "val_data.parquet")
    Config.TEST_CACHE_PATH = os.path.join(cache_dir, "test_data.parquet")

    # Artifacts
    Config.SCALER_PATH = os.path.join(cache_dir, "scaler_stats.json")
    Config.MODEL_PATH = os.path.join(cache_dir, "bilstm_model.pth")
    Config.SUBMISSION_PATH = os.path.join(sub_dir, "submission.csv")

    # Training Hyperparameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # --------------------------------------------------------------------------
    # 4. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("Verifying utility functions...")

    # Test Haversine: Distance between (0,0) and (0,1) deg is approx 111km
    dist = haversine_distance(0, 0, 0, 1)
    assert 111000 < dist < 112000, f"Haversine calculation seems off: {dist}"

    # Test ECEF to LLA: Earth radius at equator
    # X=6378137, Y=0, Z=0 -> Lat=0, Lon=0, Alt=0
    lat, lon, alt = ecef_to_lla(6378137.0, 0.0, 0.0)
    assert (
        abs(lat) < 1e-5 and abs(lon) < 1e-5 and abs(alt) < 1e-3
    ), f"ECEF to LLA incorrect: {lat}, {lon}, {alt}"

    # Test LatLon to Meters and back
    lat_base, lon_base = 37.0, -122.0
    d_east, d_north = latlon_to_meters(
        lat_base, lon_base, lat_base + 0.01, lon_base + 0.01
    )
    lat_new, lon_new = meters_to_latlon(lat_base, lon_base, d_east, d_north)

    assert np.isclose(lat_new, lat_base + 0.01), "Lat/Lon conversion failed (Lat)"
    assert np.isclose(lon_new, lon_base + 0.01), "Lat/Lon conversion failed (Lon)"

    print("Utils verified successfully.")

    # --------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("Verifying model architecture...")
    model = SkyContextualizedCNN()

    # Create dummy inputs
    # Trajectory: (Batch, Channels, Window) -> DataLoader provides (B, C, W)
    # Sky: (Batch, Context_Dim)
    B = 4
    W = Config.WINDOW_SIZE
    C_traj = Config.TRAJECTORY_CHANNELS
    C_sky = Config.CONTEXT_INPUT_DIM

    dummy_traj = torch.randn(B, C_traj, W)
    dummy_sky = torch.randn(B, C_sky)

    output = model(dummy_traj, dummy_sky)

    # Output should be (B, 2) -> d_east, d_north
    assert output.shape == (
        B,
        2,
    ), f"Model output shape mismatch. Expected {(B, 2)}, got {output.shape}"
    print("Model architecture verified.")

    # --------------------------------------------------------------------------
    # 6. Run Training Pipeline
    # --------------------------------------------------------------------------
    print("\nStarting Training Pipeline...")
    # This processes the data, caches it, and runs 1 epoch of training
    train_model(load_cached_data=False)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model file was not created after training.")
    print("Training pipeline completed.")

    # --------------------------------------------------------------------------
    # 7. Run Inference Pipeline
    # --------------------------------------------------------------------------
    print("\nStarting Inference Pipeline...")
    # This generates the submission file using the trained model
    generate_submission(load_cached_data=False)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check if rows match metadata
    assert len(sub_df) == len(
        demo_test_meta
    ), f"Submission row count {len(sub_df)} does not match test metadata {len(demo_test_meta)}"

    # Check columns
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    print("Inference pipeline completed.")
    print("\nDemo execution successful!")


if __name__ == "__main__":
    run_demo()
