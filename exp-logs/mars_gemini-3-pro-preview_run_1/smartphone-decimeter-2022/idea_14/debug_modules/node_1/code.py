import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, WGS84Utils
from library.dataset import load_data, GNSSDataset, gnss_collate_fn
from library.model import ResUNet1D
from library.loss import DeepSupervisionMAELoss
from library.train import Trainer, run_training
from library.inference import generate_submission


def create_mini_metadata():
    """
    Creates smaller metadata files in the working directory to ensure
    the demonstration runs quickly by processing only a few drives.
    """
    print("Creating mini metadata files for demonstration...")

    # Define working directory for demo
    demo_dir = os.path.join("working", "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Train/Val Metadata Subset
    train_meta_path = Config.TRAIN_METADATA_PATH
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(
            f"Original train metadata not found at {train_meta_path}"
        )

    df_train = pd.read_csv(train_meta_path)

    # Select 2 unique drives for training and 1 for validation
    unique_drives = df_train["drive_id"].unique()
    if len(unique_drives) < 2:
        raise ValueError("Not enough drives in training data to create a subset.")

    train_drives = unique_drives[:2]

    # Create mini dataframes
    mini_train_df = df_train[df_train["drive_id"].isin(train_drives)].copy()

    # Split the mini train further into train/val for the demo
    # We'll just take the first drive for train, second for val
    demo_train_df = mini_train_df[mini_train_df["drive_id"] == train_drives[0]].copy()
    demo_val_df = mini_train_df[mini_train_df["drive_id"] == train_drives[1]].copy()

    # Save to working dir
    demo_train_path = os.path.join(demo_dir, "mini_train_meta.csv")
    demo_val_path = os.path.join(demo_dir, "mini_val_meta.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)

    print(f"  Saved mini train metadata: {len(demo_train_df)} rows")
    print(f"  Saved mini val metadata: {len(demo_val_df)} rows")

    # 2. Test Metadata Subset
    test_meta_path = Config.TEST_METADATA_PATH
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Original test metadata not found at {test_meta_path}")

    df_test = pd.read_csv(test_meta_path)
    unique_test_drives = df_test["drive_id"].unique()

    if len(unique_test_drives) > 0:
        test_drive = unique_test_drives[0]
        demo_test_df = df_test[df_test["drive_id"] == test_drive].copy()
    else:
        demo_test_df = df_test.copy()

    demo_test_path = os.path.join(demo_dir, "mini_test_meta.csv")
    demo_test_df.to_csv(demo_test_path, index=False)
    print(f"  Saved mini test metadata: {len(demo_test_df)} rows")

    return demo_train_path, demo_val_path, demo_test_path, demo_dir


def verify_wgs84_utils():
    """
    Verifies the correctness of the coordinate transformation utilities.
    """
    print("\nVerifying WGS84 Utilities...")

    # Test Case: Move 1 degree North and 1 degree East from (0, 0)
    # Note: At equator, 1 deg lat ~ 110.574 km, 1 deg lon ~ 111.320 km
    ref_lat, ref_lon = 0.0, 0.0
    target_lat, target_lon = 1.0, 1.0

    # Forward: Deg -> Meters
    north, east = WGS84Utils.degrees_to_meters(target_lat, target_lon, ref_lat, ref_lon)

    print(f"  (0,0) -> (1,1): North={north:.2f}m, East={east:.2f}m")

    # Expected approx values
    assert north > 110000 and north < 112000, "North calculation seems off"
    assert east > 110000 and east < 112000, "East calculation seems off"

    # Backward: Meters -> Deg
    rec_lat, rec_lon = WGS84Utils.meters_to_degrees(north, east, ref_lat, ref_lon)

    print(f"  Recovered: Lat={rec_lat:.6f}, Lon={rec_lon:.6f}")

    # Check consistency
    assert np.isclose(rec_lat, target_lat), "Latitude reconstruction failed"
    assert np.isclose(rec_lon, target_lon), "Longitude reconstruction failed"

    print("  WGS84Utils verification passed.")


def run_demo():
    # 1. Setup Environment
    set_seed(42)

    # 2. Verify Utilities
    verify_wgs84_utils()

    # 3. Prepare Mini Dataset
    train_meta, val_meta, test_meta, work_dir = create_mini_metadata()

    # 4. Patch Config for Demo
    print("\nConfiguring environment for demo...")
    Config.WORK_DIR = work_dir
    Config.TRAIN_METADATA_PATH = train_meta
    Config.VAL_METADATA_PATH = val_meta
    Config.TEST_METADATA_PATH = test_meta

    # Point caches to the demo directory
    Config.TRAIN_CACHE = os.path.join(work_dir, "cache", "train_processed.parquet")
    Config.VAL_CACHE = os.path.join(work_dir, "cache", "val_processed.parquet")
    Config.TEST_CACHE = os.path.join(work_dir, "cache", "test_processed.parquet")
    Config.MODEL_CHECKPOINT = os.path.join(work_dir, "best_model.pth")

    # Create submission dir in working to avoid cluttering root
    demo_sub_dir = os.path.join("working", "demo_submission")
    os.makedirs(demo_sub_dir, exist_ok=True)
    Config.SUBMISSION_DIR = demo_sub_dir
    Config.SUBMISSION_OUTPUT = os.path.join(demo_sub_dir, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 5. Run Training Pipeline
    print("\n--- Running Training Pipeline ---")
    # load_cached_data=False forces processing of our new mini metadata
    run_training(load_cached_data=False)

    # Verify checkpoint creation
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError("Model checkpoint was not created during training!")
    print("Training successful. Checkpoint created.")

    # 6. Run Inference Pipeline
    print("\n--- Running Inference Pipeline ---")
    generate_submission(load_cached_data=False, batch_size=2)

    # Verify submission creation
    if not os.path.exists(Config.SUBMISSION_OUTPUT):
        raise FileNotFoundError("Submission file was not created!")

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_OUTPUT)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not all(col in sub_df.columns for col in expected_cols):
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {sub_df.columns.tolist()}"
        )

    print("Inference successful.")

    # 7. Model Logic Verification (Deep Supervision)
    print("\n--- Verifying Model Architecture ---")
    model = ResUNet1D()
    # Create dummy input: (Batch, Channels, Length)
    # Config.IN_CHANNELS is derived from INPUT_FEATURES list length
    dummy_input = torch.randn(2, Config.IN_CHANNELS, 128)

    # Training mode: should return 3 outputs
    model.train()
    outputs = model(dummy_input)
    assert (
        isinstance(outputs, tuple) and len(outputs) == 3
    ), "Model did not return 3 outputs in training mode"
    print("Model training forward pass: OK (Deep Supervision active)")

    # Eval mode: should return 1 output
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert torch.is_tensor(output), "Model did not return a single tensor in eval mode"
    assert output.shape == (2, 2, 128), f"Unexpected output shape: {output.shape}"
    print("Model eval forward pass: OK")


if __name__ == "__main__":
    run_demo()
