import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
from library.config import Config
from library.utils import (
    lla_to_ecef,
    ecef_to_lla,
    ecef_to_enu,
    enu_to_ecef,
    calc_haversine_error,
)
from library.preprocessing import GNSSPreprocessor
from library.dataset import GNSSSequenceDataset
from library.model import StratifiedAttentionResUNet
from library.loss import DecimatedDeepSupervisionLoss
from library.trainer import Trainer


def setup_demo_environment():
    """
    Sets up a demo environment by overriding Config paths and parameters
    to run quickly on a small subset of data.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Training Hyperparameters for Speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.TRAIN_SEQUENCE_LENGTH = 32  # Short sequence for demo
    Config.INFERENCE_OVERLAP = 0
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.DEBUG = True

    print(f"Working Directory: {Config.WORKING_DIR}")


def create_mini_metadata():
    """
    Creates mini metadata CSVs by sampling a few drives from the original metadata.
    This ensures the pipeline runs on real data but finishes quickly.
    """
    print("\n--- Creating Mini Metadata ---")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 1 drive for train, 1 for val, 1 for test
    train_drives = orig_train_meta["drive_id"].unique()[:1]
    val_drives = orig_val_meta["drive_id"].unique()[:1]
    test_drives = orig_test_meta["drive_id"].unique()[:1]

    mini_train = orig_train_meta[orig_train_meta["drive_id"].isin(train_drives)].copy()
    mini_val = orig_val_meta[orig_val_meta["drive_id"].isin(val_drives)].copy()
    mini_test = orig_test_meta[orig_test_meta["drive_id"].isin(test_drives)].copy()

    # Limit rows further to ensure very fast processing
    mini_train = mini_train.head(500)
    mini_val = mini_val.head(200)
    mini_test = mini_test.head(200)

    # Save mini metadata
    train_path = os.path.join(Config.WORKING_DIR, "mini_train_meta.csv")
    val_path = os.path.join(Config.WORKING_DIR, "mini_val_meta.csv")
    test_path = os.path.join(Config.WORKING_DIR, "mini_test_meta.csv")

    mini_train.to_csv(train_path, index=False)
    mini_val.to_csv(val_path, index=False)
    mini_test.to_csv(test_path, index=False)

    # Update Config to point to these new files
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    print(f"Mini Train: {len(mini_train)} rows")
    print(f"Mini Val:   {len(mini_val)} rows")
    print(f"Mini Test:  {len(mini_test)} rows")


def verify_utils():
    """
    Verifies the correctness of utility functions.
    """
    print("\n--- Verifying Utils ---")

    # Test LLA <-> ECEF
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = lla_to_ecef(lat, lon, alt)
    lat_rec, lon_rec, alt_rec = ecef_to_lla(x, y, z)

    print(f"Original LLA: {lat}, {lon}, {alt}")
    print(f"Recovered LLA: {lat_rec:.6f}, {lon_rec:.6f}, {alt_rec:.6f}")

    assert np.isclose(lat, lat_rec, atol=1e-5), "Latitude mismatch"
    assert np.isclose(lon, lon_rec, atol=1e-5), "Longitude mismatch"
    assert np.isclose(alt, alt_rec, atol=1e-3), "Altitude mismatch"

    # Test Haversine
    dist = calc_haversine_error(lat, lon, lat + 0.0001, lon)
    print(f"Haversine distance (small lat shift): {dist:.4f} m")
    assert dist > 0, "Distance should be positive"

    print("Utils verification passed.")


def verify_preprocessing():
    """
    Runs the preprocessor on the mini training set.
    """
    print("\n--- Verifying Preprocessing ---")

    preprocessor = GNSSPreprocessor()

    # Process Train
    df_train = preprocessor.process_data(split="train", load_cached_data=False)

    assert not df_train.empty, "Processed train DataFrame is empty"
    assert "target_east" in df_train.columns, "Target columns missing in train"

    # Check for stratified features
    expected_col_part = "global_Cn0DbHz_mean"
    assert any(
        expected_col_part in c for c in df_train.columns
    ), f"Feature {expected_col_part} missing"

    print(f"Processed Train Shape: {df_train.shape}")
    print("Preprocessing verification passed.")


def verify_dataset_and_model():
    """
    Instantiates the dataset and model, runs a forward pass, and checks loss.
    """
    print("\n--- Verifying Dataset and Model ---")

    # 1. Dataset
    # Note: Preprocessing was run in the previous step, so cache exists.
    train_dataset = GNSSSequenceDataset(split="train", load_cached_data=True)

    print(f"Dataset length (sequences): {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty. Check sequence generation logic.")

    X, targets, metadata = train_dataset[0]

    print(f"Input Shape: {X.shape}")  # Should be (C, L)
    print(f"Target Keys: {list(targets.keys())}")

    # Check input channels match Config
    assert (
        X.shape[0] == Config.IN_CHANNELS
    ), f"Input channels {X.shape[0]} != Config {Config.IN_CHANNELS}"
    assert (
        X.shape[1] == Config.TRAIN_SEQUENCE_LENGTH
    ), f"Seq len {X.shape[1]} != Config {Config.TRAIN_SEQUENCE_LENGTH}"

    # 2. Model
    model = StratifiedAttentionResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        filters=[16, 32, 64, 128],  # Reduced filters for demo speed
        deep_supervision=True,
    ).to(Config.DEVICE)

    # Add batch dimension
    # Cite debug_lesson_17: Manually construct batch size > 1 to satisfy BatchNorm requirements
    X_batch = X.unsqueeze(0).repeat(2, 1, 1).to(Config.DEVICE)

    # Forward Pass
    outputs = model(X_batch)

    print(f"Model produced {len(outputs)} outputs (Deep Supervision enabled)")
    # Output 0 is full res
    print(f"Output[0] shape: {outputs[0].shape}")

    assert outputs[0].shape == (2, Config.OUT_CHANNELS, Config.TRAIN_SEQUENCE_LENGTH)

    # 3. Loss
    criterion = DecimatedDeepSupervisionLoss()

    # Move targets to device
    targets_dev = {
        k: v.unsqueeze(0).repeat(2, 1).to(Config.DEVICE) for k, v in targets.items()
    }

    loss = criterion(outputs, targets_dev)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"

    print("Dataset and Model verification passed.")


def run_trainer_demo():
    """
    Runs the Trainer for one epoch and generates a submission.
    """
    print("\n--- Running Trainer Demo ---")

    # Initialize Trainer
    # We use load_cached_data=True because we processed data in verify_preprocessing
    trainer = Trainer(load_cached_data=True)

    # Run Training
    trainer.run()

    # Generate Submission
    trainer.generate_submission()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated with {len(df_sub)} rows.")
        print(df_sub.head())
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Prepare Data
    create_mini_metadata()

    # 3. Verify Components
    verify_utils()
    verify_preprocessing()
    verify_dataset_and_model()

    # 4. Run Full Pipeline
    run_trainer_demo()

    print("\nDemo completed successfully.")
