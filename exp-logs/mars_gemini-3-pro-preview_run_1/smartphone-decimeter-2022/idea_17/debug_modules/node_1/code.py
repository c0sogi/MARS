import os
import shutil
import pandas as pd
import numpy as np
import torch
import math

# Import from the provided library
from library.config import Config
from library.utils import fix_seed, WGS84Utils
from library.features import process_dataset, process_drive
from library.dataset import (
    SmartphoneLocationDataset,
    get_train_val_loaders,
    get_test_loader,
)
from library.model import CascadedResUNet
from library.loss import CascadedDeepSupervisionLoss
from library.trainer import Trainer


def run_demo():
    print("--- Starting Demo Script ---")

    # 1. Setup Reproducibility
    fix_seed(42)

    # 2. Setup Temporary Configuration for Demo
    print("\n[Setup] Configuring environment for demo...")
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "train_processed.parquet")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "val_processed.parquet")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "test_processed.parquet")

    # Create cache dir
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE_PATH), exist_ok=True)
    Config.create_directories()

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TRAIN_WINDOW_SIZE = 64  # Smaller window
    Config.TRAIN_WINDOW_STRIDE = 32

    # 3. Create Mini Metadata
    # We need to point to actual files in ./input, but only process a few of them.
    print("\n[Setup] Creating mini metadata files...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Select a single drive for training and validation to ensure data exists
    # We'll split one drive into train/val for this demo to ensure we have data
    # In real training we split by drive_id, here we just want to run code.
    if not orig_train_meta.empty:
        sample_drive = orig_train_meta["drive_id"].unique()[0]
        print(f"  Using drive {sample_drive} for train/val demo.")

        drive_data = orig_train_meta[orig_train_meta["drive_id"] == sample_drive].copy()

        # Split simply by index for demo purposes (first 80% train, last 20% val)
        split_idx = int(len(drive_data) * 0.8)
        mini_train = drive_data.iloc[:split_idx]
        mini_val = drive_data.iloc[split_idx:]

        # Save mini metadata
        mini_train_path = os.path.join(DEMO_DIR, "mini_train_meta.csv")
        mini_val_path = os.path.join(DEMO_DIR, "mini_val_meta.csv")
        mini_train.to_csv(mini_train_path, index=False)
        mini_val.to_csv(mini_val_path, index=False)

        # Update Config to point to mini metadata
        Config.TRAIN_META_PATH = mini_train_path
        Config.VAL_META_PATH = mini_val_path
    else:
        print("  Warning: No training metadata found.")

    # Handle Test Metadata
    if not orig_test_meta.empty:
        sample_test_drive = orig_test_meta["drive_id"].unique()[0]
        print(f"  Using drive {sample_test_drive} for test demo.")
        mini_test = orig_test_meta[
            orig_test_meta["drive_id"] == sample_test_drive
        ].copy()
        mini_test_path = os.path.join(DEMO_DIR, "mini_test_meta.csv")
        mini_test.to_csv(mini_test_path, index=False)
        Config.TEST_META_PATH = mini_test_path
    else:
        print("  Warning: No test metadata found. Skipping test data creation.")
        Config.TEST_META_PATH = os.path.join(DEMO_DIR, "dummy_test.csv")
        pd.DataFrame(columns=orig_test_meta.columns).to_csv(
            Config.TEST_META_PATH, index=False
        )

    # 4. Verify WGS84Utils
    print("\n[Verification] Testing WGS84Utils...")
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0  # 1 degree North

    # Approx 111km per degree latitude
    dist = WGS84Utils.haversine_distance(lat1, lon1, lat2, lon2)
    print(f"  Haversine distance (1 deg Lat): {dist:.2f} meters")
    assert 110000 < dist < 112000, "Haversine distance calculation seems off"

    dEast, dNorth = WGS84Utils.latlon_to_meters_diff(lat1, lon1, lat2, lon2)
    print(f"  Local approximation dNorth: {dNorth:.2f} meters")
    assert 110000 < dNorth < 112000, "dNorth calculation seems off"
    assert abs(dEast) < 1.0, "dEast should be near 0 for North movement"

    new_lat, new_lon = WGS84Utils.meters_to_latlon(lat1, lon1, dEast, dNorth)
    print(f"  Reconstructed Lat/Lon: {new_lat:.6f}, {new_lon:.6f}")
    assert abs(new_lat - lat2) < 1e-5, "Latitude reconstruction failed"
    assert abs(new_lon - lon2) < 1e-5, "Longitude reconstruction failed"
    print("  WGS84Utils verified.")

    # 5. Verify Feature Processing (process_drive)
    print("\n[Verification] Testing Feature Processing...")
    if not orig_train_meta.empty:
        # Use the first row from our mini train metadata
        row = mini_train.iloc[0]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        # We pass the subset of ground truth corresponding to this drive/phone
        gt_subset = mini_train[
            (mini_train["drive_id"] == drive_id)
            & (mini_train["phone_name"] == phone_name)
        ].copy()

        print(f"  Processing drive: {drive_id}, phone: {phone_name}")
        processed_df = process_drive(drive_id, phone_name, gnss_path, gt_subset)

        if processed_df is not None:
            print(f"  Processed DataFrame shape: {processed_df.shape}")
            # Check for required columns
            expected_cols = Config.FEATURE_COLS + Config.TARGET_COLS
            missing_cols = [c for c in expected_cols if c not in processed_df.columns]
            assert (
                not missing_cols
            ), f"Missing columns in processed data: {missing_cols}"
            print("  Feature processing successful.")
        else:
            print(
                "  Feature processing returned None (possibly missing file or empty). Skipping assertion."
            )

    # 6. Verify Model Architecture
    print("\n[Verification] Testing Model Architecture...")
    model = CascadedResUNet()
    # Input shape: (Batch, Channels, Length)
    dummy_input = torch.randn(2, Config.INPUT_DIM, 128)

    # Forward pass
    final_out, aux_outs = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Final output shape: {final_out.shape}")

    assert final_out.shape == (2, Config.OUTPUT_DIM, 128), "Final output shape mismatch"
    assert len(aux_outs) == 2, "Expected 2 auxiliary outputs"
    print("  Model forward pass successful.")

    # 7. Verify Loss Function
    print("\n[Verification] Testing Loss Function...")
    criterion = CascadedDeepSupervisionLoss()
    dummy_targets = torch.randn(2, Config.OUTPUT_DIM, 128)
    dummy_mask = torch.ones(2, 128)

    loss, metrics = criterion((final_out, aux_outs), dummy_targets, dummy_mask)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert "loss_final" in metrics, "Missing loss_final metric"
    print("  Loss calculation successful.")

    # 8. Run Full Training Loop (Trainer)
    print("\n[Execution] Running Trainer (Fit)...")
    trainer = Trainer(run_name="demo_run")

    # We use debug=False here because we manually created small metadata files.
    # If we used debug=True, it might slice our already small data to nothing.
    # However, get_train_val_loaders calls process_dataset which uses the metadata files we set in Config.
    scaler = trainer.fit(debug=False)

    print("  Training loop completed.")

    # 9. Run Prediction
    print("\n[Execution] Running Trainer (Predict)...")
    trainer.predict(scaler)

    # Verify submission file exists
    if os.path.exists(Config.SUBMISSION_OUTPUT_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)
        print(f"  Submission generated with {len(sub_df)} rows.")
        # Check for NaNs in prediction
        nans = sub_df[["LatitudeDegrees", "LongitudeDegrees"]].isnull().sum().sum()
        if nans > 0:
            print(f"  Warning: Submission contains {nans} NaNs.")
        else:
            print("  Submission contains valid predictions.")
    else:
        print("  Error: Submission file not found.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
