import sys
import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, geodetic_to_enu, enu_to_geodetic, haversine_distance
from library.model import TransUNet1D
from library.data_loader import get_dataloaders
from library.trainer import Trainer, generate_submission


def main():
    print("=== Starting Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Safety
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2  # Process only 2 drives for speed
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead

    # Use a specific directory for this demo to avoid clutter
    Config.WORKING_DIR = "./working/demo_run/cache"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"

    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Dir: {Config.WORKING_DIR}")
    print(f"    Submission Dir: {Config.SUBMISSION_DIR}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")
    lat0, lon0 = 37.42, -122.08  # Approx Google HQ

    # Test Identity
    e, n = geodetic_to_enu(lat0, lon0, lat0, lon0)
    assert np.isclose(e, 0) and np.isclose(
        n, 0
    ), "geodetic_to_enu identity check failed"

    lat_rec, lon_rec = enu_to_geodetic(0, 0, lat0, lon0)
    assert np.isclose(lat_rec, lat0) and np.isclose(
        lon_rec, lon0
    ), "enu_to_geodetic identity check failed"

    # Test Distance (1 degree lat is approx 111km)
    dist = haversine_distance(lat0, lon0, lat0 + 1.0, lon0)
    print(f"    Distance for 1 deg lat: {dist:.2f} meters")
    assert 110000 < dist < 112000, "Haversine distance calculation seems incorrect"
    print("    Utils verified successfully.")

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("\n[3] Loading Data (Preprocessing subset)...")
    # load_cached_data=False forces preprocessing of the debug subset
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    if len(train_loader) > 0:
        # Inspect one batch
        features, targets = next(iter(train_loader))
        print(f"    Feature Batch Shape: {features.shape} (B, C, L)")
        print(f"    Target Batch Shape:  {targets.shape} (B, 2, L)")

        # Assertions
        assert features.dim() == 3, "Features should be 3D tensor"
        assert targets.dim() == 3, "Targets should be 3D tensor"
        assert (
            features.shape[1] == Config.INPUT_DIM
        ), f"Expected {Config.INPUT_DIM} input channels"
        assert (
            targets.shape[1] == Config.OUTPUT_DIM
        ), f"Expected {Config.OUTPUT_DIM} output channels"
        assert (
            features.shape[2] == Config.WINDOW_SIZE
        ), f"Expected window size {Config.WINDOW_SIZE}"
    else:
        print(
            "    Warning: Train loader is empty. Check if input data exists for debug samples."
        )

    # ---------------------------------------------------------
    # 4. Model Instantiation
    # ---------------------------------------------------------
    print("\n[4] Instantiating Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = TransUNet1D().to(device)

    # Test Forward Pass with Dummy Data
    dummy_input = torch.randn(2, Config.INPUT_DIM, Config.WINDOW_SIZE).to(device)
    try:
        output = model(dummy_input)
        print(f"    Model Output Shape: {output.shape}")
        assert output.shape == (
            2,
            Config.OUTPUT_DIM,
            Config.WINDOW_SIZE,
        ), "Model output shape mismatch"
        print("    Model forward pass successful.")
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Initialize Trainer
    trainer = Trainer(model, device, optimizer)

    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")

    # Run fit (1 epoch as per config)
    if len(train_loader) > 0:
        trainer.fit(
            train_loader,
            val_loader,
            num_epochs=Config.NUM_EPOCHS,
            checkpoint_path=checkpoint_path,
        )

        if os.path.exists(checkpoint_path):
            print("    Checkpoint saved successfully.")
        else:
            # If validation loss didn't improve (unlikely in epoch 1 vs inf), it might not save.
            # But Trainer saves if val_loss < inf.
            print("    Warning: Checkpoint was not found (check trainer logic).")
    else:
        print("    Skipping training due to empty dataloader.")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")

    # We need to ensure test_loader is valid
    if len(test_loader) > 0:
        try:
            generate_submission(trainer, test_loader)

            sub_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
            if os.path.exists(sub_file):
                df_sub = pd.read_csv(sub_file)
                print(f"    Submission File Generated: {sub_file}")
                print(f"    Rows: {len(df_sub)}")
                print(f"    Columns: {df_sub.columns.tolist()}")

                required_cols = [
                    "tripId",
                    "UnixTimeMillis",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                ]
                for col in required_cols:
                    assert col in df_sub.columns, f"Missing column {col} in submission"
            else:
                raise FileNotFoundError("Submission file was not created.")

        except Exception as e:
            print(f"    Inference failed: {e}")
            # Don't fail the whole script for inference if training worked, just report
            pass
    else:
        print("    Skipping inference due to empty test loader.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
