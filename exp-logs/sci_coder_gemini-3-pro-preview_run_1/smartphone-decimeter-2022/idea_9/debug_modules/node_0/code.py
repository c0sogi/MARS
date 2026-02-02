import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn

# Import library modules
from library.config import Config
from library.utils import wgs84_to_cartesian, cartesian_to_wgs84, haversine_distance
from library.data_processing import load_data, prepare_drive_data
from library.dataset import GnssDriveDataset, gnss_collate_fn, get_dataloaders
from library.model import HybridResUNetGRU
from library.train import train_epoch, validate_epoch
from library.inference import predict_drive


def run_demo():
    print("--- Starting Demonstration ---")

    # 1. Configuration Setup
    print("\n[1] Configuring for fast demonstration...")
    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_DRIVE_COUNT = 2  # Only load 2 drives
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists for cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")
    lat_ref, lon_ref = 37.4, -122.1
    lat_target, lon_target = 37.41, -122.09

    # Forward transform
    north, east = wgs84_to_cartesian(lat_target, lon_target, lat_ref, lon_ref)
    print(f"  WGS84 -> Cartesian: North={north:.2f}, East={east:.2f}")

    # Inverse transform
    lat_rec, lon_rec = cartesian_to_wgs84(north, east, lat_ref, lon_ref)
    print(f"  Cartesian -> WGS84: Lat={lat_rec:.6f}, Lon={lon_rec:.6f}")

    # Check reconstruction
    assert np.isclose(lat_target, lat_rec), "Latitude reconstruction failed"
    assert np.isclose(lon_target, lon_rec), "Longitude reconstruction failed"

    # Distance
    dist = haversine_distance(lat_ref, lon_ref, lat_target, lon_target)
    print(f"  Haversine Distance: {dist:.2f} meters")
    assert dist > 0, "Distance should be positive"
    print("  Utils verification passed.")

    # 3. Data Loading and Processing
    print("\n[3] Loading Data (Train Split)...")
    # This uses load_data which respects Config.DEBUG
    # We set load_cached_data=False to verify the raw data processing logic
    try:
        train_data = load_data(split="train", load_cached_data=False)
        print(f"  Loaded {len(train_data)} drives.")

        if len(train_data) == 0:
            print("  No data loaded. Checking metadata existence...")
            if os.path.exists(Config.TRAIN_METADATA_PATH):
                print(f"  Metadata found at {Config.TRAIN_METADATA_PATH}")
                df = pd.read_csv(Config.TRAIN_METADATA_PATH)
                print(f"  Metadata rows: {len(df)}")
            else:
                print("  Metadata NOT found.")
            raise ValueError("No training data loaded. Cannot proceed.")

        sample_drive = train_data[0]
        print(f"  Sample Drive ID: {sample_drive['drive_id']}")
        print(f"  Features shape: {sample_drive['features'].shape}")  # (T, C)
        print(f"  Targets shape: {sample_drive['targets'].shape}")  # (T, 2)

    except Exception as e:
        print(f"  Data loading failed: {e}")
        raise

    # 4. Dataset and DataLoader
    print("\n[4] Initializing Dataset and DataLoader...")
    dataset = GnssDriveDataset(train_data)
    print(f"  Dataset length: {len(dataset)}")

    item = dataset[0]
    # Dataset returns features transposed: (Channels, Time)
    print(f"  Item features shape: {item['features'].shape}")
    print(f"  Item targets shape: {item['targets'].shape}")

    # Verify DataLoader collation
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=gnss_collate_fn, shuffle=False
    )

    batch = next(iter(loader))
    print(f"  Batch features shape: {batch['features'].shape}")  # (B, C, T_max)
    print(f"  Batch targets shape: {batch['targets'].shape}")  # (B, 2, T_max)
    print(f"  Batch mask shape: {batch['mask'].shape}")  # (B, T_max)

    assert batch["features"].dim() == 3, "Batch features should be 3D"
    assert batch["targets"].dim() == 3, "Batch targets should be 3D"

    # 5. Model Initialization and Forward Pass
    print("\n[5] Initializing Model...")
    model = HybridResUNetGRU()
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Calculate input dim based on config to verify model init
    expected_input_dim = Config.CN0_BINS + Config.ELEVATION_BINS + 3 + 3 + 1 + 1
    print(f"  Expected Input Dim: {expected_input_dim}")

    # Run forward pass with batch
    features = batch["features"].to(device)
    print(f"  Input tensor shape: {features.shape}")

    outputs = model(features)
    print(f"  Output tensor shape: {outputs.shape}")

    assert outputs.shape[0] == features.shape[0], "Batch size mismatch"
    assert outputs.shape[1] == 2, "Output channels should be 2 (North, East)"
    assert outputs.shape[2] == features.shape[2], "Temporal dimension mismatch"
    print("  Model forward pass successful.")

    # 6. Training Step
    print("\n[6] Running Training Step...")
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch (which iterates the small debug loader)
    loss = train_epoch(model, loader, criterion, optimizer, device)
    print(f"  Epoch Loss: {loss:.6f}")
    assert loss > 0, "Loss should be positive"

    # 7. Validation Step
    print("\n[7] Running Validation Step...")
    # Using the same loader for validation just to demonstrate the function
    val_loss, val_dist, val_score = validate_epoch(model, loader, criterion, device)
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"  Val Mean Distance: {val_dist:.6f} m")
    print(f"  Val Score: {val_score:.6f}")

    # 8. Inference Function
    print("\n[8] Testing Inference Function...")
    # Test predict_drive with the batch features
    preds = predict_drive(model, features, device)
    print(f"  Predictions shape: {preds.shape}")
    assert preds.shape == outputs.shape, "Inference output shape mismatch"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
