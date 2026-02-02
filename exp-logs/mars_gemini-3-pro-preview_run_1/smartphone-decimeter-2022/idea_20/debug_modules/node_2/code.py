import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

from library.config import CFG
from library.utils import wgs84_to_enu, enu_to_wgs84, seed_everything
from library.data_preprocessing import process_dataset
from library.dataset import GnssSequenceDataset, get_scaler
from library.model import ResUNet1D
from library.engine import train_model


def run_demo():
    # 1. Setup and Configuration Override for Demo
    print("\n[1] Setup and Configuration")
    seed_everything(42)

    # Create a working directory for this demo
    demo_dir = os.path.join(CFG.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Override CFG for speed
    CFG.WORKING_DIR = demo_dir
    CFG.EPOCHS = 2
    CFG.BATCH_SIZE = 4
    CFG.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    CFG.PATIENCE = 1

    print(f"Working directory set to: {CFG.WORKING_DIR}")
    print(f"Device: {CFG.DEVICE}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions")
    lat_ref, lon_ref = 37.4, -122.1
    lat_target, lon_target = 37.41, -122.09

    # Forward conversion
    east, north = wgs84_to_enu(lat_target, lon_target, lat_ref, lon_ref)

    # Inverse conversion
    lat_rec, lon_rec = enu_to_wgs84(east, north, lat_ref, lon_ref)

    # Check consistency
    print(f"Reference: ({lat_ref}, {lon_ref})")
    print(f"Target:    ({lat_target}, {lon_target})")
    print(f"ENU:       East={east:.2f}m, North={north:.2f}m")
    print(f"Recovered: ({lat_rec:.6f}, {lon_rec:.6f})")

    assert np.isclose(lat_target, lat_rec, atol=1e-5), "Latitude reconstruction failed"
    assert np.isclose(lon_target, lon_rec, atol=1e-5), "Longitude reconstruction failed"
    print("Coordinate conversion round-trip successful.")

    # 3. Data Preprocessing Demo
    print("\n[3] Data Preprocessing")

    # Load full training metadata
    if not os.path.exists(CFG.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {CFG.TRAIN_METADATA_PATH}")

    full_meta = pd.read_csv(CFG.TRAIN_METADATA_PATH)

    # Sample a small subset (e.g., 1 drive) for the demo to run quickly
    # We pick a drive that has data
    sample_drive_id = full_meta["drive_id"].unique()[0]
    mini_meta = full_meta[full_meta["drive_id"] == sample_drive_id].copy()

    # Limit to first 200 rows to keep it very fast
    mini_meta = mini_meta.head(200)

    print(
        f"Creating mini metadata with {len(mini_meta)} samples from drive {sample_drive_id}"
    )
    mini_meta_path = os.path.join(demo_dir, "mini_train_meta.csv")
    mini_meta.to_csv(mini_meta_path, index=False)

    # Define cache path for this demo
    mini_cache_path = os.path.join(demo_dir, "cache", "train_processed.parquet")

    # Process the dataset using the library function
    # Note: We set load_cached_data=False to force processing
    df_processed = process_dataset(
        metadata_path=mini_meta_path,
        cache_path=mini_cache_path,
        load_cached_data=False,
        debug=False,
    )

    print(f"Processed DataFrame shape: {df_processed.shape}")
    assert not df_processed.empty, "Preprocessing resulted in empty dataframe"
    assert "target_east" in df_processed.columns, "Target East column missing"
    assert "target_north" in df_processed.columns, "Target North column missing"

    # 4. Dataset Creation
    print("\n[4] Dataset Creation")

    # Split into train/val for the demo (80/20 split of the mini data)
    split_idx = int(len(df_processed) * 0.8)
    df_train = df_processed.iloc[:split_idx].copy()
    df_val = df_processed.iloc[split_idx:].copy()

    # Fit scaler
    scaler = get_scaler(df_train, CFG.FEATURE_COLS)

    # Transform
    df_train[CFG.FEATURE_COLS] = scaler.transform(df_train[CFG.FEATURE_COLS])
    df_val[CFG.FEATURE_COLS] = scaler.transform(df_val[CFG.FEATURE_COLS])

    target_cols = ["target_east", "target_north"]

    # Instantiate Datasets
    train_dataset = GnssSequenceDataset(
        df_train,
        CFG.FEATURE_COLS,
        target_cols,
        mode="train",
        sequence_length=32,  # Short seq for demo
    )
    val_dataset = GnssSequenceDataset(
        df_val, CFG.FEATURE_COLS, target_cols, mode="val", sequence_length=32
    )

    print(f"Train Dataset length: {len(train_dataset)}")
    print(f"Val Dataset length: {len(val_dataset)}")

    # Test __getitem__
    features, targets, mask, indices = train_dataset[0]
    print(f"Sample Features Shape: {features.shape} (Channels, Length)")
    print(f"Sample Targets Shape: {targets.shape} (Dims, Length)")
    print(f"Sample Mask Shape: {mask.shape} (Length)")

    assert features.shape[0] == len(CFG.FEATURE_COLS), "Incorrect feature channels"
    assert features.shape[1] == 32, "Incorrect sequence length"
    assert targets.shape[0] == 2, "Incorrect target dimensions"

    # 5. Model Instantiation
    print("\n[5] Model Initialization")
    model = ResUNet1D()
    model.to(CFG.DEVICE)

    # Test Forward Pass
    dummy_input = torch.randn(2, len(CFG.FEATURE_COLS), 32).to(
        CFG.DEVICE
    )  # Batch size 2
    outputs = model(dummy_input)

    print(f"Model Output format: List of length {len(outputs)}")
    print(f"Main Head Output Shape: {outputs[0].shape}")

    assert len(outputs) == 3, "Model should return 3 outputs (Main + 2 Aux)"
    assert outputs[0].shape == (2, 2, 32), "Main output shape mismatch"
    assert outputs[1].shape == (
        2,
        2,
        16,
    ), "Aux1 output shape mismatch (should be 1/2 res)"
    assert outputs[2].shape == (
        2,
        2,
        8,
    ), "Aux2 output shape mismatch (should be 1/4 res)"
    print("Model forward pass successful.")

    # 6. Training Loop Demo
    print("\n[6] Training Loop Demo")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        drop_last=False,
    )

    # Run Training
    # We update CFG.BEST_MODEL_PATH to be inside our demo dir
    CFG.BEST_MODEL_PATH = os.path.join(demo_dir, "models", "best_model.pth")
    os.makedirs(os.path.dirname(CFG.BEST_MODEL_PATH), exist_ok=True)

    trained_model = train_model(model, train_loader, val_loader)

    print("\n[7] Verification")
    if os.path.exists(CFG.BEST_MODEL_PATH):
        print(f"Successfully saved best model to {CFG.BEST_MODEL_PATH}")
    else:
        raise AssertionError("Model checkpoint was not saved.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
