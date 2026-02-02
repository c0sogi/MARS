import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import NFLDataLoader
from library.dataset import ContactDataset
from library.model import TDSRVNet
from library.trainer import Trainer


def run_demo():
    # 1. Setup & Configuration
    print("=== Setting up Demo Configuration ===")
    seed_everything(42)

    # Define temporary demo directories
    DEMO_DIR = "./working/demo_run"
    DEMO_INPUT = os.path.join(DEMO_DIR, "input")
    os.makedirs(DEMO_INPUT, exist_ok=True)

    # Modify Config globally to use demo paths and settings
    Config.WORKING_DIR = os.path.join(DEMO_DIR, "working")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64  # Small batch for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # 2. Create a Tiny Subset of Data (Speed Optimization)
    print("=== Creating Tiny Dataset Subset ===")

    # Load original metadata
    full_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Pick 1 game_play for train and 1 for validation
    train_gp = full_train_meta["game_play"].unique()[0]
    val_gp = full_val_meta["game_play"].unique()[0]

    print(f"Selected Train Play: {train_gp}")
    print(f"Selected Val Play:   {val_gp}")

    # Filter Metadata
    demo_train_meta = full_train_meta[full_train_meta["game_play"] == train_gp].copy()
    demo_val_meta = full_val_meta[full_val_meta["game_play"] == val_gp].copy()

    # Save Demo Metadata
    demo_train_meta_path = os.path.join(DEMO_INPUT, "train_meta.csv")
    demo_val_meta_path = os.path.join(DEMO_INPUT, "val_meta.csv")
    demo_train_meta.to_csv(demo_train_meta_path, index=False)
    demo_val_meta.to_csv(demo_val_meta_path, index=False)

    # Update Config Paths
    Config.TRAIN_METADATA_PATH = demo_train_meta_path
    Config.VAL_METADATA_PATH = demo_val_meta_path

    # Filter and Save Tracking Data
    # We need tracking for both the train play and val play
    relevant_plays = [train_gp, val_gp]

    print("Filtering Tracking Data...")
    # Read only necessary columns to speed up initial load if possible,
    # but here we read full and filter since we need to save a valid subset file.
    full_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    demo_tracking = full_tracking[
        full_tracking["game_play"].isin(relevant_plays)
    ].copy()
    demo_tracking_path = os.path.join(DEMO_INPUT, "train_tracking.csv")
    demo_tracking.to_csv(demo_tracking_path, index=False)
    Config.TRAIN_TRACKING_PATH = demo_tracking_path

    # Filter and Save Helmets Data
    print("Filtering Helmets Data...")
    full_helmets = pd.read_csv(Config.TRAIN_HELMETS_PATH)
    demo_helmets = full_helmets[full_helmets["game_play"].isin(relevant_plays)].copy()
    demo_helmets_path = os.path.join(DEMO_INPUT, "train_helmets.csv")
    demo_helmets.to_csv(demo_helmets_path, index=False)
    Config.TRAIN_HELMETS_PATH = demo_helmets_path

    # 3. Feature Generation & Data Loading
    print("\n=== Running NFLDataLoader ===")
    data_loader = NFLDataLoader()

    # Load Train
    # Note: load_cached_data=False forces regeneration using our new demo files
    train_data = data_loader.load_split("train", load_cached_data=False)

    # Validate Train Data
    print(f"Train Feature Shape: {train_data['X_num'].shape}")
    assert train_data["X_num"].shape[0] == len(
        demo_train_meta
    ), f"Mismatch in train samples: {train_data['X_num'].shape[0]} vs {len(demo_train_meta)}"
    assert train_data["y"] is not None, "Train targets missing"

    # Load Val
    val_data = data_loader.load_split("val", load_cached_data=False)

    # Validate Val Data
    print(f"Val Feature Shape: {val_data['X_num'].shape}")
    assert val_data["X_num"].shape[0] == len(
        demo_val_meta
    ), f"Mismatch in val samples: {val_data['X_num'].shape[0]} vs {len(demo_val_meta)}"

    # 4. Dataset & PyTorch Loader
    print("\n=== Initializing ContactDataset & DataLoader ===")
    train_dataset = ContactDataset(
        train_data["X_num"], train_data["X_cat"], train_data["y"]
    )
    val_dataset = ContactDataset(val_data["X_num"], val_data["X_cat"], val_data["y"])

    # Verify Dataset Item Structure
    sample_item, sample_target = train_dataset[0]
    x_kin, x_vis, x_cat = sample_item

    # Expected dimensions based on Config
    # Window=11.
    # Kinematic per step = (16 raw * 2 players) + 4 derived = 36. Total = 36 * 11 = 396
    # Visual per step = (5 raw * 2 players) = 10. Total = 10 * 11 = 110
    expected_kin_dim = 396
    expected_vis_dim = 110

    assert (
        x_kin.shape[0] == expected_kin_dim
    ), f"Expected kin dim {expected_kin_dim}, got {x_kin.shape[0]}"
    assert (
        x_vis.shape[0] == expected_vis_dim
    ), f"Expected vis dim {expected_vis_dim}, got {x_vis.shape[0]}"
    assert x_cat.shape[0] == 4, "Expected 4 categorical features"

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 5. Model Initialization
    print("\n=== Initializing TDSRVNet Model ===")
    model = TDSRVNet()

    # Test Forward Pass with a single batch
    device = Config.DEVICE
    model.to(device)

    dummy_batch = next(iter(train_loader))
    (d_kin, d_vis, d_cat), d_y = dummy_batch
    d_kin, d_vis, d_cat = d_kin.to(device), d_vis.to(device), d_cat.to(device)

    with torch.no_grad():
        output = model(d_kin, d_vis, d_cat)

    assert output.shape == (
        d_kin.size(0),
    ), f"Model output shape mismatch: {output.shape}"
    print("Forward pass successful.")

    # 6. Training Loop
    print("\n=== Starting Training (1 Epoch) ===")
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, epochs=1)

    # Verify artifacts
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model checkpoint not found after training"
    print("Training artifact verified.")

    # 7. Inference / Prediction
    print("\n=== Running Inference on Validation Set ===")
    # Using val_loader as test loader for demonstration
    preds = trainer.predict(val_loader)

    assert len(preds) == len(
        val_dataset
    ), f"Prediction count mismatch: {len(preds)} vs {len(val_dataset)}"

    print(f"Generated {len(preds)} predictions.")
    print(f"Sample predictions (first 5): {preds[:5]}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
