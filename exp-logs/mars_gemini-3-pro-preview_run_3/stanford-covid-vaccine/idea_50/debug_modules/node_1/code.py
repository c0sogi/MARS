import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, compute_mcrmse
from library.data import get_dataloaders
from library.model import SDBR_BiGRU
from library.train import train_model


def run_demo():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print(">>> Setting up configuration for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set deterministic seed
    seed_everything(42)

    # Modify Config for a fast run
    # We use a separate working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_cache.npy")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_cache.npy")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_cache.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Optimization for speed
    Config.SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 2  # Only 2 epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Subset Size: {Config.SUBSET_SIZE}, Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n>>> Loading and verifying data...")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to verify logic
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    features = batch["features"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]
    ids = batch["id"]

    # Verify Shapes
    # Features: (Batch, Seq_Len=107, Input_Dim=14)
    assert features.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Feature shape mismatch. Expected ({Config.BATCH_SIZE}, 107, 14), got {features.shape}"

    # Targets: (Batch, Pred_Len=68, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"Target shape mismatch. Expected ({Config.BATCH_SIZE}, 68, 5), got {targets.shape}"

    # Pair Indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Pair indices shape mismatch. Got {pair_indices.shape}"

    print("Data shapes verified successfully.")
    print(f"Features: {features.shape}")
    print(f"Targets:  {targets.shape}")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> Initializing model and running forward pass...")

    device = torch.device("cpu")  # Use CPU for simple demo verification
    model = SDBR_BiGRU().to(device)

    # Move batch to device
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)

    # Forward pass
    preds = model(features, pair_indices, pair_masks)

    # Verify Output Shape
    # Model outputs predictions for the full sequence length (107)
    # Shape: (Batch, Seq_Len=107, Num_Targets=5)
    assert preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Prediction shape mismatch. Expected ({Config.BATCH_SIZE}, 107, 5), got {preds.shape}"

    print("Forward pass successful.")
    print(f"Predictions: {preds.shape}")

    # ==========================================
    # 4. Loss Calculation Verification
    # ==========================================
    print("\n>>> Verifying Loss Calculation...")

    criterion = MCRMSELoss()

    # Calculate loss
    # The loss function inside library.utils handles the slicing of predictions
    # from 107 down to 68 to match targets.
    loss = criterion(preds, targets.to(device))

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print(f"Loss calculated successfully: {loss.item():.4f}")

    # ==========================================
    # 5. Full Training Pipeline
    # ==========================================
    print("\n>>> Executing Training Pipeline...")

    # We use the train_model function from library.train
    # This handles the loop, validation, and saving.
    best_score = train_model(
        epochs=Config.EPOCHS,
        subset_size=Config.SUBSET_SIZE,
        batch_size=Config.BATCH_SIZE,
        lr=1e-3,
        save_path=Config.MODEL_SAVE_PATH,
    )

    print(f"Training pipeline finished. Best Score: {best_score:.4f}")

    # Verify model file was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved.")

    # ==========================================
    # 6. Inference / Test Verification
    # ==========================================
    print("\n>>> Verifying Inference on Test Data...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_batch = next(iter(test_loader))
    t_feats = test_batch["features"].to(device)
    t_pidx = test_batch["pair_indices"].to(device)
    t_mask = test_batch["pair_masks"].to(device)

    with torch.no_grad():
        test_preds = model(t_feats, t_pidx, t_mask)

    # Test predictions should be (Batch, 107, 5)
    assert test_preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Test prediction shape mismatch. Got {test_preds.shape}"

    print("Inference verification successful.")
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
