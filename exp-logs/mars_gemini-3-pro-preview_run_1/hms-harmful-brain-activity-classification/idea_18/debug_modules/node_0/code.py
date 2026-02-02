import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import library components
from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits
from library.data import get_dataloader
from library.models import BottleneckProjectedFusionNet
from library.train import train_one_epoch, validate_one_epoch


def run_demonstration():
    print("--- Starting Library Demonstration ---")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config parameters for a fast demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.SUBSET_SIZE = 16  # Only use 16 samples
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in demo
    Config.CACHE_DIR = "./working/demo_cache"
    Config.OUTPUT_DIR = "./working/demo_output"

    # Ensure clean working directories
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration configured for rapid execution.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline (Loading, Processing, Caching)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing Data Pipeline...")

    # Load metadata
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # Select a tiny subset for the demo
    demo_df = full_train_df.head(Config.SUBSET_SIZE).copy()
    print(f"Selected {len(demo_df)} samples for demonstration.")

    # Initialize DataLoader
    # This handles raw data loading, preprocessing, and caching to disk
    train_loader = get_dataloader(
        demo_df, Config, mode="train", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Validate Batch Shapes
    print("Verifying DataLoader output shapes...")
    eeg_batch, spec_batch, target_batch = next(iter(train_loader))

    # Check EEG: (Batch, Channels, Time) -> (4, 20, 5000)
    expected_eeg_shape = (Config.BATCH_SIZE, Config.EEG_CHANNELS, Config.EEG_SEQ_LEN)
    if eeg_batch.shape != expected_eeg_shape:
        raise AssertionError(
            f"EEG shape mismatch. Expected {expected_eeg_shape}, got {eeg_batch.shape}"
        )

    # Check Spectrogram: (Batch, Channels, H, W) -> (4, 5, 512, 512)
    expected_spec_shape = (
        Config.BATCH_SIZE,
        Config.SPEC_CHANNELS,
        Config.SPEC_SIZE[0],
        Config.SPEC_SIZE[1],
    )
    if spec_batch.shape != expected_spec_shape:
        raise AssertionError(
            f"Spectrogram shape mismatch. Expected {expected_spec_shape}, got {spec_batch.shape}"
        )

    # Check Targets: (Batch, Classes) -> (4, 6)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    if target_batch.shape != expected_target_shape:
        raise AssertionError(
            f"Target shape mismatch. Expected {expected_target_shape}, got {target_batch.shape}"
        )

    print("Data Pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model...")

    device = Config.DEVICE
    model = BottleneckProjectedFusionNet(Config).to(device)

    # Move batch to device
    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)
    target_batch = target_batch.to(device)

    print("Executing Forward Pass...")
    logits = model(eeg_batch, spec_batch)

    # Validate Output Shape
    if logits.shape != expected_target_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"
        )

    print("Model initialization and forward pass passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Simulating Training Step...")

    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = KLDivLossWithLogits()

    # Scheduler setup
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = OneCycleLR(
        optimizer, max_lr=Config.LR, total_steps=total_steps, pct_start=0.1
    )

    # Run one epoch of training
    train_loss = train_one_epoch(
        train_loader, model, optimizer, scheduler, criterion, device
    )

    print(f"Training Epoch Completed. Loss: {train_loss:.6f}")

    if np.isnan(train_loss) or train_loss <= 0:
        raise AssertionError("Training loss is invalid (NaN or <= 0).")

    # Run one epoch of validation
    val_loss = validate_one_epoch(train_loader, model, criterion, device)
    print(f"Validation Epoch Completed. Loss: {val_loss:.6f}")

    # -------------------------------------------------------------------------
    # 5. Inference & Probability Check
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Inference Constraints...")

    model.eval()
    with torch.no_grad():
        logits = model(eeg_batch, spec_batch)
        probs = torch.softmax(logits, dim=1)

        # Check if probabilities sum to 1
        sums = probs.sum(dim=1).cpu().numpy()
        print(f"Probability Sums: {sums}")

        if not np.allclose(sums, 1.0, atol=1e-5):
            raise AssertionError("Predicted probabilities do not sum to 1.0!")

    print("Inference verification passed.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
