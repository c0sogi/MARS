import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, kl_divergence
from library.data import get_dataloaders
from library.models import HarmfulBrainActivityModel
from library.engine import fit

if __name__ == "__main__":
    # --- 1. Configuration & Setup ---
    print(">>> Setting up configuration for demonstration...")

    # Override Config for speed and offline execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Hyperparameters for fast demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.TRAIN_SAMPLE_SIZE = 50  # Only use 50 samples for training logic
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PRETRAINED = False  # Disable downloading weights for offline safety

    # Create directories
    Config.make_dirs()

    # Set deterministic seed
    seed_everything(Config.SEED)
    logger = get_logger("demo")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # --- 2. Data Loading & Verification ---
    print("\n>>> Loading Data (Debug Mode)...")

    # get_dataloaders with debug=True loads only 100 samples per split
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch to verify data pipeline
    batch = next(iter(train_loader))
    eeg_data = batch["eeg"]
    spec_data = batch["spec"]
    targets = batch["target"]

    print(
        f"Batch Shapes -> EEG: {eeg_data.shape}, Spec: {spec_data.shape}, Targets: {targets.shape}"
    )

    # Assertions to verify data pipeline logic
    # EEG: (Batch, 20 channels, 5000 time steps)
    assert eeg_data.shape == (
        Config.BATCH_SIZE,
        20,
        5000,
    ), f"EEG shape mismatch. Expected {(Config.BATCH_SIZE, 20, 5000)}, got {eeg_data.shape}"

    # Spectrogram: (Batch, 5 channels, 512 height, 512 width)
    assert spec_data.shape == (
        Config.BATCH_SIZE,
        5,
        512,
        512,
    ), f"Spectrogram shape mismatch. Expected {(Config.BATCH_SIZE, 5, 512, 512)}, got {spec_data.shape}"

    # Targets: (Batch, 6 classes)
    assert targets.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 6)}, got {targets.shape}"

    # Check probability sum constraint
    target_sums = targets.sum(dim=1)
    assert torch.allclose(
        target_sums, torch.ones_like(target_sums), atol=1e-4
    ), "Target probabilities do not sum to 1.0"

    print("Data pipeline verification passed.")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n>>> Initializing Model...")

    model = HarmfulBrainActivityModel(config=Config)
    model.to(Config.DEVICE)

    # Perform a dummy forward pass to verify architecture
    with torch.no_grad():
        # Move inputs to device
        dummy_eeg = eeg_data.to(Config.DEVICE)
        dummy_spec = spec_data.to(Config.DEVICE)

        logits = model(dummy_eeg, dummy_spec)

    print(f"Model Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 6)}, got {logits.shape}"

    print("Model architecture verification passed.")

    # --- 4. Training Loop Execution ---
    print("\n>>> Starting Training Loop...")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
    )

    # Run the training engine
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=Config,
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Training successful. Checkpoint found at: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            "Training finished but 'best_model.pth' was not created."
        )

    # --- 5. Inference & Metric Verification ---
    print("\n>>> Verifying Inference and Metric...")

    # Load best model
    best_model = HarmfulBrainActivityModel(config=Config)
    state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
    best_model.load_state_dict(state_dict)
    best_model.to(Config.DEVICE)
    best_model.eval()

    # Run inference on the validation batch loaded earlier
    # (Re-using the val_loader would be standard, but we use the pre-fetched batch for atomic verification)
    val_batch = next(iter(val_loader))
    val_eeg = val_batch["eeg"].to(Config.DEVICE)
    val_spec = val_batch["spec"].to(Config.DEVICE)
    val_targets = val_batch["target"].numpy()  # Keep on CPU for metric calc

    with torch.no_grad():
        # Get probabilities
        val_probs = best_model.predict(val_eeg, val_spec).cpu().numpy()

    print(f"Predictions Shape: {val_probs.shape}")
    print(f"Sample Prediction: {val_probs[0]}")

    # Verify predictions sum to 1
    pred_sums = val_probs.sum(axis=1)
    assert np.allclose(
        pred_sums, 1.0, atol=1e-4
    ), "Model predictions do not sum to 1.0 (Softmax failure)"

    # Calculate KL Divergence manually
    metric_val = kl_divergence(val_targets, val_probs)
    print(f"Manual KL Divergence on batch: {metric_val:.4f}")

    assert not np.isnan(metric_val), "Computed metric is NaN."

    print("\n>>> Demonstration Complete. All systems operational.")
