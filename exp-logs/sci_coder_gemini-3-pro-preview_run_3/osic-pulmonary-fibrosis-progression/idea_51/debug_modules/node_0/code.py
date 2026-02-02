import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    seed_everything,
    get_global_stats,
    laplace_log_likelihood_score,
)
from library.data import get_dataloaders
from library.model import ARLRNet
from library.train import Trainer, LaplaceNLLLoss


def run_demo():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup & Override for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Override Config defaults to run quickly on a tiny subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10  # Use only 10 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script stability

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Apply setup to create directories
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Data Pipeline...")

    # This will trigger caching for the debug samples
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Verify Keys
    expected_keys = {
        "image",
        "stream_a",
        "stream_b",
        "patient_week",
        "target",
        "fvc_raw",
    }
    assert (
        set(batch.keys()) == expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Image: (B, 3, 260, 260) -> Config.IMG_SIZE is 260
    imgs = batch["image"]
    assert imgs.ndim == 4 and imgs.shape[1] == 3, f"Image shape mismatch: {imgs.shape}"
    assert imgs.shape[0] == Config.BATCH_SIZE, f"Batch size mismatch: {imgs.shape[0]}"

    # Stream A: (B, 2)
    stream_a = batch["stream_a"]
    assert stream_a.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Stream A shape mismatch: {stream_a.shape}"

    # Stream B: (B, 5)
    stream_b = batch["stream_b"]
    assert stream_b.shape == (
        Config.BATCH_SIZE,
        5,
    ), f"Stream B shape mismatch: {stream_b.shape}"

    print("Data Pipeline verification successful. Batch shapes correct.")

    # -------------------------------------------------------------------------
    # 3. Model Logic Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    # Get global stats for model initialization
    global_mean, global_std = get_global_stats()

    # Instantiate Model
    model = ARLRNet(global_std_target=global_std)
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    imgs = imgs.to(Config.DEVICE)
    stream_a = stream_a.to(Config.DEVICE)
    stream_b = stream_b.to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        outputs = model(imgs, stream_a, stream_b)

    # Verify Output Shape: (B, 2) -> [mu, sigma]
    assert outputs.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Model output shape mismatch: {outputs.shape}"

    # Verify Sigma Constraint (Must be positive)
    sigmas = outputs[:, 1]
    assert torch.all(sigmas > 0), "Model produced non-positive confidence values!"

    print("Model forward pass successful. Output shapes and constraints verified.")

    # -------------------------------------------------------------------------
    # 4. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Metric Calculation...")

    # Test Case 1: Perfect prediction with minimum confidence
    # Metric = - (sqrt(2) * 0) / 70 - ln(sqrt(2) * 70)
    #        = 0 - ln(98.99) ≈ -4.595
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([70])  # Min sigma

    score = laplace_log_likelihood_score(y_true, y_pred, sigma)
    expected = -np.log(np.sqrt(2) * 70)

    assert np.isclose(
        score, expected, atol=1e-4
    ), f"Metric calculation failed. Got {score}, expected {expected}"

    # Test Case 2: Large error (clipped at 1000)
    # Error = 2000 -> Clipped to 1000
    # Sigma = 100
    # Metric = - (sqrt(2) * 1000) / 100 - ln(sqrt(2) * 100)
    #        = - 14.142 - 4.951 ≈ -19.09
    y_true_bad = np.array([2000])
    y_pred_bad = np.array([4000])
    sigma_bad = np.array([100])

    score_bad = laplace_log_likelihood_score(y_true_bad, y_pred_bad, sigma_bad)
    expected_bad = -(np.sqrt(2) * 1000) / 100 - np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score_bad, expected_bad, atol=1e-4
    ), f"Metric clipping failed. Got {score_bad}, expected {expected_bad}"

    print("Metric function logic verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, (global_mean, global_std))

    # Run Fit
    trainer.fit()

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Training successful. Checkpoint found at: {checkpoint_path}")
    else:
        # Note: If validation score doesn't improve (which is possible in 1 epoch with random weights),
        # best_model might not be saved if we initialized best_score to -inf and the first run was valid.
        # However, the Trainer logic saves on the first improvement. Since best_score starts at -inf,
        # the first validation score (even if bad) is > -inf, so it should save.
        raise AssertionError("Checkpoint file was not created!")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
