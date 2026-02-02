import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_loader
from library.model import DSDBiGRUModel
from library.train import Trainer


def run_demo():
    print("==== RNA Degradation Prediction Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demo Speed and Isolation
    # ---------------------------------------------------------
    # We modify the Config class directly to isolate artifacts
    # and ensure the demo runs within the time limit.

    DEMO_DIR = "./working/demo_execution"
    # Clean up previous demo run if exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"1. Configuring environment (Dir: {DEMO_DIR})...")

    # Update paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_data.npz")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "demo_model.pth")

    # Reduce computational load for the demo
    Config.EPOCHS = 2  # Run just 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Loading and Processing Demonstration
    # ---------------------------------------------------------
    print("\n2. Demonstrating Data Loading and Processing...")

    # Load validation data.
    # Since the cache file defined above doesn't exist, this will trigger
    # the 'process_dataframe' logic in library.data.
    val_loader = get_loader(
        "val", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch a single batch to inspect structure
    batch = next(iter(val_loader))

    inputs = batch["inputs"]
    bpp_indices = batch["bpp_indices"]
    bpp_mask = batch["bpp_mask"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"   Batch loaded. Batch size: {inputs.shape[0]}")

    # Assertions to verify data shapes
    # Input: (B, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch: {inputs.shape}"

    # Targets: (B, Seq_Len=107, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {targets.shape}"

    # Structure indices: (B, Seq_Len=107)
    assert bpp_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"BPP indices shape mismatch: {bpp_indices.shape}"

    print("   Data shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization and Inference Demonstration
    # ---------------------------------------------------------
    print("\n3. Demonstrating Model Initialization and Inference...")

    device = Config.DEVICE
    model = DSDBiGRUModel(Config).to(device)

    # Move batch data to the appropriate device
    inputs = inputs.to(device)
    bpp_indices = bpp_indices.to(device)
    bpp_mask = bpp_mask.to(device)

    # Perform inference (Forward Pass)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, bpp_indices, bpp_mask)

    print(f"   Model output shape: {outputs.shape}")

    # Verify output shape matches (B, L, 5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Model output shape does not match expected dimensions."

    print("   Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Metric Calculation Demonstration
    # ---------------------------------------------------------
    print("\n4. Demonstrating Metric Calculation (MCRMSE)...")

    # Move targets to device for comparison
    targets = targets.to(device)

    # Calculate MCRMSE
    # This function handles the slicing to the first 68 positions internally
    score = mcrmse_metric(outputs, targets)

    print(f"   Calculated MCRMSE: {score:.4f}")
    assert (
        isinstance(score, float) and score >= 0
    ), "Metric should be a non-negative float."

    # ---------------------------------------------------------
    # 5. Full Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n5. Running Training Loop (Short Run)...")

    # Instantiate Trainer
    # It will use the modified Config class attributes we set earlier
    trainer = Trainer(config=Config)

    # Run training
    # This will:
    # 1. Process training data (and cache it to demo_execution/train_data.npz)
    # 2. Train for 2 epochs
    # 3. Validate after each epoch
    # 4. Save the best model to demo_execution/demo_model.pth
    trainer.run()

    # Verify that the model artifact was created
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print(f"   Training complete. Model saved to {Config.BEST_MODEL_PATH}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
