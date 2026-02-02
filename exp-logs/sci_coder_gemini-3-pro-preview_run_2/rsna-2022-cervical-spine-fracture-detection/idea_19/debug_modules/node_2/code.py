import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, get_test_dataloader
from library.model import CalibratedHierarchicalSeqModel
from library.loss import WeightedMultiLabelLoss
from library.train import run_training
from library.inference import predict_test_set


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> [1/6] Setting up configuration for fast demonstration...")
    seed_everything(42)

    # Define a temporary working directory for this demo to avoid conflicts
    demo_dir = os.path.join("working", "demo_execution")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to point to the demo directory
    Config.WORKING_DIR = demo_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.LAST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "last_checkpoint.pth")
    Config.LOG_FILE = os.path.join(demo_dir, "demo.log")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Cache paths to ensure we don't use stale data or write to restricted areas
    Config.TRAIN_CACHE_PATH = os.path.join(demo_dir, "train_paths_cache.parquet")
    Config.VAL_CACHE_PATH = os.path.join(demo_dir, "val_paths_cache.parquet")
    Config.TEST_CACHE_PATH = os.path.join(demo_dir, "test_paths_cache.parquet")

    # Override Hyperparameters for speed (Tiny model, tiny data)
    Config.IMAGE_SIZE = (128, 128)  # Reduce from 384 to 128
    Config.SEQ_LENGTH = 16  # Reduce from 96 to 16
    Config.BATCH_SIZE = 2  # Small batch size
    Config.ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing overhead
    Config.EPOCHS = 1  # Only 1 epoch
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SAMPLE_SIZE = 6  # Only use 6 samples

    # Ensure necessary directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print(">>> [2/6] Verifying Data Loading...")
    # load_cached_data=False forces re-computation of paths for our demo cache location
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "Train loader is empty. Check metadata or debug sample size."
        )

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    # Validate Shapes
    # Expected: (Batch, Seq, Channels=3, Height, Width)
    expected_img_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}, expected {expected_img_shape}"

    # Expected: (Batch, Num_Classes=8)
    assert targets.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Target shape mismatch. Got {targets.shape}"

    print("    Data Loading passed.")

    # -------------------------------------------------------------------------
    # 3. Model Logic Verification
    # -------------------------------------------------------------------------
    print(">>> [3/6] Verifying Model Architecture...")
    device = get_device()

    # Initialize model (pretrained=False to avoid downloading weights during demo)
    model = CalibratedHierarchicalSeqModel(pretrained=False)
    model = model.to(device)

    # Move batch to device
    images = images.to(device)
    targets = targets.to(device)

    # Forward Pass
    logits = model(images)

    print(f"    Logits Shape: {logits.shape}")

    # Validate Output
    assert logits.shape == (Config.BATCH_SIZE, 8), "Model output shape is incorrect."
    assert torch.isfinite(logits).all(), "Model produced NaN or Inf values."

    print("    Model Forward Pass passed.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print(">>> [4/6] Verifying Loss Calculation...")
    loss_fn = WeightedMultiLabelLoss().to(device)

    loss = loss_fn(logits, targets)

    print(f"    Loss Value: {loss.item():.4f}")

    # Validate Loss
    assert loss.dim() == 0, "Loss must be a scalar."
    assert loss.item() >= 0, "Loss must be non-negative."

    print("    Loss Calculation passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # -------------------------------------------------------------------------
    print(">>> [5/6] Simulating Training Loop...")

    # Run a short training session
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        accumulation_steps=Config.ACCUMULATION_STEPS,
        debug=Config.DEBUG,
    )

    # Validate Checkpoints
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model checkpoint was not created at {Config.BEST_MODEL_PATH}"
        )

    print("    Training simulation passed. Checkpoints created.")

    # -------------------------------------------------------------------------
    # 6. Inference Simulation
    # -------------------------------------------------------------------------
    print(">>> [6/6] Simulating Inference...")

    # Run inference using the trained (checkpointed) model
    predict_test_set(load_cached_data=False, batch_size=Config.BATCH_SIZE, debug=True)

    # Validate Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(sub_df)}")

    # Check Schema
    assert "row_id" in sub_df.columns, "Submission missing 'row_id' column."
    assert "fractured" in sub_df.columns, "Submission missing 'fractured' column."

    # Check Probability Range
    probs = sub_df["fractured"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain values outside [0, 1]."

    print("    Inference simulation passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
