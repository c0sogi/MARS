import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import get_fold_loaders, get_test_loader
from library.model import CCTICNN
from library.train import run_fold


def run_demo():
    print("=== Starting Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config to run a lightweight version
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)

    # Re-define paths based on new working dir
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Speed optimizations
    Config.DEBUG = True  # Use subset of data
    Config.DEBUG_SUBSET_SIZE = 40  # Only 40 samples
    Config.BATCH_SIZE = 4  # Small batch
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_FOLDS = 2  # We will only run fold 0 anyway
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Create directories
    Config.create_directories()

    # Set global seed
    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading Pipeline...")

    # Get loaders for Fold 0
    # This triggers data processing and caching (or loads from cache if exists)
    train_loader, val_loader = get_fold_loaders(fold_index=0, load_cached_data=True)

    # Verify Train Loader
    print(f"Train Loader batches: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader is empty!"

    # Fetch one batch to check shapes
    (images, angles), labels, ids = next(iter(train_loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Image: (B, 3, 75, 75)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    # Angle: (B,)
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    # Labels: (B,)
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture (CCTI-CNN)...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = CCTICNN().to(device)

    # Forward pass with the batch fetched earlier
    # Model expects angles to be handled internally even if input is (B,)
    outputs = model(images.to(device), angles.to(device))

    print(f"Model Output Shape: {outputs.shape}")

    # Assertion: Output should be (B, 1) logits
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert torch.isfinite(outputs).all(), "Model produced NaN or Inf values"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Fold 0)...")

    # run_fold handles the entire training process for one fold
    # It returns the best validation loss
    best_val_loss = run_fold(fold_index=0)

    print(f"Training completed. Best Validation Loss: {best_val_loss}")
    assert isinstance(best_val_loss, float), "run_fold did not return a float loss"
    assert best_val_loss >= 0, "Loss cannot be negative"

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission (Inference)...")

    test_loader = get_test_loader(load_cached_data=True)

    # Load the best model from the training step
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file not found!"

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    predictions = []
    ids_list = []

    with torch.no_grad():
        for (t_images, t_angles), _, t_ids in test_loader:
            t_images = t_images.to(device)
            t_angles = t_angles.to(device)

            logits = model(t_images, t_angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids_list.extend(t_ids)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids_list, "is_iceberg": predictions})

    # Save submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(df_sub.head())

    assert len(df_sub) == len(ids_list), "Submission length mismatch"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
