import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# 1. Import Config first to patch it before other modules bind to its values
from library.config import Config

# =============================================================================
# Configuration Patching for Demo
# =============================================================================
print("Configuring demo parameters...")
# Set a specific working directory for this demo
Config.WORKING_DIR = "./working/demo_execution"
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Update paths to use the demo directory to avoid conflicts
Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npz")
Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npz")
Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npz")
Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
Config.SUBMISSION_FILE_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

# Reduce compute requirements for speed
Config.MAX_EPOCHS = 2
Config.BATCH_SIZE = 16
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
Config.STEM_FILTERS = 32  # Reduce model size for speed
Config.RNN_HIDDEN_DIM = 32
Config.INTERACTION_DIM = 64
Config.RNN_LAYERS = 2

# Import library modules after patching Config
from library.utils import set_seed, mcrmse_loss, compute_score
from library.data import get_dataloaders, RNADataset
from library.model import RNAModel
from library.train import run_training, generate_submission


def main():
    print(f"Running demo on device: {Config.DEVICE}")
    set_seed(Config.SEED)

    # =========================================================================
    # 1. Data Loading Verification
    # =========================================================================
    print("\n--- 1. Testing Data Loading ---")

    # Force reprocessing to ensure data pipeline works (ignore existing cache)
    if os.path.exists(Config.TRAIN_CACHE):
        os.remove(Config.TRAIN_CACHE)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    features = batch["features"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]
    ids = batch["id"]

    # Assertions to verify data integrity
    assert features.ndim == 3, f"Features should be 3D, got {features.shape}"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {features.shape[2]}"
    assert (
        targets.shape[2] == 5
    ), f"Targets should have 5 channels, got {targets.shape[2]}"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Pair indices shape mismatch"

    print("Data batch structure verified.")

    # =========================================================================
    # 2. Model Initialization and Forward Pass Verification
    # =========================================================================
    print("\n--- 2. Testing Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Move batch to device
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(features, pair_indices, pair_masks)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for model output
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, 5)}, got {outputs.shape}"

    print("Model forward pass successful.")

    # =========================================================================
    # 3. Loss Function Verification
    # =========================================================================
    print("\n--- 3. Testing Loss Calculation ---")

    # Test training loss
    loss = mcrmse_loss(outputs, targets)
    print(f"Calculated MCRMSE Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # Test scoring function (Competition Metric logic)
    # Move tensors to CPU as compute_score expects numpy or CPU tensors
    score = compute_score(
        outputs.detach().cpu().numpy(), targets.detach().cpu().numpy()
    )
    print(f"Validation Score (MCRMSE on scored cols): {score:.4f}")
    assert score >= 0, "Score must be non-negative"

    print("Loss and metric functions verified.")

    # =========================================================================
    # 4. Full Training Loop Execution
    # =========================================================================
    print("\n--- 4. Running Training Loop (Shortened) ---")

    # We use the library function run_training, which uses the patched Config
    # This will train for Config.MAX_EPOCHS (set to 2)
    run_training(train_loader, val_loader)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print("Training loop completed and model saved.")

    # =========================================================================
    # 5. Inference and Submission Generation
    # =========================================================================
    print("\n--- 5. Generating Submission ---")

    generate_submission(test_loader)

    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: N_test * 107. N_test is 240. 240 * 107 = 25680.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print("Submission generated and verified.")
    print("\nAll demo tasks completed successfully.")


if __name__ == "__main__":
    main()
