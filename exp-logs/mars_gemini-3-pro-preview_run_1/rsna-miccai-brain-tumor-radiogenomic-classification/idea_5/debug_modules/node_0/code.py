import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import BraTSDataset
from library.model import SiameseEfficientNet
from library.engine import run_training, predict_and_submit


def main():
    print("Initializing Demonstration Script...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We modify the Config class directly to optimize for a quick run
    # without modifying the source file.

    # Use a specific demo directory in working/
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    # Paths for outputs
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Speed optimizations
    Config.IMG_SIZE = 128  # Smaller images for faster processing
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Main process only to avoid overhead
    Config.EARLY_STOPPING_PATIENCE = 1

    # Re-run setup to create the new directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration updated for speed.")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n--- Step 2: Data Loading & Verification ---")

    # Load a tiny subset of data (8 train, 4 val)
    train_dataset = BraTSDataset(split="train", load_cached_data=False, debug_limit=8)
    val_dataset = BraTSDataset(split="val", load_cached_data=False, debug_limit=4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Verify Batch Shapes
    # Expected Input: (Batch, Views=3, Channels=3, H, W)
    # Expected Target: (Batch,)
    inputs, targets = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    assert inputs.shape == (
        Config.BATCH_SIZE,
        3,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect input shape. Expected {(Config.BATCH_SIZE, 3, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {inputs.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape. Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"

    print("Data loading verification passed.")

    # ==========================================
    # 3. Model Initialization & Verification
    # ==========================================
    print("\n--- Step 3: Model Initialization & Verification ---")

    device = torch.device(Config.DEVICE)
    model = SiameseEfficientNet(pretrained=False)  # False for speed, logic remains same
    model.to(device)

    # Verify Forward Pass
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)

    print(f"Logits shape: {logits.shape}")

    # Expected Output: (Batch, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect output shape. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model forward pass verification passed.")

    # ==========================================
    # 4. Training Execution
    # ==========================================
    print("\n--- Step 4: Training Execution ---")

    # Run the training loop
    best_auc = run_training(train_loader, val_loader)

    # Verify model file creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {Config.MODEL_SAVE_PATH}"

    print(f"Training finished. Best AUC: {best_auc}")
    print("Model file verification passed.")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    print("\n--- Step 5: Inference Execution ---")

    # Load test subset
    test_dataset = BraTSDataset(split="test", load_cached_data=False, debug_limit=4)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run prediction
    predict_and_submit(test_loader, test_dataset.df)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Check dimensions
    assert len(df_sub) == 4, f"Expected 4 predictions, got {len(df_sub)}"
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission columns missing."

    print("Inference verification passed.")

    print("\n==========================================")
    print(" DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
