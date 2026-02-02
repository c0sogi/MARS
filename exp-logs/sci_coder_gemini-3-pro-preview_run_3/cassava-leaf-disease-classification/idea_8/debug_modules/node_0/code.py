import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders
from library.models import get_model
from library.training import Trainer
from library.inference import generate_ensemble_predictions


def run_demo():
    print("=== Starting Cassava Leaf Disease Classification Demo ===")

    # 1. Configuration & Setup
    # Monkey-patch Config class for a fast demo run
    print("Step 1: Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample for speed
    Config.EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
    Config.IMG_SIZE = 224  # Smaller image size (ResNet standard)
    Config.MODEL_A_NAME = "resnet18"  # Lightweight model
    Config.MODEL_B_NAME = "resnet18"  # Lightweight model
    Config.WORKING_DIR = "./working/demo_output"
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Instantiate config object
    cfg = Config()
    device = get_device()
    logger = get_logger(os.path.join(cfg.WORKING_DIR, "demo.log"))

    print(f"Configuration set: Debug={cfg.DEBUG}, Device={device}")

    # 2. Data Loading Demonstration
    print("\nStep 2: Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Validation: Check DataLoader content
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch a batch to verify shapes and transforms
    imgs, targets = next(iter(train_loader))
    print(f"Batch shapes - Images: {imgs.shape}, Targets: {targets.shape}")

    # Validate Image Shape: (Batch, Channel, Height, Width)
    assert imgs.shape == (
        cfg.BATCH_SIZE,
        3,
        cfg.IMG_SIZE,
        cfg.IMG_SIZE,
    ), f"Incorrect image shape: {imgs.shape}"

    # Validate Targets: Should be (Batch, Num_Classes) due to Mixup/Cutmix or (Batch) if not mixed
    # The MixupCutmixCollator usually returns (Batch, Num_Classes)
    if targets.dim() == 2:
        assert targets.shape == (
            cfg.BATCH_SIZE,
            cfg.NUM_CLASSES,
        ), f"Incorrect target shape for mixed data: {targets.shape}"
    else:
        assert targets.shape == (
            cfg.BATCH_SIZE,
        ), f"Incorrect target shape for hard labels: {targets.shape}"

    print("Data loading verified successfully.")

    # 3. Model Training Demonstration
    print("\nStep 3: Training Model (ResNet18)...")
    model = get_model(cfg, cfg.MODEL_B_NAME)  # Using Model B name (resnet18)

    # Optimizer setup (Simplified for demo)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, logger)

    # Run training for 1 epoch
    checkpoint_name = "demo_checkpoint.pth"
    best_acc = trainer.fit(
        optimizer, scheduler=None, epochs=cfg.EPOCHS, save_name=checkpoint_name
    )

    # Validation: Check if model checkpoint was saved
    checkpoint_path = os.path.join(cfg.WORKING_DIR, checkpoint_name)
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Training completed. Checkpoint saved at {checkpoint_path}")

    # 4. Inference & Ensemble Demonstration
    print("\nStep 4: Running Ensemble Inference...")

    # We use the same checkpoint for both models to demonstrate the ensemble logic
    # without needing to train two separate networks.
    generate_ensemble_predictions(
        model_a_path=checkpoint_path, model_b_path=checkpoint_path
    )

    # Validation: Check submission file
    submission_file = cfg.SUBMISSION_PATH
    assert os.path.exists(
        submission_file
    ), f"Submission file not found at {submission_file}"

    df_sub = pd.read_csv(submission_file)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Validate Submission Content
    assert (
        "image_id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns"
    assert len(df_sub) == len(
        test_loader.dataset
    ), f"Submission row count ({len(df_sub)}) does not match test set size ({len(test_loader.dataset)})"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
