import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_datasets
from library.model import SETIModel
from library.engine import SETIEngine


def run_demo():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("Initializing demonstration...")

    # Override Config for speed and demonstration purposes
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for quick verification
    Config.epochs = 1  # Single epoch to verify training loop
    Config.batch_size = 8  # Small batch size
    Config.pretrained = False  # Disable downloading weights for speed/offline safety
    Config.num_workers = 2  # Reduce workers for small data
    Config.print_freq = 5  # Frequent logging for small steps

    # Set output directory for this demo
    Config.output_dir = os.path.join(Config.working_dir, "demo_run")
    Config.create_dirs()

    # Ensure reproducibility
    seed_everything(Config.seed)

    device = Config.device
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.debug}")
    print(f"Output Directory: {Config.output_dir}")

    # ---------------------------------------------------------
    # 2. Data Loading and Verification
    # ---------------------------------------------------------
    print("\n--- Setting up Data ---")
    train_ds, val_ds, test_ds = get_datasets(debug=Config.debug)

    # Verify dataset lengths
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    assert len(train_ds) == Config.debug_sample_size, "Train dataset size mismatch"
    assert len(val_ds) == Config.debug_sample_size, "Val dataset size mismatch"

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Verify Batch Shape
    # Fetch one batch from train_loader
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Expected: (Batch, 3, 1638, 256) due to channel expansion and vertical stacking
    assert images.shape == (
        Config.batch_size,
        3,
        1638,
        256,
    ), f"Unexpected image shape: {images.shape}"
    assert targets.shape == (
        Config.batch_size,
    ), f"Unexpected target shape: {targets.shape}"

    print("Data loading logic verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization and Verification
    # ---------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = SETIModel(pretrained=Config.pretrained)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (Config.batch_size, 1), "Model output shape mismatch"

    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 4. Engine Setup and Training Loop
    # ---------------------------------------------------------
    print("\n--- Starting Training Loop ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    engine = SETIEngine(model, device, optimizer, scheduler)

    # Run Training
    # This calls train_one_epoch and validate_one_epoch
    engine.train(train_loader, val_loader, epochs=Config.epochs)

    # Verify Model Checkpoint
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"Checkpoint verified at: {best_model_path}")

    # ---------------------------------------------------------
    # 5. Prediction / Inference
    # ---------------------------------------------------------
    print("\n--- Running Prediction ---")

    # Run Inference
    engine.predict(test_loader)

    # Verify Submission File
    submission_path = os.path.join(Config.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Check if we have predictions for the test set
    # Note: In debug mode, test_ds is subsampled, so submission size matches debug_sample_size
    # or the length of the test set if it's smaller.
    expected_len = len(test_ds)
    assert (
        len(df_sub) == expected_len
    ), f"Submission length {len(df_sub)} != Test Dataset length {expected_len}"
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns"

    print("Prediction pipeline verified.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
