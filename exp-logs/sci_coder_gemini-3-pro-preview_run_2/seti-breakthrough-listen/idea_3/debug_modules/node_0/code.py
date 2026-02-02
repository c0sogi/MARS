import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import SETIDataset, get_transforms
from library.model import SETIModel
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # --------------------------------------------------------------------------
    print("Setting up demonstration configuration...")

    # Override Config for speed and demonstration purposes
    Config.epochs = 1
    Config.batch_size = 8  # Small batch size for demo
    Config.print_freq = 2  # Frequent logging for small subset
    Config.debug = True

    # Disable TTA for faster inference in demo (optional, but kept True in Config usually)
    # We will keep Config.tta = True to demonstrate the logic, but on a small set it's fast.

    # Set output directory for demo
    demo_output_dir = "./working/demo_run"
    os.makedirs(demo_output_dir, exist_ok=True)

    # Set device
    device = Config.device
    print(f"Device: {device}")

    # Seed for reproducibility
    seed_everything(Config.seed)

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("\nPreparing data...")

    # Load metadata
    # We use a very small subset (e.g., 32 samples) to ensure the script runs quickly.
    train_df = pd.read_csv(Config.train_csv).iloc[:32].reset_index(drop=True)
    val_df = pd.read_csv(Config.val_csv).iloc[:16].reset_index(drop=True)
    test_df = pd.read_csv(Config.test_csv).iloc[:16].reset_index(drop=True)

    print(f"Train subset size: {len(train_df)}")
    print(f"Val subset size: {len(val_df)}")
    print(f"Test subset size: {len(test_df)}")

    # Initialize Datasets
    train_dataset = SETIDataset(train_df, transform=get_transforms("train"))
    val_dataset = SETIDataset(val_df, transform=get_transforms("valid"))
    test_dataset = SETIDataset(test_df, transform=get_transforms("test"))

    # Verify Dataset Output
    # Shape expected: (3, 1638, 256) after stacking and channel expansion
    sample_img, sample_target = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target: {sample_target}")

    assert sample_img.shape == (
        3,
        1638,
        256,
    ), f"Expected image shape (3, 1638, 256), got {sample_img.shape}"
    assert isinstance(sample_target, torch.Tensor), "Target should be a tensor"

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\nInitializing model...")
    # pretrained=False to avoid downloading weights during this timed run
    model = SETIModel(pretrained=False)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=Config.T_mult, eta_min=Config.min_lr
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop (1 Epoch)
    # --------------------------------------------------------------------------
    print("\nStarting training...")
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=0
    )
    print(f"Epoch 1 Training Loss: {train_loss:.4f}")

    # Assert loss is valid
    assert not np.isnan(train_loss), "Training loss is NaN"

    # --------------------------------------------------------------------------
    # 5. Validation Loop
    # --------------------------------------------------------------------------
    print("\nStarting validation...")
    val_loss, val_auc = valid_one_epoch(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # --------------------------------------------------------------------------
    # 6. Inference
    # --------------------------------------------------------------------------
    print("\nStarting inference...")
    predictions = inference_fn(model, test_loader, device)

    # --------------------------------------------------------------------------
    # 7. Verification and Submission Generation
    # --------------------------------------------------------------------------
    print("\nVerifying results...")

    # Check predictions shape
    assert len(predictions) == len(
        test_df
    ), f"Prediction count ({len(predictions)}) matches test set size ({len(test_df)})"

    # Check probability range
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions contain values outside [0, 1]"

    # Create submission dataframe
    test_df["target"] = predictions
    submission_path = os.path.join(demo_output_dir, "submission.csv")
    test_df[["id", "target"]].to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
