import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint,
    save_submission,
    load_image_with_cache,
    load_metadata,
)
from library.dataset import get_dataloaders
from library.model import WaveCACResUNet, train_one_epoch, validate, predict_tiled


def run_training(
    data_dir: str = "./input",
    work_dir: str = "./working/idea_10",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 10,
    train_samples_per_epoch: int = 100,
):
    """
    Orchestrates the training and inference pipeline for the Wave-CAC-ResUNet.
    """
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()

    # Create working directories
    os.makedirs(work_dir, exist_ok=True)
    cache_dir = os.path.join(work_dir, "cache")
    checkpoint_path = os.path.join(work_dir, "best_model.pth")
    submission_path = os.path.join(work_dir, "submission.csv")

    print(f"Running training on device: {device}")
    print(f"Working directory: {work_dir}")
    print(f"Cache directory: {cache_dir}")

    # 2. Prepare DataLoaders
    # High-density sampling is controlled by train_samples_per_epoch
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=4,
        patch_size=128,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=1,
    )

    # 3. Initialize Model and Optimization
    model = WaveCACResUNet(in_channels=1, base_filters=64).to(device)

    # Loss: MSE on the noise residual
    criterion = nn.MSELoss()

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    best_rmse = float("inf")
    no_improve_epochs = 0

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step()

        # Print metrics (full precision)
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val RMSE: {val_rmse}"
        )

        # Checkpointing and Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                },
                checkpoint_path,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 5. Inference and Submission
    print("Training finished. Starting inference on test set...")

    # Load the best model weights
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()

    # Load Test Metadata
    try:
        df_test = load_metadata("test")
    except FileNotFoundError:
        print("Test metadata not found. Cannot generate submission.")
        return

    predictions = {}
    test_cache_dir = os.path.join(cache_dir, "test")

    # Iterate over test images
    for _, row in df_test.iterrows():
        img_id = str(row["id"])
        feature_rel_path = row["feature_path"]
        feature_full_path = os.path.join(data_dir, feature_rel_path)

        # Cache path for this test image
        img_cache_path = os.path.join(test_cache_dir, f"{img_id}_noisy.npy")

        # Load image (using cache mechanism)
        img_np = load_image_with_cache(feature_full_path, img_cache_path)

        # Perform Test-Time Augmentation (TTA)
        # Strategy: Original, H-Flip, V-Flip, Rot90
        tta_preds = []

        # 1. Original
        pred_orig = predict_tiled(model, img_np, device)
        tta_preds.append(pred_orig)

        # 2. Horizontal Flip
        img_h = np.flip(img_np, axis=1)
        pred_h = predict_tiled(model, img_h, device)
        tta_preds.append(np.flip(pred_h, axis=1))

        # 3. Vertical Flip
        img_v = np.flip(img_np, axis=0)
        pred_v = predict_tiled(model, img_v, device)
        tta_preds.append(np.flip(pred_v, axis=0))

        # 4. Rotate 90 (k=1)
        img_r = np.rot90(img_np, k=1)
        pred_r = predict_tiled(model, img_r, device)
        # Inverse rotate (k=-1 or k=3)
        tta_preds.append(np.rot90(pred_r, k=-1))

        # Average predictions
        final_pred = np.mean(tta_preds, axis=0)

        predictions[img_id] = final_pred

    # Save submission
    save_submission(predictions, submission_path)
    print(f"Submission saved to {submission_path}")
