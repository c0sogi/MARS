import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import seed_everything
from library.dataset import TextDenoisingDataset
from library.model import CACResUNet
from library.inference import predict_tiled


def run_training(
    debug=Config.DEBUG,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Executes the training pipeline for the CAC-ResUNet model.

    Args:
        debug (bool): If True, trains on a small subset of data.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience epochs.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    print(
        f"Initializing training [Debug={debug}, Epochs={num_epochs}, Patience={patience}]"
    )
    seed_everything()

    device = Config.DEVICE

    # --- 1. Data Loading ---
    print("Loading datasets...")
    train_dataset = TextDenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        mode="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = TextDenoisingDataset(
        metadata_path=Config.VAL_METADATA, mode="val", load_cached_data=load_cached_data
    )

    # Handle Debug Mode
    if debug:
        print(f"Debug mode active: Using subset of {Config.DEBUG_SUBSET_SIZE} images.")
        # Calculate indices for the subset
        # Train dataset length is num_images * patches_per_image
        train_limit = Config.DEBUG_SUBSET_SIZE * Config.PATCHES_PER_IMAGE
        train_indices = range(min(len(train_dataset), train_limit))
        train_dataset = Subset(train_dataset, train_indices)

        # Val dataset length is num_images
        val_indices = range(min(len(val_dataset), Config.DEBUG_SUBSET_SIZE))
        val_dataset = Subset(val_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation processes full images individually
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Model & Optimization Setup ---
    print("Setting up model and optimizer...")
    model = CACResUNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.ETA_MIN
    )

    # Loss Function: MSE on the noise residual
    criterion = nn.MSELoss()

    # --- 3. Training Loop ---
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        # --- Train Phase ---
        model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for batch_idx, (noisy_patches, clean_patches) in enumerate(train_loader):
            noisy_patches = noisy_patches.to(device)
            clean_patches = clean_patches.to(device)

            # Target is the noise residual
            target_noise = noisy_patches - clean_patches

            optimizer.zero_grad()

            # Forward pass
            pred_noise = model(noisy_patches)

            # Calculate loss
            loss = criterion(pred_noise, target_noise)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches if num_batches > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        total_squared_error = 0.0
        total_pixels = 0

        with torch.no_grad():
            for noisy_img, clean_img, _ in val_loader:
                noisy_img = noisy_img.to(device)
                clean_img = clean_img.to(device)

                # Predict noise using tiled inference to handle full resolution
                pred_noise = predict_tiled(
                    model,
                    noisy_img,
                    patch_size=Config.PATCH_SIZE,
                    overlap=Config.TILE_OVERLAP,
                )

                # Reconstruct clean image: Clean = Noisy - Noise
                pred_clean = noisy_img - pred_noise

                # Clamp to valid pixel range [0, 1]
                pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

                # Accumulate squared errors for global RMSE
                se = (clean_img - pred_clean) ** 2
                total_squared_error += se.sum().item()
                total_pixels += clean_img.numel()

        # Calculate Global RMSE
        val_rmse = (
            np.sqrt(total_squared_error / total_pixels) if total_pixels > 0 else 0.0
        )

        # Update Scheduler
        scheduler.step()

        # --- Logging & Checkpointing ---
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.10f} | Val RMSE: {val_rmse:.10f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  > New best model saved! (RMSE: {best_rmse:.10f})")
        else:
            patience_counter += 1
            print(f"  > No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation RMSE: {best_rmse:.10f}")
