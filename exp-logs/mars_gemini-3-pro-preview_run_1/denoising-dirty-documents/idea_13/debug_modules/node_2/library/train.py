import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import UNet
from library.utils import seed_everything, rmse_score


def train_one_seed(
    seed,
    train_ds,
    val_ds,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
    patience=100,
):
    """
    Trains a U-Net model for a specific seed with the defined configuration.

    Args:
        seed (int): Random seed for reproducibility.
        train_ds (Dataset): Training dataset.
        val_ds (Dataset): Validation dataset.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        device (str): Computation device ('cuda' or 'cpu').
        num_workers (int): Number of worker threads for data loading.
        patience (int): Early stopping patience epochs.

    Returns:
        float: Best validation RMSE achieved.
    """
    # 1. Set seed for reproducibility
    seed_everything(seed)

    # 2. Create DataLoaders
    # Pin memory speeds up host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Initialize Model
    model = UNet().to(device)

    # 4. Optimizer and Scheduler
    # Using Adam with high LR and Cosine Annealing as per strategy
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Loss Function
    # MSE on inverted signal (0=bg, 1=text)
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    model_path = Config.get_model_path(seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Starting training for seed {seed}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_steps = 0

        for noisy, clean, _ in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy)

            # Compute loss
            loss = criterion(outputs, clean)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_mse_sum = 0.0
        val_pixel_count = 0

        with torch.no_grad():
            for noisy, clean, _ in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                outputs = model(noisy)

                # Calculate squared error for RMSE
                # RMSE is calculated on pixel intensities.
                # Since data is inverted, the magnitude of error |y - y_hat| is same as original.
                diff = outputs - clean
                val_mse_sum += torch.sum(diff**2).item()
                val_pixel_count += clean.numel()

        # Compute global RMSE for the epoch
        val_rmse = (
            np.sqrt(val_mse_sum / val_pixel_count) if val_pixel_count > 0 else 0.0
        )

        # --- Scheduler Step ---
        scheduler.step()

        # --- Logging ---
        # Print full precision as requested
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss} - Val RMSE: {val_rmse}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training finished for seed {seed}. Best Val RMSE: {best_val_rmse}")
    return best_val_rmse
