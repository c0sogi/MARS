import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import UNet
from library.dataset import get_dataloaders


def train_one_seed(seed, train_loader, val_loader):
    """
    Trains a single U-Net model for a specific seed to full convergence.

    Args:
        seed (int): Random seed for initialization and training.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
    """
    print(f"\n{'='*40}")
    print(f"Starting training for Seed: {seed}")
    print(f"{'='*40}")

    # 1. Set Seed for Reproducibility
    set_seed(seed)

    # 2. Initialize Model
    device = torch.device(Config.DEVICE)
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS)
    model.to(device)

    # 3. Setup Optimization
    # Loss: MSE on inverted signal (background=0, text=1)
    # This aligns with the Signal Inversion strategy
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Cosine Annealing to 0 over NUM_EPOCHS
    # Decoupled from validation performance to ensure full convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=0.0
    )

    # 4. Training Loop
    best_val_rmse = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for noisy_imgs, clean_imgs, _ in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()

            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches if num_batches > 0 else 0.0

        # Update Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # --- Validation Phase ---
        # Validate periodically to monitor progress
        if (epoch + 1) % 10 == 0 or (epoch + 1) == Config.NUM_EPOCHS:
            model.eval()
            val_mse_accum = 0.0
            total_pixels = 0

            with torch.no_grad():
                for noisy_imgs, clean_imgs, _ in val_loader:
                    noisy_imgs = noisy_imgs.to(device)
                    clean_imgs = clean_imgs.to(device)

                    outputs = model(noisy_imgs)

                    # Calculate squared error sum
                    # RMSE is calculated globally over all pixels
                    # Note: RMSE on inverted signal is identical to RMSE on original signal
                    diff = outputs - clean_imgs
                    val_mse_accum += torch.sum(diff**2).item()
                    total_pixels += clean_imgs.numel()

            val_rmse = (
                np.sqrt(val_mse_accum / total_pixels) if total_pixels > 0 else 0.0
            )

            print(
                f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] "
                f"LR: {current_lr:.8f} "
                f"Train Loss: {avg_train_loss:.10f} "
                f"Val RMSE: {val_rmse:.10f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse

    # 5. Save Final Model
    # Strategy dictates saving the fully converged model at the end of training
    save_path = Config.get_model_path(seed)
    print(f"Saving converged model to {save_path}...")

    save_checkpoint(
        {
            "epoch": Config.NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rmse": val_rmse,
            "config": {"seed": seed, "invert_signal": Config.INVERT_SIGNAL},
        },
        save_path,
    )

    print(
        f"Training finished for Seed {seed}. Best Val RMSE tracked: {best_val_rmse:.10f}"
    )


def train_ensemble():
    """
    Orchestrates the training of the entire ensemble defined in Config.ENSEMBLE_SEEDS.
    Initializes directories, loads data once, and iterates through all seeds.
    """
    # Ensure working directories exist
    Config.initialize()

    # Load DataLoaders (Cached)
    # We load them once; the random sampling in train_loader will be re-seeded
    # inside train_one_seed via set_seed -> worker_init_fn logic implicitly
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Iterate through ensemble seeds
    for seed in Config.ENSEMBLE_SEEDS:
        train_one_seed(seed, train_loader, val_loader)
