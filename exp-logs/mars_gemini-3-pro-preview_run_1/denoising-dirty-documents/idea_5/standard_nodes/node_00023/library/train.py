import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WORKING_DIR,
    SEED,
    set_seed,
    N_FOLDS,
)
from library.model import ResidualShallowUNet
from library.dataset import get_kfold_loaders
from library.utils import rmse_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for noisy, residual in loader:
        noisy = noisy.to(device)
        residual = residual.to(device)

        optimizer.zero_grad()

        # Forward pass: Predict noise residual
        pred_residual = model(noisy)

        # Loss calculated on the residual (Predicted Noise vs Actual Noise)
        loss = criterion(pred_residual, residual)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using RMSE on cleaned images.
    """
    model.eval()
    total_sse = 0.0
    total_pixels = 0

    with torch.no_grad():
        for noisy, residual_target in loader:
            noisy = noisy.to(device)
            # residual_target is needed to reconstruct the ground truth clean image
            # Target Clean = Noisy - True Residual
            target_clean = noisy - residual_target.to(device)

            # Predict residual
            pred_residual = model(noisy)

            # Reconstruct Clean Image: Clean = Noisy - Predicted Residual
            pred_clean = noisy - pred_residual

            # Clip values to valid range [0, 1]
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)
            target_clean = torch.clamp(target_clean, 0.0, 1.0)

            # Calculate Squared Error for this batch (batch size is usually 1 for val)
            diff = pred_clean - target_clean
            total_sse += torch.sum(diff**2).item()
            total_pixels += torch.numel(pred_clean)

    # Compute global RMSE
    mse = total_sse / total_pixels
    rmse = np.sqrt(mse)
    return rmse


def train_fold(
    fold_idx,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=150,
    device=DEVICE,
):
    """
    Trains a single fold with Model Checkpointing and Early Stopping.
    """
    print(f"Starting Fold {fold_idx+1}/{N_FOLDS}")

    # Initialize Model
    model = ResidualShallowUNet(n_channels=1, n_classes=1).to(device)

    # Optimizer & Scheduler
    # Using Adam with high LR and Cosine Annealing as per strategy
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    save_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")

    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Model Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        # Logging
        # Print every 50 epochs or if it's the last epoch/early stopping to reduce log spam
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            print(
                f"Fold {fold_idx+1} Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val RMSE: {val_rmse:.10f} - "
                f"Best RMSE: {best_rmse:.10f} - "
                f"Time: {time.time() - start_time:.2f}s"
            )

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} for Fold {fold_idx+1}")
            break

    print(f"Fold {fold_idx+1} Finished. Best RMSE: {best_rmse:.10f}")
    return best_rmse


def run_training(epochs=EPOCHS, n_folds=N_FOLDS, patience=150, load_cached_data=True):
    """
    Main driver function to execute K-Fold training.
    """
    # Ensure reproducibility
    set_seed(SEED)

    # Get K-Fold DataLoaders
    # This handles data loading and caching internally
    loaders = get_kfold_loaders(n_folds=n_folds, load_cached_data=load_cached_data)

    fold_scores = []

    for fold_idx, (train_loader, val_loader) in enumerate(loaders):
        score = train_fold(
            fold_idx,
            train_loader,
            val_loader,
            epochs=epochs,
            patience=patience,
            device=DEVICE,
        )
        fold_scores.append(score)

    print("-" * 30)
    print(f"All {n_folds} folds completed.")
    print(f"Average Best RMSE: {np.mean(fold_scores):.10f}")
    print("-" * 30)
