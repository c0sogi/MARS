import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.model import EZ_ResDnCNN
from library.dataset import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for i, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        # The model predicts the noise residual.
        # Target Noise = Noisy Input - Clean Target
        noise_target = inputs - targets

        optimizer.zero_grad()

        noise_pred = model(inputs)
        loss = criterion(noise_pred, noise_target)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples if total_samples > 0 else 0.0


def validate(model, dataloader, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to evaluate on.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        tuple: (Average Validation Loss, Validation RMSE)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Target Noise
            noise_target = inputs - targets

            # Predict Noise
            noise_pred = model(inputs)
            loss = criterion(noise_pred, noise_target)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Reconstruct Clean Image: Clean = Noisy - Predicted Noise
            clean_pred = inputs - noise_pred

            # Clamp pixel values to valid range [0, 1] for metric calculation
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            all_preds.append(clean_pred.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        val_rmse = calculate_rmse(all_targets, all_preds)
    else:
        val_rmse = 0.0

    return epoch_loss, val_rmse


def run_training(
    model_name,
    seed=Config.SEED,
    epochs=Config.NUM_EPOCHS,
    load_cached_data=True,
    debug=False,
):
    """
    Orchestrates the training process for a single model instance.

    Args:
        model_name (str): Identifier for saving the model checkpoint.
        seed (int): Random seed for this training run.
        epochs (int): Number of epochs to train.
        load_cached_data (bool): Whether to load pre-processed patches from disk.
        debug (bool): If True, limits the number of batches for quick testing.

    Returns:
        float: The best validation RMSE achieved.
    """
    seed_everything(seed)
    device = torch.device(Config.DEVICE)

    # --- Data Loading ---
    # Using the library function to get dataloaders
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    # Determine batch limit for debugging
    max_batches = 10 if debug else None

    # --- Model Initialization ---
    model = EZ_ResDnCNN().to(device)

    # --- Optimization ---
    # MSE Loss on the noise residual
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # --- Training Loop ---
    best_rmse = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, f"{model_name}.pth")

    print(f"Starting training: {model_name} | Device: {device} | Epochs: {epochs}")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches
        )
        val_loss, val_rmse = validate(model, val_loader, criterion, device, max_batches)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSE: {val_rmse} | "
            f"Time: {elapsed:.2f}s"
        )

        # --- Early Stopping & Checkpointing ---
        # Check if validation RMSE improved by at least MIN_DELTA
        if val_rmse < best_rmse - Config.MIN_DELTA:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training finished for {model_name}. Best RMSE: {best_rmse}")
    return best_rmse
