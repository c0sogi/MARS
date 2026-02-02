import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, save_checkpoint
from library.data_loader import get_dataloaders
from library.model import DGC_BiLSTM


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function (WeightedL1Loss).
        device: The computing device (CPU or GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (features, targets, u_out) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)
        u_out = u_out.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(features)

        # Calculate loss
        loss = criterion(preds, targets, u_out)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()

        # Accumulate loss (multiply by batch size to get total, then divide later)
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Calculates MAE specifically for the inspiratory phase (u_out == 0).

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: The loss function (used for logging generic loss, though MAE is primary).
        device: The computing device.

    Returns:
        dict: Dictionary containing 'loss' and 'mae_inspiratory'.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    total_ae_insp = 0.0
    count_insp = 0

    with torch.no_grad():
        for features, targets, u_out in loader:
            features = features.to(device)
            targets = targets.to(device)
            u_out = u_out.to(device)

            preds = model(features)

            # 1. Generic Weighted Loss
            loss = criterion(preds, targets, u_out)
            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # 2. Inspiratory MAE Calculation
            # Identify inspiratory phase indices (u_out == 0)
            insp_mask = u_out == 0

            if insp_mask.sum() > 0:
                # Absolute error
                abs_error = torch.abs(preds - targets)
                # Select only inspiratory errors
                insp_error = abs_error[insp_mask]

                total_ae_insp += insp_error.sum().item()
                count_insp += insp_mask.sum().item()

    avg_loss = running_loss / total_samples
    # Avoid division by zero if for some reason a batch has no inspiratory phase (unlikely)
    mae_insp = total_ae_insp / count_insp if count_insp > 0 else 0.0

    return {"loss": avg_loss, "mae_inspiratory": mae_insp}


def run_training(debug=Config.DEBUG):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs with a smaller subset of data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = DGC_BiLSTM().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Stretched-Horizon Protocol: T_max matches total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )

    criterion = WeightedL1Loss()

    # 5. Training Loop
    best_mae = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae_inspiratory"]

        # Step Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MAE (Insp): {val_mae}"
        )

        # Early Stopping & Checkpointing
        if val_mae < best_mae:
            print(
                f"Validation MAE improved from {best_mae} to {val_mae}. Saving checkpoint..."
            )
            best_mae = val_mae
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_mae": best_mae,
                },
                is_best=True,
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MAE (Inspiratory): {best_mae}")
