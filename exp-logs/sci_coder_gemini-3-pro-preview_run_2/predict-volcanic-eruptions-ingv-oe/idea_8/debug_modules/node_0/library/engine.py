import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, save_checkpoint


def train_fn(dataloader, model, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        dataloader (DataLoader): Training data loader.
        model (nn.Module): The model to train.
        criterion (loss_fn): The loss function (e.g., L1Loss).
        optimizer (Optimizer): The optimizer.
        device (str): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for data in dataloader:
        spectrogram = data["spectrogram"].to(device)
        tabular = data["tabular"].to(device)
        targets = data["target"].to(device)

        # Forward pass
        outputs = model(spectrogram, tabular)

        # Ensure outputs and targets have the same shape
        # outputs: [Batch, 1], targets: [Batch]
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), spectrogram.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        dataloader (DataLoader): Validation data loader.
        model (nn.Module): The model to evaluate.
        criterion (loss_fn): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for data in dataloader:
            spectrogram = data["spectrogram"].to(device)
            tabular = data["tabular"].to(device)
            targets = data["target"].to(device)

            outputs = model(spectrogram, tabular)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), spectrogram.size(0))

    return loss_meter.avg


def predict_fn(dataloader, model, device):
    """
    Generates predictions for the test set.

    Args:
        dataloader (DataLoader): Test data loader.
        model (nn.Module): The trained model.
        device (str): Device to run inference on.

    Returns:
        tuple: (segment_ids, predictions) as numpy arrays.
    """
    model.eval()
    predictions = []
    segment_ids = []

    with torch.no_grad():
        for data in dataloader:
            spectrogram = data["spectrogram"].to(device)
            tabular = data["tabular"].to(device)

            # segment_id is part of the batch
            ids = data["segment_id"]

            outputs = model(spectrogram, tabular)
            outputs = outputs.squeeze(1)

            predictions.extend(outputs.cpu().numpy())
            segment_ids.extend(ids.numpy())

    return np.array(segment_ids), np.array(predictions)


def run_training(model, train_loader, val_loader, device, target_scaler):
    """
    Orchestrates the training process with Warmup, Scheduler, and Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to compute on.
        target_scaler (TargetScaler): Scaler object to compute unscaled metrics.

    Returns:
        float: The best validation loss achieved.
    """
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler (ReduceLROnPlateau)
    # Note: We handle Warmup manually in the loop
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=False,
    )

    # Loss Function (MAE)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")
    print(f"Warmup Epochs: {Config.WARMUP_EPOCHS}, Patience: {Config.PATIENCE}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # ----------------------------------------------------------------------
        # Learning Rate Warmup Logic
        # ----------------------------------------------------------------------
        if epoch < Config.WARMUP_EPOCHS:
            # Linear Warmup: from 0 to Initial LR
            lr_scale = (epoch + 1) / Config.WARMUP_EPOCHS
            current_lr = Config.LEARNING_RATE * lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------------------------
        # Train & Validate
        # ----------------------------------------------------------------------
        train_loss = train_fn(train_loader, model, criterion, optimizer, device)
        val_loss = eval_fn(val_loader, model, criterion, device)

        # ----------------------------------------------------------------------
        # Scheduler Step (Post-Warmup)
        # ----------------------------------------------------------------------
        if epoch >= Config.WARMUP_EPOCHS:
            scheduler.step(val_loss)

        # ----------------------------------------------------------------------
        # Metrics & Logging
        # ----------------------------------------------------------------------
        elapsed = time.time() - start_time

        # Calculate Unscaled MAE for interpretability
        # MAE_unscaled = MAE_scaled * std
        val_mae_unscaled = val_loss * target_scaler.std

        print(
            f"Epoch {epoch + 1:02d}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss (Scaled): {train_loss:.8f} | "
            f"Val Loss (Scaled): {val_loss:.8f} | "
            f"Val MAE (Original): {val_mae_unscaled:.4f}"
        )

        # ----------------------------------------------------------------------
        # Early Stopping & Checkpointing
        # ----------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    print(f"Training finished. Best Val Loss (Scaled): {best_val_loss}")
    return best_val_loss
