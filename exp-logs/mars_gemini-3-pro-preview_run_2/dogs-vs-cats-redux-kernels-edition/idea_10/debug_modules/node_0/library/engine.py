import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import get_logger, calculate_log_loss, save_checkpoint

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The training data loader.
        device (torch.device): The device to use for training.
        epoch (int): The current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(data_loader)

    # BCEWithLogitsLoss is suitable for binary classification with soft targets (Mixup/CutMix)
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        # Model outputs are logits (batch_size, 1) or (batch_size, num_classes)
        outputs = model(images)

        # Ensure outputs are flattened to match targets shape if necessary
        if outputs.shape != targets.shape:
            outputs = outputs.view(targets.shape)

        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to use for evaluation.

    Returns:
        float: Log Loss on the validation set.
        np.ndarray: Predicted probabilities.
        np.ndarray: Ground truth labels.
    """
    model.eval()
    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            # Validation targets are standard (0 or 1), not mixed
            targets = targets.to(device)

            outputs = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Flatten if necessary
            probs = probs.view(-1)
            targets = targets.view(-1)

            preds.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds)
    targets_list = np.concatenate(targets_list)

    # Calculate metric
    val_loss = calculate_log_loss(targets_list, preds)

    return val_loss, preds, targets_list


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    checkpoint_path,
):
    """
    Main training loop with Early Stopping and Scheduler management.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Total number of epochs.
        patience: Early stopping patience.
        checkpoint_path: Path to save the best model.

    Returns:
        tuple: (best_val_loss, best_predictions, best_targets)
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_preds = None
    best_targets = None

    logger.info(f"Starting training for {epochs} epochs with patience {patience}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Evaluate
        val_loss, val_preds, val_targets = evaluate(model, val_loader, device)

        # Step Scheduler (Cosine Annealing is typically stepped per epoch)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Time: {elapsed:.2f}s - "
            f"LR: {current_lr:.2e} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Log Loss: {val_loss:.15f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds
            best_targets = val_targets
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Validation loss improved. Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation Log Loss: {best_val_loss:.15f}")

    return best_val_loss, best_preds, best_targets
