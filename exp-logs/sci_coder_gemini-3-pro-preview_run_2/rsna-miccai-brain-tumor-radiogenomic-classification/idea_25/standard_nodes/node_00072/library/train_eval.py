import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library import config, utils


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (str): The device to run on (cpu or cuda).
        epoch (int): The current epoch number.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    duration = time.time() - start_time

    if logger:
        logger.info(f"Epoch {epoch} Training: Loss={avg_loss}, Time={duration:.2f}s")
    else:
        print(f"Epoch {epoch} Training: Loss={avg_loss}, Time={duration:.2f}s")

    return avg_loss


def validate(model, loader, criterion, device, logger=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (str): The device to run on.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item()
            num_batches += 1

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate results
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in the batch/subset
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5
        if logger:
            logger.warning(
                "ROC AUC calculation failed (likely single class in validation). Defaulting to 0.5."
            )

    return avg_loss, auc_score


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    num_epochs=config.NUM_EPOCHS,
    device=config.DEVICE,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the full training loop with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer, optional): Optimizer. If None, creates AdamW based on config.
        num_epochs (int): Maximum number of epochs.
        device (str): Device to run on.
        patience (int): Early stopping patience.

    Returns:
        nn.Module: The best model state (loaded from best checkpoint).
    """
    # Setup Logger
    logger = utils.get_logger("training")
    logger.info(f"Starting training on device: {device}")

    # Move model to device
    model = model.to(device)

    # Setup Criterion
    criterion = nn.BCEWithLogitsLoss()

    # Setup Optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    # Tracking variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device, logger)

        # Log metrics
        # Requirement: Print full precision without rounding
        logger.info(
            f"Epoch {epoch} Summary: Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )
        print(f"Epoch {epoch} Val AUC: {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            logger.info(f"New best AUC! Saving model to {best_model_path}")
            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                filename=best_model_path,
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best AUC: {best_auc}")

    # Load best model weights before returning
    if os.path.exists(best_model_path):
        checkpoint = utils.load_checkpoint(best_model_path, model, device=device)
        logger.info(
            f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}"
        )

    return model
