import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import get_logger, kl_divergence

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    epoch: int,
) -> float:
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (0-indexed).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move data to device
        eeg = batch["eeg"].to(device, non_blocking=True)
        spec = batch["spec"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model returns raw logits
        logits = model(eeg, spec)

        # Calculate Loss
        # nn.KLDivLoss expects:
        #   input: log-probabilities (LogSoftmax)
        #   target: probabilities
        #   reduction: 'batchmean' (mathematically correct for KL)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, targets, reduction="batchmean")

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Accumulate loss
        batch_size = eeg.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> tuple[float, float, np.ndarray]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (Average Loss, Average KL Metric, Predictions Array)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            spec = batch["spec"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            logits = model(eeg, spec)

            # Calculate Loss (for monitoring)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, targets, reduction="batchmean")

            # Accumulate loss
            batch_size = eeg.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for metric calculation
            # Convert logits to probabilities via Softmax
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Aggregate results
    avg_loss = running_loss / dataset_size
    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0)

    # Calculate official metric using utility function
    metric = kl_divergence(ground_truth, predictions)

    return avg_loss, metric, predictions


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
):
    """
    Orchestrates the full training loop with Early Stopping.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        scheduler: Scheduler.
        config: Configuration object.
    """
    device = config.DEVICE
    model.to(device)

    best_metric = float("inf")
    patience_counter = 0
    patience = 3  # Stop if no improvement for 3 epochs (though Config.EPOCHS is 5)

    logger.info(f"Starting training on device: {device}")

    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validation Step
        val_loss, val_metric, _ = validate(model, val_loader, device)

        # Logging (Full precision as requested)
        logger.info(
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val KL Metric: {val_metric}"
        )

        # Checkpointing & Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            save_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model improved. Saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation KL Metric: {best_metric}")
