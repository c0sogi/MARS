import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from library.config import Config
from library.utils import get_logger, compute_metrics

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch using mixed precision and gradient accumulation.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training set.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (optional).
        device: The device to run training on.
        scaler: GradScaler for mixed precision training.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    total_loss = 0.0
    dataset_size = 0

    # Define loss function (supports soft targets)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    num_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        response_mask = batch["response_mask"].to(device)
        scalars = batch["scalars"].to(device)
        targets = batch["target"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Context
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_FP16):
            # Forward pass
            logits = model(input_ids, attention_mask, response_mask, scalars)
            loss = criterion(logits, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass (scaled)
        scaler.scale(loss).backward()

        # Track loss (scale back up for logging to represent actual loss)
        current_loss = loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        total_loss += current_loss * batch_size
        dataset_size += batch_size

        # Gradient Accumulation Step
        # Step if accumulation limit reached OR if it's the last batch
        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0 or (
            step + 1
        ) == num_steps:
            # Unscale gradients
            scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Zero gradients
            optimizer.zero_grad()

    avg_loss = total_loss / dataset_size
    logger.info(f"Epoch {epoch} - Training Loss: {avg_loss}")

    return avg_loss


def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation set.
        device: The device to run evaluation on.

    Returns:
        Tuple[float, Dict]: Average validation loss and a dictionary of metrics.
    """
    model.eval()

    total_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_mask = batch["response_mask"].to(device)
            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            batch_size = input_ids.size(0)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_FP16):
                logits = model(input_ids, attention_mask, response_mask, scalars)
                loss = criterion(logits, targets)

            total_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Softmax for probabilities
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / dataset_size

    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"Validation Loss: {avg_loss}")
    logger.info(f"Validation Metrics: {metrics}")

    return avg_loss, metrics


def predict(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> np.ndarray:
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the test set.
        device: The device to run inference on.

    Returns:
        np.ndarray: Predicted probabilities of shape (N, 3).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_mask = batch["response_mask"].to(device)
            scalars = batch["scalars"].to(device)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_FP16):
                logits = model(input_ids, attention_mask, response_mask, scalars)

            # Apply Softmax
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
