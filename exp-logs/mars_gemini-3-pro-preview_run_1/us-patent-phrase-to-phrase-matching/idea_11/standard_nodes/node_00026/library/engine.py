import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from library.config import Config
from library.utils import compute_metrics


def train_fn(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to run training on (CPU/GPU).
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    final_loss = 0.0
    count = 0

    # Initialize GradScaler for Mixed Precision if enabled
    scaler = torch.amp.GradScaler("cuda", enabled=Config.fp16)

    for batch_idx, data in enumerate(data_loader):
        # Move inputs to device
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()

        # Mixed Precision Context
        with torch.amp.autocast("cuda", enabled=Config.fp16):
            outputs = model(**data)
            loss = outputs.loss

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer and Scheduler steps
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        final_loss += loss.item()
        count += 1

    avg_loss = final_loss / count
    print(f"Epoch {epoch+1} | Training Loss: {avg_loss}")

    return avg_loss


def valid_fn(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        Tuple[float, Dict[str, float]]: Average validation loss and metrics dictionary.
    """
    model.eval()
    final_loss = 0.0
    count = 0
    preds = []
    labels = []

    with torch.no_grad():
        for data in data_loader:
            for k, v in data.items():
                data[k] = v.to(device)

            # Mixed Precision Context (optional for inference but good for consistency)
            with torch.amp.autocast("cuda", enabled=Config.fp16):
                outputs = model(**data)
                loss = outputs.loss

            final_loss += loss.item()
            count += 1

            # Collect predictions and labels
            # outputs.logits shape: (batch_size, 1)
            preds.append(outputs.logits.detach().cpu().numpy())
            labels.append(data["labels"].detach().cpu().numpy())

    avg_loss = final_loss / count

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Compute metrics (Pearson)
    metrics = compute_metrics((preds, labels))

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Pearson: {metrics['pearson']}")

    return avg_loss, metrics


def inference_fn(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for test data.
        device: Device to run inference on.

    Returns:
        np.ndarray: Array of predictions.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in data_loader:
            for k, v in data.items():
                data[k] = v.to(device)

            with torch.amp.autocast("cuda", enabled=Config.fp16):
                outputs = model(**data)

            # outputs.logits shape: (batch_size, 1)
            preds.append(outputs.logits.detach().cpu().numpy())

    # Concatenate and flatten
    predictions = np.concatenate(preds, axis=0).flatten()

    return predictions.astype(np.float32)
