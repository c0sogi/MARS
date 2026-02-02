import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, Optional

from library.config import Config
from library.utils import AverageMeter
from library.transforms import MixUp


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mixup_fn: Optional[MixUp] = None,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to run training on.
        epoch: Current epoch number.
        mixup_fn: MixUp augmentation instance.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # KL Div Loss expects input as log-probabilities and target as probabilities
    criterion = nn.KLDivLoss(reduction="batchmean")

    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        # Apply MixUp
        if mixup_fn is not None:
            data, targets = mixup_fn(data, targets)

        optimizer.zero_grad()

        # Forward pass
        # Model expects x_eeg and x_spec
        logits = model(data["eeg"], data["spec"])

        # Compute Loss
        # Apply LogSoftmax to logits for KLDivLoss
        log_probs = F.log_softmax(logits, dim=1)
        loss = criterion(log_probs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        loss_meter.update(loss.item(), targets.size(0))

    print(f"Epoch {epoch} Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: Validation dataloader.
        device: Device to run evaluation on.

    Returns:
        Average validation loss.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device, non_blocking=True)

            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model(data["eeg"], data["spec"])

            # Compute Loss
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            loss_meter.update(loss.item(), targets.size(0))

    print(f"Validation Loss: {loss_meter.avg}")
    return loss_meter.avg


def inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = Config.SUBMISSION_PATH,
) -> None:
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: The neural network model.
        dataloader: Test dataloader.
        device: Device to run inference on.
        save_path: Path to save the submission CSV.
    """
    model.eval()

    all_preds = []
    all_eeg_ids = []

    print("Starting Inference...")

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device, non_blocking=True)

            # Forward pass
            logits = model(data["eeg"], data["spec"])

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)

            # Store results
            all_preds.append(probs.cpu().numpy())
            all_eeg_ids.extend(data["eeg_id"].cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission_df.insert(0, "eeg_id", all_eeg_ids)

    # Ensure eeg_id is integer
    submission_df["eeg_id"] = submission_df["eeg_id"].astype(int)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")
