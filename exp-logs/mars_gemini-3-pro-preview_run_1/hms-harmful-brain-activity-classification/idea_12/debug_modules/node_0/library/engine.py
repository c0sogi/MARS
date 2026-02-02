import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from library.config import Config
from library.utils import kl_divergence_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: Config,
    scaler: GradScaler,
) -> float:
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Computation device (CPU/GPU).
        epoch: Current epoch number.
        config: Configuration object.
        scaler: GradScaler for mixed precision training.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # KLDivLoss expects log-probabilities as input
    criterion = nn.KLDivLoss(reduction="batchmean")

    for batch_idx, (eeg, spec, targets) in enumerate(dataloader):
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = eeg.size(0)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type="cuda", enabled=config.USE_AMP):
            logits = model(eeg, spec)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler Step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Train Loss: {epoch_loss}")
    return epoch_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
) -> float:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for validation data.
        device: Computation device.
        config: Configuration object.

    Returns:
        The average KL Divergence score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for eeg, spec, targets in dataloader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward Pass (Mixed Precision optional but good for speed)
            with torch.amp.autocast(device_type="cuda", enabled=config.USE_AMP):
                logits = model(eeg, spec)
                probs = F.softmax(logits, dim=1)

            all_preds.append(probs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate Competition Metric
    val_score = kl_divergence_loss(y_pred, y_true)

    print(f"Validation KL Divergence: {val_score}")
    return val_score


def inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
) -> np.ndarray:
    """
    Generates predictions for the test set.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for test data.
        device: Computation device.
        config: Configuration object.

    Returns:
        Numpy array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for eeg, spec, _ in dataloader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=config.USE_AMP):
                logits = model(eeg, spec)
                probs = F.softmax(logits, dim=1)

            all_preds.append(probs.float().cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: Config,
):
    """
    Orchestrates the full training loop including checkpointing.
    """
    scaler = GradScaler(enabled=config.USE_AMP)
    best_score = float("inf")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
        )

        val_score = validate(
            model=model, dataloader=val_loader, device=device, config=config
        )

        # Checkpoint Best Model
        if val_score < best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Score: {best_score}")
