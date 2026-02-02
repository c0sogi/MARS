import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import mcrmse_metric


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function (MSE).
        device (torch.device): Device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move inputs to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, dist)  # Shape: (B, 107, 3)

        # Slice predictions to scored length (first 68 positions)
        # Targets in the dataset are already sliced to 68, but we ensure consistency
        preds_scored = preds[:, : Config.SCORED_LEN, :]
        targets_scored = targets[:, : Config.SCORED_LEN, :]

        # Compute Masked MSE Loss
        loss = criterion(preds_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Norm 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Update weights
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        running_loss += loss.item() * seq.size(0)
        dataset_size += seq.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set using MCRMSE.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to run on.

    Returns:
        float: The MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, dist)  # Shape: (B, 107, 3)

            # Slice to scored length
            preds_scored = preds[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute MCRMSE
    # Note: mcrmse_metric handles the column-wise averaging
    score = mcrmse_metric(all_preds, all_targets)

    return score


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    epochs,
    patience=5,
    save_path="best_model.pth",
):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Total number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.

    Returns:
        model: The model loaded with the best weights.
    """
    best_score = float("inf")
    patience_counter = 0

    # Ensure directory exists for saving model
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )

    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)

        # Step scheduler (Cosine Annealing)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training completed. Best Val MCRMSE: {best_score:.10f}")

    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
