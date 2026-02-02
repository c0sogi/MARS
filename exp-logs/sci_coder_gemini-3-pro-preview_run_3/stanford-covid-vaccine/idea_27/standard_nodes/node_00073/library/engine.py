import torch
import torch.nn as nn
import time
import os
from library.config import Config
from library.utils import MCRMSE


def mcrmse_loss(preds, targets, scored_len=Config.PRED_LEN):
    """
    Calculates the MCRMSE loss for the scored sequence positions.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
        targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
        scored_len (int): Number of positions to score (default 68).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Slice to scored length to ignore padding in the loss calculation
    preds_sliced = preds[:, :scored_len, :]
    targets_sliced = targets[:, :scored_len, :]

    # Calculate MSE per column (averaging over Batch and Sequence dimensions)
    # Shape: (5,)
    mse = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse + 1e-8)  # Add epsilon for numerical stability

    # Filter for scored columns only
    if hasattr(Config, "SCORED_COLS_INDICES"):
        rmse = rmse[Config.SCORED_COLS_INDICES]

    # Calculate Mean of RMSEs across the 5 targets
    loss = torch.mean(rmse)

    return loss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): Device to run on ('cuda' or 'cpu').
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move inputs to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_masks)

        # Compute Loss (Unweighted MCRMSE on scored positions)
        loss = mcrmse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to run on.

    Returns:
        float: Global MCRMSE score.
    """
    model.eval()
    metric = MCRMSE()

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(features, pair_indices, pair_masks)

            # Update metric accumulator
            # The MCRMSE class handles slicing to Config.PRED_LEN internally
            metric.update(outputs, targets)

    # Compute global metric over the entire dataset
    score = metric.compute()
    return score


def fit(model, train_loader, val_loader, device=Config.DEVICE):
    """
    Orchestrates the training process with Early Stopping and Scheduler.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to run on.

    Returns:
        nn.Module: The trained model with the best weights loaded.
    """
    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validation Step
        val_score = evaluate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print metrics (Full precision for val_score)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping Logic
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            # Save Best Model
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_score})")
        else:
            patience_counter += 1
            print(
                f"  >>> EarlyStopping Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # Load the best model weights before returning
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    return model
