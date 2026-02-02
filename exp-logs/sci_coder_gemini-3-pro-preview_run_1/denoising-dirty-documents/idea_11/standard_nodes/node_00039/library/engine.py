import torch
import torch.nn.functional as F
import numpy as np
from library.utils import save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The neural network model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: 'cuda' or 'cpu'.

    Returns:
        float: The average MSE for the epoch.
    """
    model.train()
    total_sse = 0.0
    total_pixels = 0

    for batch in dataloader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()

        outputs = model(noisy)

        # Optimization objective: Mean MSE
        loss = F.mse_loss(outputs, clean, reduction="mean")
        loss.backward()
        optimizer.step()

        # Metric tracking: Sum of Squared Errors
        # We recompute or scale to get sum for accurate global average
        batch_sse = F.mse_loss(outputs, clean, reduction="sum").item()
        total_sse += batch_sse
        total_pixels += clean.numel()

    avg_mse = total_sse / total_pixels
    return avg_mse


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        float: The global RMSE over the validation set.
    """
    model.eval()
    total_sse = 0.0
    total_pixels = 0

    with torch.no_grad():
        for batch in dataloader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            outputs = model(noisy)

            # Aggregate Sum of Squared Errors
            batch_sse = F.mse_loss(outputs, clean, reduction="sum").item()
            total_sse += batch_sse
            total_pixels += clean.numel()

    mse = total_sse / total_pixels
    rmse = np.sqrt(mse)
    return rmse


def fit_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, save_path
):
    """
    Runs the full training loop, prioritizing full convergence.
    Saves the best model checkpoint based on validation RMSE.

    Args:
        model: The neural network model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: 'cuda' or 'cpu'.
        epochs: Total number of epochs to train.
        save_path: File path to save the best model checkpoint.

    Returns:
        float: The best validation RMSE achieved.
    """
    best_val_rmse = float("inf")

    for epoch in range(epochs):
        # Training Step
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        train_rmse = np.sqrt(train_mse)

        # Validation Step
        val_rmse = evaluate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | Train RMSE: {train_rmse} | Val RMSE: {val_rmse}"
        )

        # Checkpointing
        # We save the best model found during the convergence process
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_rmse": float(val_rmse),
                },
                save_path,
            )

    return best_val_rmse
