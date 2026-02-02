import os
import time
import torch
import numpy as np
from library.config import Config
from library.loss import MaskedMSELoss, mcrmse


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function (MaskedMSELoss).
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, dist)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (stabilize BiGRU)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (average_loss, mcrmse_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Move inputs to device
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(seq, loop, dist)

            # Compute loss (for tracking purposes)
            loss = criterion(preds, targets)
            running_loss += loss.item()
            num_batches += 1

            # Store predictions and targets for MCRMSE calculation
            # Move to CPU to save GPU memory during accumulation
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Aggregate results
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        full_preds = torch.cat(all_preds, dim=0)
        full_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE
        score = mcrmse(full_preds, full_targets)
    else:
        score = 0.0

    return avg_loss, score


def fit(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Orchestrates the training and validation process.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device string.
        epochs: Number of epochs to train.

    Returns:
        float: Best validation MCRMSE score achieved.
    """
    criterion = MaskedMSELoss()
    best_mcrmse = float("inf")

    # Ensure working directory exists for saving the model
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")
    print(f"Total Epochs: {epochs}")
    print("-" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Step Scheduler (Cosine Annealing updates per epoch)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), save_path)
            saved_msg = f"-> Model Saved to {save_path}"
        else:
            saved_msg = ""

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val MCRMSE: {val_mcrmse:.8f} {saved_msg}"
        )

    print("-" * 60)
    print(f"Training Complete. Best Validation MCRMSE: {best_mcrmse:.8f}")
    return best_mcrmse
