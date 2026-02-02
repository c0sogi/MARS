import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.data import get_dataloaders
from library.model import StackedInteractionDenseNet


def masked_mcrmse_loss(preds, targets, mask):
    """
    Calculates the MCRMSE loss only on the scored columns.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
        targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
        mask (torch.Tensor): Mask of shape (Batch, Seq_Len) indicating valid positions.

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Indices corresponding to: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = Config.SCORED_TARGET_INDICES

    # Filter predictions and targets
    preds_filtered = preds[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # Ensure mask is broadcastable: (B, L) -> (B, L, 1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    # Calculate squared errors
    sq_diff = (preds_filtered - targets_filtered) ** 2

    # Apply mask (zero out invalid positions)
    masked_sq_diff = sq_diff * mask

    # Sum squared errors over batch and sequence dimensions
    # Shape becomes (Num_Scored_Cols,)
    sum_sq_errors = masked_sq_diff.sum(dim=(0, 1))

    # Count valid positions per column
    # Expand mask to match channel dimension: (B, L, 1) -> (B, L, Num_Scored_Cols)
    # Shape becomes (Num_Scored_Cols,)
    counts = mask.expand_as(preds_filtered).sum(dim=(0, 1))

    # Calculate MSE per column (add epsilon to avoid division by zero)
    mse = sum_sq_errors / (counts + 1e-12)

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # MCRMSE is the mean of the column-wise RMSEs
    loss = torch.mean(rmse)

    return loss


def validate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: The computed MCRMSE score.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(inputs, partner_indices)

            tracker.update(outputs, targets, mask)

    score = tracker.compute()
    return score


def train_model():
    """
    Executes the training pipeline:
    1. Initializes data, model, optimizer, scheduler.
    2. Runs the training loop with validation.
    3. Saves the best model based on validation MCRMSE.
    4. Implements early stopping.
    """
    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Initialize Model
    print("Initializing model...")
    model = StackedInteractionDenseNet().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduce LR if validation score plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE
    )

    # Training Loop Variables
    best_score = float("inf")
    early_stop_counter = 0
    early_stop_patience = Config.PATIENCE * 2  # Slightly more tolerant than scheduler

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, partner_indices)

            # Compute loss
            loss = masked_mcrmse_loss(outputs, targets, mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        val_score = validate_model(model, val_loader, device)

        # Scheduler update
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  > New best model saved! Score: {best_score}")
        else:
            early_stop_counter += 1
            print(
                f"  > No improvement. Early stopping counter: {early_stop_counter}/{early_stop_patience}"
            )

        # Early Stopping
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Score: {best_score}")
