import torch
import numpy as np
import os
import random
from library.config import Config
from library.utils import MCRMSELoss, calculate_metric


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: The optimizer.
        device: Device to train on.
        loss_fn: The loss function (MCRMSELoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        bpp_masks = batch["bpp_masks"].to(device)
        targets = batch["targets"].to(device)

        batch_size = inputs.size(0)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, bpp_indices, bpp_masks)

        # Loss calculation (MCRMSELoss handles slicing internally)
        loss = loss_fn(outputs, targets)

        # Backpropagation
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Device to evaluate on.

    Returns:
        float: The MCRMSE score calculated on scored columns and positions.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric using utils function (handles slicing and column filtering)
    score = calculate_metric(all_preds, all_targets)
    return score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Main training loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    set_seed(Config.SEED)
    loss_fn = MCRMSELoss()

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score} | "
            f"LR: {current_lr}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
