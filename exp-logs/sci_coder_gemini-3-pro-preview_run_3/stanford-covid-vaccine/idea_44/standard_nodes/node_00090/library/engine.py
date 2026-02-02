import torch
import torch.nn as nn
import os
from library.config import Config
from library.utils import compute_mcrmse


def train_fn(model, data_loader, optimizer, device, criterion):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Device to run training on.
        criterion: Loss function (MSELoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(features, pair_indices, pair_masks)

        # Strategy: Slice to scored sequence length (first 68 positions)
        # Objective: Standard MCRMSE (via MSE surrogate) on all 5 targets
        outputs_sliced = outputs[:, : Config.SEQ_SCORED, :]
        targets_sliced = targets[:, : Config.SEQ_SCORED, :]

        loss = criterion(outputs_sliced, targets_sliced)
        loss.backward()

        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set using the official MCRMSE metric.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        float: MCRMSE score.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features, pair_indices, pair_masks)

            # Accumulate full outputs; slicing and filtering is handled by compute_mcrmse
            preds_list.append(outputs.cpu())
            targets_list.append(targets.cpu())

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    score = compute_mcrmse(preds, targets)
    return score


def run_training(
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
    Runs the full training loop with early stopping and model checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    criterion = nn.MSELoss()
    best_score = float("inf")
    patience_counter = 0

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, device, criterion)
        val_score = eval_fn(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Early Stopping Logic
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best Val MCRMSE: {best_score}")
