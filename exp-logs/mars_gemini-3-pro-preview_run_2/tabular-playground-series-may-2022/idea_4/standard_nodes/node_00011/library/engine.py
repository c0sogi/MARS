import torch
import torch.nn as nn
import numpy as np
import os
from library.utils import compute_auc
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run on ('cpu' or 'cuda').

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        tokens = batch["tokens"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        outputs = model(continuous, tokens)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = continuous.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (str): Device to run on.

    Returns:
        tuple: (Average Validation Loss, Validation AUC)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            tokens = batch["tokens"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(continuous, tokens)
            loss = criterion(outputs, targets)

            batch_size = continuous.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        auc_score = compute_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return epoch_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.
        device (str): Device to run on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            tokens = batch["tokens"].to(device)

            outputs = model(continuous, tokens)
            all_preds.append(outputs.cpu().numpy())

    if len(all_preds) > 0:
        return np.vstack(all_preds).flatten()
    else:
        return np.array([])


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Orchestrates the training loop with early stopping based on AUC.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        optimizer (Optimizer): Optimizer.
        device (str): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
        scheduler (lr_scheduler): Learning rate scheduler (optional).
    """
    criterion = nn.BCELoss()
    best_auc = -1.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        # Print metrics with full precision
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc} | LR: {current_lr}"
        )

        # Early Stopping Logic based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
