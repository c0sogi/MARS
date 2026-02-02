import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import get_rmse, unscale_target
from library.config import Config


def train_fn(model, data_loader, optimizer, device, criterion, scheduler=None):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        data_loader: Training DataLoader.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        criterion: Loss function.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, dense, targets in data_loader:
        images = images.to(device)
        dense = dense.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images, dense)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step the scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    return running_loss / dataset_size


def valid_fn(model, data_loader, device, criterion):
    """
    Executes validation for one epoch.

    Args:
        model: The PyTorch model.
        data_loader: Validation DataLoader.
        device: 'cuda' or 'cpu'.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, rmse_score, raw_predictions_scaled)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, dense, targets in data_loader:
            images = images.to(device)
            dense = dense.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images, dense)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate results
    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    # Unscale to calculate RMSE in original [1, 100] range
    unscaled_preds = unscale_target(all_preds)
    unscaled_targets = unscale_target(all_targets)

    rmse = get_rmse(unscaled_targets, unscaled_preds)

    return avg_loss, rmse, all_preds


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    patience,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        device: 'cuda' or 'cpu'.
        epochs: Maximum number of epochs.
        save_path: Path to save the best model state_dict.
        patience: Number of epochs to wait for improvement before stopping.

    Returns:
        np.ndarray: Best validation predictions (OOF) scaled [0, 1].
    """
    # Using BCELoss because the model output has a Sigmoid activation
    criterion = nn.BCELoss()

    best_loss = np.inf
    best_rmse = np.inf
    best_preds = None
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_fn(
            model, train_loader, optimizer, device, criterion, scheduler
        )
        val_loss, val_rmse, val_preds = valid_fn(model, val_loader, device, criterion)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val RMSE: {val_rmse}"
        )

        # Early Stopping logic based on Validation Loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_rmse = val_rmse
            best_preds = val_preds
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [Improved] Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  [No Improve] Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("  Early stopping triggered.")
            break

    print(f"Training Complete. Best Val Loss: {best_loss}, Best Val RMSE: {best_rmse}")
    return best_preds


def predict(model, data_loader, device):
    """
    Generates predictions for a dataset (e.g., test set).

    Args:
        model: The trained PyTorch model.
        data_loader: DataLoader for inference.
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray: Unscaled predictions [1, 100].
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, dense, _ in data_loader:
            images = images.to(device)
            dense = dense.to(device)
            outputs = model(images, dense)
            all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    # Return unscaled predictions for final submission
    return unscale_target(all_preds)


def save_submission(ids, predictions, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        ids (list or np.ndarray): List of ID strings.
        predictions (list or np.ndarray): List of predicted scores.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
