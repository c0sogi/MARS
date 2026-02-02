import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint, save_metrics, print_metrics


def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to use.
        criterion (nn.Module): The loss function.

    Returns:
        dict: Dictionary containing 'loss' and 'auc' for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for data, target in loader:
        data = data.to(device)
        target = target.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(output).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC safely (handle cases with only one class in batch/epoch)
    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {"loss": epoch_loss, "auc": auc}


def valid_one_epoch(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        device (torch.device): The device to use.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (metrics_dict, predictions_array)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)

            all_targets.append(target.cpu().numpy())
            all_preds.append(torch.sigmoid(output).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {"val_loss": epoch_loss, "val_auc": auc}, all_preds


def fit_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    log_path,
):
    """
    Runs the full training loop with Early Stopping and Checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The learning rate scheduler.
        device (torch.device): The device to use.
        num_epochs (int): Maximum number of epochs.
        save_path (str): Path to save the best model checkpoint.
        log_path (str): Path to save the training logs (CSV).

    Returns:
        float: The best validation loss achieved.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure output directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )

        # Validation Phase
        val_metrics, _ = valid_one_epoch(model, valid_loader, device, criterion)

        # Scheduler Step (Cosine Annealing typically steps per epoch)
        if scheduler is not None:
            scheduler.step()

        # Combine and Print Metrics
        metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        print_metrics(metrics)
        save_metrics(metrics, log_path)

        # Early Stopping & Checkpointing (Monitor Validation Loss)
        val_loss = val_metrics["val_loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_val_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader (no labels).
        device (torch.device): The device to use.

    Returns:
        np.ndarray: Array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in loader:
            # Test loader returns only data (based on library/data.py)
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def save_submission(predictions, clips, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Array of probabilities.
        clips (np.ndarray): Array of clip filenames.
        output_path (str): Path to save the submission CSV.
    """
    df = pd.DataFrame({"clip": clips, "probability": predictions.flatten()})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
