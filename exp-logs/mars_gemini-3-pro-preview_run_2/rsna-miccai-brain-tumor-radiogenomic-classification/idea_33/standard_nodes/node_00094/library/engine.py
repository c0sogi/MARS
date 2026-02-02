import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Torch optimizer.
        criterion (Loss): Loss function (e.g., BCEWithLogitsLoss).
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary containing 'loss' and 'auc'.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate results
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Calculate AUC
        # Handle edge case where only one class is present in batch
        try:
            if len(np.unique(all_targets)) > 1:
                auc = roc_auc_score(all_targets, all_probs)
            else:
                auc = 0.5
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    return {"loss": avg_loss, "auc": auc}


def run_training(model, train_loader, val_loader, optimizer, epochs, device, save_path):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        optimizer (Optimizer): Optimizer.
        epochs (int): Number of epochs.
        device (str): Device.
        save_path (str): Path to save the best model.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.16f} - "
            f"Val Loss: {val_loss:.16f} - "
            f"Val AUC: {val_auc:.16f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path} (AUC: {best_auc:.16f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Training complete. Best Validation AUC: {best_auc:.16f}")


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test data loader.
        device (str): Device.

    Returns:
        pd.DataFrame: DataFrame with 'BraTS21ID' and 'MGMT_value'.
    """
    model.eval()
    results = []

    logger.info("Starting inference with TTA...")

    with torch.no_grad():
        for inputs, subject_ids in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            # 1. Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, [3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, [2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average Probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            avg_probs = avg_probs.cpu().numpy().flatten()

            # Store results
            # subject_ids is a tensor of IDs
            ids = subject_ids.numpy()

            for bid, prob in zip(ids, avg_probs):
                results.append({"BraTS21ID": bid, "MGMT_value": prob})

    return pd.DataFrame(results)


def generate_submission(model, test_loader, device, output_path):
    """
    Wrapper to generate submission file from the best model.
    """
    # Ensure model is in eval mode
    model.eval()

    # Generate predictions
    df_pred = predict_with_tta(model, test_loader, device)

    # Sort by ID just in case
    df_pred = df_pred.sort_values("BraTS21ID")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_pred.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
