import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import DEVICE, MODEL_SAVE_PATH
from library.utils import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The MIL network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (BCELoss).
        optimizer (Optimizer): Optimizer (AdamW).
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        # Reshape targets to match model output (Batch, 1)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass: inputs are (B, N, C, H, W)
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)

        # Detach and move to CPU for metrics
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate AUC
    # Handle case where batch might contain only one class which raises ValueError
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The MIL network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    model, train_loader, val_loader, optimizer, num_epochs, patience, device
):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): The optimizer.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        device (str): Device to run on.

    Returns:
        nn.Module: The model loaded with the best weights.
    """
    # Since model output is Sigmoid, use BCELoss
    criterion = nn.BCELoss()

    model = model.to(device)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # Reload best model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    return model


def generate_submission(model, test_loader, output_path, device):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        output_path (str): Path to save the submission CSV.
        device (str): Device to run on.
    """
    model.eval()
    model = model.to(device)

    all_preds = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Flatten to 1D array
            all_preds.extend(outputs.cpu().numpy().flatten())

    # Retrieve BraTS21IDs from the dataset
    # Assumes the loader is not shuffled (standard for test)
    test_df = test_loader.dataset.df
    ids = test_df["BraTS21ID"].values

    # Create DataFrame
    submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": all_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
