import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to train on ('cuda' or 'cpu').
        criterion (Loss): The loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        # Apply Label Smoothing
        # smooth_targets = targets * (1 - epsilon) + 0.5 * epsilon
        smooth_targets = (
            targets * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, smooth_targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The validation data loader.
        device (str): Device to evaluate on.
        criterion (Loss): The loss function.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Handle edge case where only one class is present in the batch
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            val_auc = 0.5
    else:
        val_auc = 0.5

    return val_loss, val_auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=5,
):
    """
    Orchestrates the training process with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to use.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        nn.Module: The model with the best weights loaded.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on {device} for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model weights
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Aggregates instance-level predictions to subject-level predictions via mean.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (str): Device to use.

    Returns:
        pd.DataFrame: DataFrame with columns ['BraTS21ID', 'MGMT_value']
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, subject_ids in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Collect results
            # subject_ids is a tuple/list from the dataloader
            for sid, prob in zip(subject_ids, probs):
                results.append({"BraTS21ID": int(sid), "prob": prob})

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("Warning: No predictions generated.")
        return pd.DataFrame(columns=["BraTS21ID", "MGMT_value"])

    # Consensus Aggregation: Mean probability per subject
    # Since we have 3 instances per subject, we group by ID and take the mean
    df_agg = df_results.groupby("BraTS21ID")["prob"].mean().reset_index()
    df_agg.rename(columns={"prob": "MGMT_value"}, inplace=True)

    return df_agg
