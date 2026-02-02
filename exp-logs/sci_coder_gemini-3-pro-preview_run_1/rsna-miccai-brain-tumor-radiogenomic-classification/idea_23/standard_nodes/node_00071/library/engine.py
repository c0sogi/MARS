import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.utils import calculate_roc_auc, print_metric, save_checkpoint
from library.config import MODEL_SAVE_PATH


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the training set.
        optimizer: The optimizer.
        device: The device to run on (cpu or cuda).
        criterion: The loss function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        # Targets need to be (B, 1) for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the validation set.
        device: The device to run on.
        criterion: The loss function.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Flatten arrays for metric calculation
    all_targets = np.array(all_targets).flatten()
    all_preds = np.array(all_preds).flatten()

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc_score


def train_model(
    model, train_loader, val_loader, optimizer, device, num_epochs, patience
):
    """
    Manages the full training loop with Early Stopping.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: The optimizer.
        device: The device to run on.
        num_epochs: Maximum number of epochs.
        patience: Patience for early stopping.

    Returns:
        model: The trained model with best weights loaded.
    """
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        # Print metrics with full precision
        print_metric("Train Loss", train_loss)
        print_metric("Val Loss", val_loss)
        print_metric("Val AUC", val_auc)

        # Early Stopping Logic based on AUC maximization
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save the best model state
            save_checkpoint(model.state_dict(), MODEL_SAVE_PATH)
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc}")

    # Load best model weights if any training occurred and improved
    if best_val_auc > 0:
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            # Handle case where checkpoint might be full dict or just state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded best model weights.")
        except Exception as e:
            print(f"Warning: Could not load best model weights: {e}")

    return model


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.

    Args:
        model: The trained model.
        dataloader: DataLoader for the test set.
        device: The device to run on.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_preds)


def generate_submission_csv(model, test_loader, test_ids, device, submission_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        test_ids: Array of BraTS21IDs corresponding to the test set.
        device: The device to run on.
        submission_path: Path to save the submission CSV.
    """
    print("Generating predictions for submission...")
    preds = predict(model, test_loader, device)

    # Ensure lengths match
    if len(preds) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(preds)}) does not match number of IDs ({len(test_ids)})"
        )

    df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
