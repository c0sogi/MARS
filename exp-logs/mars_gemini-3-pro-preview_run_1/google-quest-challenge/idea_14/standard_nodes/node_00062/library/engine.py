import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import compute_metric


def train_fn(dataloader, model, optimizer, device, scheduler):
    """
    Performs one epoch of training.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        device: The device to run on (CPU/GPU).
        scheduler: The learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    final_loss = 0
    count = 0
    criterion = nn.BCEWithLogitsLoss()

    for _, data in enumerate(dataloader):
        # Move inputs to device
        ids_q = data["input_ids_q"].to(device, dtype=torch.long)
        mask_q = data["attention_mask_q"].to(device, dtype=torch.long)
        ids_a = data["input_ids_a"].to(device, dtype=torch.long)
        mask_a = data["attention_mask_a"].to(device, dtype=torch.long)
        pool_mask_a = data["pooling_mask_a"].to(device, dtype=torch.float)
        targets = data["labels"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Update weights
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()
        count += 1

    return final_loss / count


def eval_fn(dataloader, model, device):
    """
    Performs evaluation on the validation set.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The neural network model.
        device: The device to run on (CPU/GPU).

    Returns:
        tuple: (Average validation loss, Spearman correlation score)
    """
    model.eval()
    final_loss = 0
    count = 0
    preds = []
    targets_list = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            ids_q = data["input_ids_q"].to(device, dtype=torch.long)
            mask_q = data["attention_mask_q"].to(device, dtype=torch.long)
            ids_a = data["input_ids_a"].to(device, dtype=torch.long)
            mask_a = data["attention_mask_a"].to(device, dtype=torch.long)
            pool_mask_a = data["pooling_mask_a"].to(device, dtype=torch.float)
            targets = data["labels"].to(device, dtype=torch.float)

            outputs = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)
            loss = criterion(outputs, targets)

            final_loss += loss.item()
            count += 1

            # Apply sigmoid to get probabilities in [0, 1]
            outputs = torch.sigmoid(outputs)

            preds.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    targets_list = np.concatenate(targets_list)

    score = compute_metric(targets_list, preds)

    return final_loss / count, score


def run_training(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience=3
):
    """
    Orchestrates the training process across multiple epochs with early stopping.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to run on.
        epochs: Total number of epochs.
        patience: Number of epochs to wait for improvement before early stopping.
    """
    best_score = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss}")

        val_loss, val_score = eval_fn(val_loader, model, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val Metric: {val_score}")

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Score Improved. Saved Best Model to {Config.MODEL_SAVE_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
