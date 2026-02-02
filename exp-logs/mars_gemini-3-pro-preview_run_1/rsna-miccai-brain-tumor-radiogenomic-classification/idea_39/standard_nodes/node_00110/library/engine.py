import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import (
    DEVICE,
    MODEL_SAVE_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SEED,
)
from library.model import EfficientNetExpert
from library.utils import seed_everything


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Handle edge case where batch might contain only one class
    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    return avg_loss, auc, all_preds, all_targets


def train_expert(
    expert_name, fold_idx, train_loader, val_loader, epochs, patience=PATIENCE
):
    """
    Orchestrates the training process for a specific expert model.
    Implements Early Stopping and saves the best model.

    Args:
        expert_name (str): Name of the expert ('lower', 'center', 'upper').
        fold_idx (int): Current fold index.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        float: Best validation AUC achieved.
    """
    seed_everything(SEED)

    # Initialize Model
    model = EfficientNetExpert(num_classes=1).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_auc = 0.0
    best_loss = float("inf")
    epochs_no_improve = 0

    save_path = os.path.join(
        MODEL_SAVE_DIR, f"best_model_{expert_name}_fold{fold_idx}.pth"
    )

    print(f"Starting training for Expert: {expert_name}, Fold: {fold_idx}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Logic based on AUC (primary metric)
        # Note: Sometimes loss is used, but for this competition AUC is the metric.
        if val_auc > best_auc:
            best_auc = val_auc
            best_loss = val_loss  # Track loss at best AUC
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Finished training {expert_name} Fold {fold_idx}. Best AUC: {best_auc}")
    return best_auc


def predict_expert(model, dataloader, device):
    """
    Generates predictions using a trained model.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Inference dataloader.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle case where dataloader returns (images, labels) or just (images)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, dtype=torch.float32)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds).flatten()
