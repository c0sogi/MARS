import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import GLiClassModel
from library.data import get_loaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training of a single epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape matches logits

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates slice-level predictions to subject-level predictions before computing AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, subject_ids in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.append(subject_ids.numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    # Create a DataFrame for aggregation
    df_val = pd.DataFrame(
        {"BraTS21ID": all_ids, "prob": all_probs, "target": all_targets}
    )

    # Aggregate by Subject ID (Mean Consensus)
    # We take the mean of probabilities and the mean of targets (targets are constant per subject)
    df_agg = df_val.groupby("BraTS21ID").mean()

    # Compute AUC
    try:
        val_auc = roc_auc_score(df_agg["target"], df_agg["prob"])
    except ValueError:
        # Handle edge case where only one class is present in batch/split
        val_auc = 0.5

    return avg_loss, val_auc


def run_training(fold=None, num_epochs=Config.NUM_EPOCHS):
    """
    Main function to run the training pipeline.

    Args:
        fold (int, optional): Fold index for cross-validation. If None, uses fixed split.
        num_epochs (int): Number of epochs to train.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # Load Data
    print(f"Initializing DataLoaders (Fold: {fold})...")
    train_loader, val_loader = get_loaders(fold=fold)

    # Initialize Model
    print("Initializing Model...")
    model = GLiClassModel(
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.IN_CHANNELS,
    )
    model = model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved at epoch {epoch} with AUC: {val_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best AUC: {best_auc} at epoch {best_epoch}.")
