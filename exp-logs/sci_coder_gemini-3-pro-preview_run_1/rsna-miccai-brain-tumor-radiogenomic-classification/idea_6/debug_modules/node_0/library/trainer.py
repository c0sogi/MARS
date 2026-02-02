import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import compute_auc
from library.dataset import get_dataloader
from library.model import MultiPlanarSiameseNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        sagittal = batch["sagittal"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        batch_size = labels.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(axial, coronal, sagittal)
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            sagittal = batch["sagittal"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            batch_size = labels.size(0)

            logits = model(axial, coronal, sagittal)
            loss = criterion(logits, labels)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc_score = compute_auc(all_targets, all_preds)
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training(df_train, df_val, fold_idx):
    """
    Runs the training pipeline for a specific fold.

    Args:
        df_train (pd.DataFrame): Training metadata.
        df_val (pd.DataFrame): Validation metadata.
        fold_idx (int): Index of the current fold (for saving files).

    Returns:
        float: Best validation AUC score achieved.
    """
    print(f"\nStarting training for Fold {fold_idx}...")

    device = Config.DEVICE

    # Initialize DataLoaders
    train_loader = get_dataloader(
        df_train,
        split_name="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    val_loader = get_dataloader(
        df_val,
        split_name="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Initialize Model
    model = MultiPlanarSiameseNet(pretrained=True)
    model.to(device)

    # Optimizer and Loss
    # Using AdamW with settings from Config/Idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth"
    )

    print(f"Training on {len(df_train)} samples, Validating on {len(df_val)} samples.")

    for epoch in range(Config.MAX_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        # We maximize AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Fold {fold_idx} finished. Best Val AUC: {best_auc}")
    return best_auc
