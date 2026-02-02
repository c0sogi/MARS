import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    WORKING_DIR,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
)
from library.utils import log_metric, set_seed
from library.data import get_data_loaders
from library.model import QCWBN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels)

        # Backward pass
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
    Returns average loss, true labels, and predicted probabilities.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images, angles)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return avg_loss, all_labels, all_preds


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def run_fold(fold_index, debug=False):
    """
    Runs the training and evaluation loop for a single fold.

    Args:
        fold_index (int): The fold index to run.
        debug (bool): If True, runs with a smaller dataset for debugging.

    Returns:
        dict: Contains validation loss, validation predictions, test predictions, and targets.
    """
    print(f"Starting Fold {fold_index}...")
    set_seed()  # Ensure reproducibility

    # Get DataLoaders
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        fold_index=fold_index, load_cached_data=True, debug=debug
    )

    # Initialize Model
    model = QCWBN().to(DEVICE)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: Reduce LR when validation loss stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early Stopping variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_counter = 0

    # Path to save the best model
    save_path = os.path.join(WORKING_DIR, f"model_fold_{fold_index}.pth")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step(val_loss)

        # Logging
        log_metric("Train", "Loss", train_loss, epoch)
        log_metric("Val", "Loss", val_loss, epoch)

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model to disk immediately
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best weights for final inference
    print(f"Loading best weights from {save_path}...")
    model.load_state_dict(torch.load(save_path))

    # Final Validation Predictions (for OOF)
    final_val_loss, final_val_labels, final_val_preds = validate(
        model, val_loader, criterion, DEVICE
    )
    print(f"Final Validation Loss (Fold {fold_index}): {final_val_loss}")

    # Test Predictions
    print("Generating test predictions...")
    test_preds = predict(model, test_loader, DEVICE)

    return {
        "fold": fold_index,
        "val_loss": final_val_loss,
        "val_labels": final_val_labels,
        "val_preds": final_val_preds,
        "test_preds": test_preds,
        "test_ids": test_ids,
    }
