import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.model import IcebergResNet18
from library.data_loader import get_dataloaders, load_data, get_test_loader


def predict_with_tta(model, images, angles, device):
    """
    Predicts with Test Time Augmentation: Original, H-Flip, V-Flip.
    Returns average probability.
    """
    model.eval()
    with torch.no_grad():
        # Move to device
        images = images.to(device)
        angles = angles.to(device)

        # 1. Original
        logits_orig = model(images, angles)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip (dim 3 is width)
        images_h = torch.flip(images, [3])
        logits_h = model(images_h, angles)
        probs_h = torch.sigmoid(logits_h)

        # 3. Vertical Flip (dim 2 is height)
        images_v = torch.flip(images, [2])
        logits_v = model(images_v, angles)
        probs_v = torch.sigmoid(logits_v)

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v) / 3.0

    return avg_probs.cpu().numpy()


def validate_tta(model, loader, device):
    """
    Evaluates the model on the validation set using TTA.
    Computes Log Loss on the averaged probabilities.
    Cite solution_lesson_node_00025: TTA-Validation Gap.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            # predict_with_tta handles device movement for inputs
            probs = predict_with_tta(model, images, angles, device)
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Clip predictions to avoid log(0)
    all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)

    return log_loss(all_targets, all_preds)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (Batch, 1)

        # Apply Label Smoothing (Cite solution_lesson_node_00005)
        if Config.LABEL_SMOOTHING > 0:
            with torch.no_grad():
                labels = (
                    labels * (1.0 - Config.LABEL_SMOOTHING)
                    + 0.5 * Config.LABEL_SMOOTHING
                )

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / dataset_size


def train_model(train_idx, val_idx, device):
    """
    Trains a single model on the provided split.
    """
    print(f"\n--- Starting Single Model Training ---")
    print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(train_idx, val_idx)

    # Initialize Model
    model = IcebergResNet18().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Use TTA for validation metric (Cite solution_lesson_node_00025)
        val_loss = validate_tta(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss (TTA): {val_loss:.10f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Finished Training. Best Val Loss (TTA): {best_val_loss:.10f}")
    return best_val_loss


# generate_submission removed in favor of library/inference.py
