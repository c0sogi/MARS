import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from library.config import Config
from library.utils import seed_everything, calculate_metrics, EarlyStopping
from library.data_loader import get_fold_loaders
from library.model import PPCWBN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_logits = []

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (images, angles)
        outputs = model(images, angles)

        # Squeeze outputs to match labels shape (B,)
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        all_targets.append(labels.detach().cpu().numpy())
        all_logits.append(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)

    metrics = calculate_metrics(all_targets, all_logits)
    metrics["loss"] = epoch_loss

    return metrics


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            all_targets.append(labels.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)

    metrics = calculate_metrics(all_targets, all_logits)
    metrics["loss"] = epoch_loss

    return metrics


def run_fold(fold_idx):
    """
    Runs the training and evaluation pipeline for a single fold.

    Args:
        fold_idx (int): The index of the fold to process.

    Returns:
        dict: Best validation metrics and path to saved model.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting Fold {fold_idx} on {device}...")

    # 1. Get Data Loaders
    # This handles strict leakage prevention and scaling
    train_loader, val_loader, scaling_stats, angle_mean = get_fold_loaders(fold_idx)

    # Save scaling stats for this fold to use during inference
    stats_path = os.path.join(Config.WORKING_DIR, f"stats_fold_{fold_idx}.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "scaling_stats": [
                    list(s) for s in scaling_stats
                ],  # Convert tuples to lists
                "angle_mean": float(angle_mean),
            },
            f,
        )

    # 2. Initialize Model
    model = PPCWBN()
    model.to(device)

    # 3. Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Reduce LR when validation loss stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # Early Stopping: Save best model
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=checkpoint_path
    )

    # 4. Training Loop
    best_val_metrics = None

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Print metrics (Full precision)
        print(f"Fold {fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(
            f"  Train Loss: {train_metrics['loss']} | Acc: {train_metrics['accuracy']}"
        )
        print(f"  Val Loss:   {val_metrics['loss']} | Acc: {val_metrics['accuracy']}")

        # Step Scheduler
        scheduler.step(val_metrics["loss"])

        # Check Early Stopping
        early_stopping(val_metrics["loss"], model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Load Best Weights
    print(f"Loading best weights for Fold {fold_idx}...")
    early_stopping.load_best_weights(model)

    # Final Validation on Best Weights to confirm metrics
    final_val_metrics = validate(model, val_loader, criterion, device)
    print(f"Fold {fold_idx} Final Best Val Loss: {final_val_metrics['loss']}")
    print(f"Fold {fold_idx} Final Best Val Acc:  {final_val_metrics['accuracy']}")

    return {
        "fold": fold_idx,
        "val_loss": final_val_metrics["loss"],
        "val_accuracy": final_val_metrics["accuracy"],
        "model_path": checkpoint_path,
        "stats_path": stats_path,
    }
