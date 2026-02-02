import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    compute_column_wise_rmsle,
    EarlyStopping,
    save_submission,
)
from library.data import get_dataloaders
from library.model import PGWDS


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move data to device
        batch_data = {
            "atomic_features": batch["atomic_features"].to(device),
            "batch_indices": batch["batch_indices"].to(device),
            "global_features": batch["global_features"].to(device),
        }
        targets = batch["targets"].to(device)

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes MSE Loss and Column-wise RMSLE.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch_data = {
                "atomic_features": batch["atomic_features"].to(device),
                "batch_indices": batch["batch_indices"].to(device),
                "global_features": batch["global_features"].to(device),
            }
            targets = batch["targets"].to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = running_loss / count

    # Concatenate for metric calculation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate RMSLE (Note: inputs are already log-transformed, so this computes RMSE on logs)
    rmsle = compute_column_wise_rmsle(all_preds, all_targets)

    return avg_loss, rmsle


def run_training(
    batch_size=Config.BATCH_SIZE,
    epochs=Config.NUM_EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
):
    """
    Orchestrates the full training pipeline with Early Stopping and LR Scheduling.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Get Data Loaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = PGWDS().to(device)

    # 3. Setup Optimizer, Criterion, Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE, verbose=True, path="best_model.pt"
    )

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val RMSLE: {val_rmsle}"
        )

        # Check early stopping
        early_stopping(val_loss, model, optimizer, scheduler, epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Load best model for return
    best_epoch, best_val_loss = load_checkpoint(model, filename="best_model.pt")
    print(f"Loaded best model from epoch {best_epoch} with Val Loss: {best_val_loss}")

    return model


def generate_submission(model, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Generates predictions for the test set using the trained model.
    Applies inverse transformation to targets (exp(x) - 1).
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get test loader
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    model.eval()
    all_ids = []
    all_preds = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for batch in test_loader:
            batch_data = {
                "atomic_features": batch["atomic_features"].to(device),
                "batch_indices": batch["batch_indices"].to(device),
                "global_features": batch["global_features"].to(device),
            }
            ids = batch["id"]

            # Forward pass (log space)
            outputs = model(batch_data)

            # Inverse transform: exp(x) - 1
            # Note: Targets were log1p transformed.
            preds_original_scale = torch.expm1(outputs)

            all_ids.extend(ids)
            all_preds.append(preds_original_scale.cpu().numpy())

    # Concatenate predictions
    final_preds = np.concatenate(all_preds, axis=0)

    # Save submission
    save_submission(all_ids, final_preds, filename="submission.csv")
    print(
        f"Submission saved to {os.path.join(Config.SUBMISSION_DIR, 'submission.csv')}"
    )
