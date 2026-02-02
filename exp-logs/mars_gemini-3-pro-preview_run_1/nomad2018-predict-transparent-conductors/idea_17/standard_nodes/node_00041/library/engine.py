import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    NUM_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    TARGET_COLS,
    WORKING_DIR,
)
from library.dataset import get_dataloaders
from library.model import PCWDSModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for atom_x, batch_indices, global_x, targets, _ in loader:
        # Move data to device
        atom_x = atom_x.to(device)
        batch_indices = batch_indices.to(device)
        global_x = global_x.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atom_x, batch_indices, global_x)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    # For column-wise metrics
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for atom_x, batch_indices, global_x, targets, _ in loader:
            atom_x = atom_x.to(device)
            batch_indices = batch_indices.to(device)
            global_x = global_x.to(device)
            targets = targets.to(device)

            outputs = model(atom_x, batch_indices, global_x)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Calculate column-wise RMSLE (since targets are already log1p transformed, RMSE on them is RMSLE)
    if count > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=0)
        rmsle_per_col = np.sqrt(mse_per_col)
        mean_rmsle = np.mean(rmsle_per_col)
    else:
        mean_rmsle = 0.0

    return avg_loss, mean_rmsle


def run_training(load_cached_data=True):
    """
    Main training loop with Early Stopping and Learning Rate Scheduling.
    """
    print(f"Starting training on device: {DEVICE}")

    # 1. Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = PCWDSModel().to(DEVICE)

    # 3. Setup Optimizer, Loss, Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Reduce LR when validation loss plateaus
    # Cite debug_lesson_1: Remove Deprecated verbose Parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Mean Squared Error Loss (on log-transformed targets)
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_rmsle = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val RMSLE: {val_rmsle:.8f}"
        )

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved! New best Val Loss: {best_val_loss:.8f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    """
    print("Generating submission...")

    # 1. Get Test DataLoader
    # We don't need train/val loaders here, but the function returns them all
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load Best Model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_SAVE_PATH}. Run training first."
        )

    model = PCWDSModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_ids = []
    all_preds = []

    # 3. Inference
    with torch.no_grad():
        for atom_x, batch_indices, global_x, _, ids in test_loader:
            atom_x = atom_x.to(DEVICE)
            batch_indices = batch_indices.to(DEVICE)
            global_x = global_x.to(DEVICE)

            outputs = model(atom_x, batch_indices, global_x)

            # Inverse transform: exp(y) - 1 (since we trained on log1p)
            preds_original_scale = torch.expm1(outputs)

            all_ids.extend(ids.numpy())
            all_preds.append(preds_original_scale.cpu().numpy())

    # 4. Format Submission
    all_preds = np.concatenate(all_preds, axis=0)

    submission_df = pd.DataFrame(all_preds, columns=TARGET_COLS)
    submission_df.insert(0, "id", all_ids)

    # Sort by ID to ensure consistency
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
