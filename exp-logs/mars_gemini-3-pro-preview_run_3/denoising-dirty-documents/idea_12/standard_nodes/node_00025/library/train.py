import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import (
    WORKING_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS_STAGE_1,
    NUM_EPOCHS_STAGE_2,
    PATIENCE,
    DEVICE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything, calculate_rmse
from library.model import CAResDnCNN
from library.dataset import DenoisingDataset, get_processed_data


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Calculate ground truth noise residual
        # Model predicts noise, so target for loss is (Input - Clean)
        noise_target = inputs - targets

        optimizer.zero_grad()

        # Forward pass: Predict noise
        noise_pred = model(inputs)

        # Loss calculation
        loss = criterion(noise_pred, noise_target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average MSE loss and average RMSE of reconstructed images.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            noise_target = inputs - targets

            noise_pred = model(inputs)
            loss = criterion(noise_pred, noise_target)

            running_loss += loss.item() * inputs.size(0)

            # Reconstruct clean image for RMSE calculation
            # Clean = Input - Predicted Noise
            clean_pred = inputs - noise_pred

            # Clip values to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Store for RMSE calculation (move to CPU to save GPU memory)
            all_preds.append(clean_pred.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate RMSE using the utility function
    val_rmse = calculate_rmse(all_targets, all_preds)

    return epoch_loss, val_rmse


def run_training_stage(stage_name, model, train_loader, val_loader, num_epochs, device):
    """
    Runs a specific training stage (e.g., Sparse or Dense).
    Handles Optimizer, Scheduler, and Early Stopping.
    """
    print(f"\n--- Starting {stage_name} ---")

    # Initialize Optimizer and Scheduler for this stage
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, f"best_model_{stage_name}.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{num_epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val RMSE: {val_rmse}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved for {stage_name}.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch} for {stage_name}.")
            break

    # Load best weights from this stage before returning
    if os.path.exists(best_model_path):
        print(f"Loading best weights from {stage_name} (Val Loss: {best_val_loss})")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def train_model(limit=None):
    """
    Main function to execute the curriculum training pipeline.

    Args:
        limit (int, optional): Limit the number of source images for debugging.
    """
    seed_everything(SEED)

    # Initialize Model
    model = CAResDnCNN().to(DEVICE)

    # --- Preparation: Validation Data ---
    # We use the same validation set for both stages to have comparable metrics
    print("Loading Validation Data...")
    val_patches, val_targets = get_processed_data(
        mode="val", stride_type="sparse", load_cached_data=True, limit=limit
    )
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # --- Stage 1: Sparse Data (Convergence) ---
    print("Loading Stage 1 Training Data (Sparse)...")
    train_patches_s1, train_targets_s1 = get_processed_data(
        mode="train", stride_type="sparse", load_cached_data=True, limit=limit
    )

    train_dataset_s1 = DenoisingDataset(
        train_patches_s1, train_targets_s1, augment=True
    )
    train_loader_s1 = DataLoader(
        train_dataset_s1,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    model = run_training_stage(
        stage_name="Stage_1",
        model=model,
        train_loader=train_loader_s1,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS_STAGE_1,
        device=DEVICE,
    )

    # Clean up Stage 1 memory
    del train_patches_s1, train_targets_s1, train_dataset_s1, train_loader_s1
    import gc

    gc.collect()

    # --- Stage 2: Dense Data (Refinement) ---
    print("Loading Stage 2 Training Data (Dense)...")
    train_patches_s2, train_targets_s2 = get_processed_data(
        mode="train", stride_type="dense", load_cached_data=True, limit=limit
    )

    train_dataset_s2 = DenoisingDataset(
        train_patches_s2, train_targets_s2, augment=True
    )
    train_loader_s2 = DataLoader(
        train_dataset_s2,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    model = run_training_stage(
        stage_name="Stage_2",
        model=model,
        train_loader=train_loader_s2,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS_STAGE_2,
        device=DEVICE,
    )

    # Save Final Model
    final_model_path = os.path.join(WORKING_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training Complete. Final model saved to {final_model_path}")

    return model
