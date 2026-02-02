import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import hashlib
import sys

# Import from the provided library files
from library.config import Config
from library.data_processing import get_data_loaders
from library.model import VentilatorNet


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss (MAE) strictly on the inspiratory phase (u_out == 0).
    """

    def forward(self, pred, target, u_out):
        # Mask is 1 where u_out is 0 (inspiratory phase)
        mask = 1 - u_out

        # Calculate absolute error
        loss = torch.abs(pred - target) * mask

        # Return mean over valid elements (avoid div by zero)
        return loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Handles one epoch of training with auxiliary loss and gradient clipping.
    """
    model.train()
    running_loss = 0.0

    for x, u_out, y in loader:
        x = x.to(device)
        u_out = u_out.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass (returns final_pred and aux_pred)
        pred, aux_pred = model(x, u_out)

        # Calculate losses
        loss_main = criterion(pred, y, u_out)
        loss_aux = criterion(aux_pred, y, u_out)

        # Composite loss
        loss = loss_main + Config.AUX_WEIGHT * loss_aux

        # Backward pass
        loss.backward()

        # Strict gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimization steps
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    # Return average loss per batch for logging
    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    total_mae = 0.0
    total_count = 0.0

    with torch.no_grad():
        for x, u_out, y in loader:
            x = x.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            # Forward pass (eval mode returns only final_pred)
            pred = model(x, u_out)

            # Calculate metric components
            mask = 1 - u_out
            mae = torch.abs(pred - y) * mask

            total_mae += mae.sum().item()
            total_count += mask.sum().item()

    # Return global MAE
    return total_mae / (total_count + 1e-8)


def predict(model, loader, device):
    """
    Generates flat predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, u_out in loader:
            x = x.to(device)
            u_out = u_out.to(device)

            pred = model(x, u_out)

            # Flatten batch and sequence dimensions
            all_preds.append(pred.cpu().numpy().flatten())

    return np.concatenate(all_preds)


def run_training():
    """
    Main execution function:
    1. Loads data
    2. Initializes model, optimizer, scheduler
    3. Runs training loop with early stopping
    4. Generates submission
    """
    print(f"Initializing experiment: {Config.EXPERIMENT_NAME}")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)
    device = torch.device(Config.DEVICE)

    # 2. Model Setup
    model = VentilatorNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.SCHEDULER_PCT_START,
    )

    criterion = MaskedL1Loss()

    # 3. Training Loop
    best_mae = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_mae = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.9f}"
        )

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
            # print(f"  New best model saved! MAE: {best_mae:.9f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val MAE: {best_mae:.9f}")

    # 4. Inference and Submission
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate predictions
    predictions = predict(model, test_loader, device)

    # Retrieve Test IDs from cache
    # Replicate hash logic from library.data_processing
    feature_version = "v1_physics_robust_uniform"
    cache_hash = hashlib.md5(
        f"{feature_version}_{Config.DEBUG}_{Config.EXPERIMENT_NAME}".encode()
    ).hexdigest()

    test_ids_path = os.path.join(Config.CACHE_DIR, f"test_ids_{cache_hash}.npy")

    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"Cached test_ids not found at {test_ids_path}")

    test_ids = np.load(test_ids_path).flatten()

    # Ensure lengths match
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Shape mismatch: IDs {len(test_ids)} vs Preds {len(predictions)}"
        )

    # Create submission file
    submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
