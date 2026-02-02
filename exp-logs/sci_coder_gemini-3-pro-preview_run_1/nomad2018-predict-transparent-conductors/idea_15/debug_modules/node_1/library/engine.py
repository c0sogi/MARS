import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function (MSE).
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for atomic_x, global_x, mask, targets, _ in loader:
        atomic_x = atomic_x.to(device)
        global_x = global_x.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_x, global_x, mask)

        # Compute loss (MSE on log-transformed targets)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (Average Loss, Column-wise RMSLE array, Mean RMSLE)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for atomic_x, global_x, mask, targets, _ in loader:
            atomic_x = atomic_x.to(device)
            global_x = global_x.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            outputs = model(atomic_x, global_x, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate Column-wise RMSLE
    # Since model outputs and targets are log1p transformed,
    # RMSE in this space is equivalent to RMSLE in original space.
    mse_col = np.mean((all_preds - all_targets) ** 2, axis=0)
    rmsle_col = np.sqrt(mse_col)

    mean_rmsle = np.mean(rmsle_col)

    return epoch_loss, rmsle_col, mean_rmsle


def train_model(model, train_loader, val_loader):
    """
    Full training loop with Early Stopping and Learning Rate Scheduler.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
    """
    device = Config.DEVICE
    model.to(device)

    # Loss function: MSE Loss on log-transformed targets
    criterion = nn.MSELoss()

    # Optimizer: AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation metric stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    best_metric = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(
        f"Architecture: Atomic {Config.ATOMIC_HIDDEN_DIM} | Global {Config.GLOBAL_HIDDEN_DIM} | Fusion {Config.FUSION_HIDDEN_DIM}"
    )

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle_col, val_mean_rmsle = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step based on Mean RMSLE
        scheduler.step(val_mean_rmsle)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE (Mean): {val_mean_rmsle:.6f} | "
            f"Form E RMSLE: {val_rmsle_col[0]:.6f} | "
            f"Bandgap RMSLE: {val_rmsle_col[1]:.6f}"
        )

        # Early Stopping and Model Checkpointing
        if val_mean_rmsle < best_metric:
            best_metric = val_mean_rmsle
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! (Metric: {best_metric:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Mean RMSLE: {best_metric:.6f}")


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set using the best saved model.
    Applies inverse transformation (expm1) to revert log1p scaling.

    Args:
        model: The PyTorch model structure.
        test_loader: Test DataLoader.
    """
    device = Config.DEVICE
    model.to(device)

    # Load best weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model for submission.")
    else:
        print("Warning: No saved model found. Using current weights.")

    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for atomic_x, global_x, mask, _, ids in test_loader:
            atomic_x = atomic_x.to(device)
            global_x = global_x.to(device)
            mask = mask.to(device)

            outputs = model(atomic_x, global_x, mask)

            # Inverse transform: exp(x) - 1
            # Since we trained on log1p(y), output is log(1+y)
            preds_original_scale = torch.expm1(outputs)

            # Ensure non-negative energies (physics constraint)
            preds_original_scale = torch.clamp(preds_original_scale, min=0.0)

            preds_list.append(preds_original_scale.cpu().numpy())
            ids_list.extend(ids.numpy())

    all_preds = np.vstack(preds_list)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to match sample submission order
    submission_df.sort_values("id", inplace=True)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
