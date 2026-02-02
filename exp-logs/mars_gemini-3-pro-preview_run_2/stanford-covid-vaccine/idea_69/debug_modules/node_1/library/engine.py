import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, format_submission
from library.data import get_loaders
from library.model import RHS_GFN
from library.loss import MCRMSELoss


def train_fn(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass returns (y_2, y_1)
        y_2, y_1 = model(inputs, partner_indices)

        # Calculate composite loss
        # Loss is calculated on scored positions/columns only (handled by criterion)
        loss_2 = criterion(y_2, targets)
        loss_1 = criterion(y_1, targets)
        loss = loss_2 + 0.5 * loss_1

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, device):
    """
    Evaluates the model on the validation set using Global MCRMSE.
    Accumulates SSE globally before computing RMSE to avoid batch-size bias.
    """
    model.eval()

    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_cols_indices = Config.SCORED_INDICES
    num_scored_cols = len(scored_cols_indices)

    # Accumulators for SSE per column
    # Shape: (3,) corresponding to the 3 scored columns
    global_sse = np.zeros(num_scored_cols, dtype=np.float64)
    total_elements = 0

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward pass - we only care about the final refined prediction y_2
            y_2, _ = model(inputs, partner_indices)

            # Slice to scored sequence length (0-67)
            # Shape: (B, 68, 5)
            pred_scored = y_2[:, : Config.SCORED_LEN, :]
            true_scored = targets[:, : Config.SCORED_LEN, :]

            # Slice to scored columns
            # Shape: (B, 68, 3)
            pred_scored = pred_scored[:, :, scored_cols_indices]
            true_scored = true_scored[:, :, scored_cols_indices]

            # Convert to numpy
            pred_np = pred_scored.cpu().numpy()
            true_np = true_scored.cpu().numpy()

            # Calculate squared error
            squared_diff = (pred_np - true_np) ** 2

            # Sum errors over batch and sequence length for each column
            # Result shape: (3,)
            batch_sse = np.sum(squared_diff, axis=(0, 1))

            global_sse += batch_sse
            total_elements += inputs.size(0) * Config.SCORED_LEN

    # Calculate RMSE per column
    # total_elements is the count of rows * scored_len.
    # Since we sum over (0, 1), we divide by total_elements to get MSE per column.
    mse_per_col = global_sse / total_elements
    rmse_per_col = np.sqrt(mse_per_col)

    # MCRMSE is the mean of the column RMSEs
    mcrmse = np.mean(rmse_per_col)

    return mcrmse


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, partner_indices, _, ids in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            y_2, _ = model(inputs, partner_indices)

            # Move to CPU
            preds = y_2.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate all batches
    # Shape: (N, 107, 5)
    final_preds = np.concatenate(all_preds, axis=0)

    return all_ids, final_preds


def run(debug=False):
    """
    Main execution function.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_loaders(debug=debug)

    # 2. Initialize Model
    model = RHS_GFN().to(device)

    # 3. Setup Training Components
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = eval_fn(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_score})")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("\nStarting inference on test set...")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    test_ids, test_preds = predict_test(model, test_loader, device)

    # 6. Submission
    format_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    print("Process complete.")
