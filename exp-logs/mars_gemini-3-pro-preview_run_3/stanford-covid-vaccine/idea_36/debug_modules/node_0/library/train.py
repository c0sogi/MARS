import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_loaders
from library.model import DDPNBiGRU


def loss_fn_5_targets(preds, targets, mask):
    """
    Calculates MCRMSE loss on all 5 target columns.
    """
    # Indices for all 5 columns: 0, 1, 2, 3, 4
    target_indices = list(range(Config.NUM_TARGETS))

    loss = 0.0
    eps = 1e-6

    for idx in target_indices:
        p = preds[:, :, idx]
        t = targets[:, :, idx]

        # Squared Error
        se = (p - t) ** 2

        # Apply mask (1 for valid positions, 0 otherwise)
        se = se * mask

        # Mean Squared Error over valid positions
        total_valid = mask.sum()
        if total_valid < 1:
            total_valid = 1.0

        mse = se.sum() / total_valid

        # RMSE
        rmse = torch.sqrt(mse + eps)
        loss += rmse

    # Average over all 5 columns
    return loss / len(target_indices)


def train_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        X = batch["X"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        y = batch["y"].to(device)
        target_masks = batch["target_masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X, pair_indices, pair_masks)

        # Compute loss on all 5 targets
        loss = loss_fn_5_targets(preds, y, target_masks)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Validates the model and returns the official MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            y = batch["y"]  # Keep on CPU

            preds = model(X, pair_indices, pair_masks)
            all_preds.append(preds.cpu())
            all_targets.append(y)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metric using the official utility (scores only specific columns)
        mcrmse, col_scores = calculate_mcrmse(all_preds, all_targets)
        return mcrmse, col_scores
    else:
        return float("inf"), {}


def generate_submission_file(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    ids = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            batch_ids = batch["id"]

            preds = model(X, pair_indices, pair_masks)

            ids.extend(batch_ids)
            preds_list.append(preds.cpu().numpy())

    if not preds_list:
        print("No predictions generated.")
        return

    preds_array = np.concatenate(preds_list, axis=0)  # (N, 107, 5)

    submission_rows = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds_array[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_data = {
                "id_seqpos": row_id,
                "reactivity": sample_preds[seqpos, 0],
                "deg_Mg_pH10": sample_preds[seqpos, 1],
                "deg_pH10": sample_preds[seqpos, 2],
                "deg_Mg_50C": sample_preds[seqpos, 3],
                "deg_50C": sample_preds[seqpos, 4],
            }
            submission_rows.append(row_data)

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main orchestration function.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_loaders()

    # Initialize Model
    model = DDPNBiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_mcrmse, val_scores = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss (5-target): {train_loss:.6f} | Val MCRMSE (3-target): {val_mcrmse:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")

    # Load best model for submission
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Loading best model for submission generation...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
        generate_submission_file(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print("Error: Best model file not found.")
