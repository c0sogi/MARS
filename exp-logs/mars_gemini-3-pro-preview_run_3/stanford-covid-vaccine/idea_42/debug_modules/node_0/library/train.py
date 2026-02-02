import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel


def criterion_mcrmse(y_pred, y_true):
    """
    Calculates MCRMSE on all 5 target columns for the training objective.
    Slices data to the scored sequence length (68) before calculation.

    Args:
        y_pred (torch.Tensor): (Batch, Seq, 5)
        y_true (torch.Tensor): (Batch, Seq, 5)

    Returns:
        torch.Tensor: Scalar loss
    """
    # Slice to scored length
    seq_scored = Config.SEQ_SCORED
    pred_sliced = y_pred[:, :seq_scored, :]
    true_sliced = y_true[:, :seq_scored, :]

    # Compute MSE per column (Batch, Seq, Cols) -> (Cols,)
    mse_per_col = torch.mean((pred_sliced - true_sliced) ** 2, dim=(0, 1))

    # Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Average RMSE across all 5 columns
    loss = torch.mean(rmse_per_col)

    return loss


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        adj = batch["adj_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, adj)

        # Calculate loss (MCRMSE on all 5 columns)
        loss = criterion_mcrmse(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the scored MCRMSE metric.
    Aggregates predictions globally before calculation.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adj_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, adj)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    y_pred_global = torch.cat(all_preds, dim=0)
    y_true_global = torch.cat(all_targets, dim=0)

    # Calculate metric using the utility function (handles slicing and column selection)
    score = scored_mcrmse(y_pred_global, y_true_global)

    return score.item()


def run_training():
    """
    Main training routine.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model
    print("Initializing Model...")
    model = RNAModel().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_score < (best_score - Config.MIN_DELTA):
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")


def generate_submission():
    """
    Generates predictions for the test set and creates the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    print(f"Loading best model from {Config.MODEL_PATH}...")
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adj_indices"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(inputs, adj)  # (Batch, 107, 5)

            # Store results
            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # We need to flatten: id_seqpos for every position
    submission_data = []

    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            # Create dictionary for this row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    run_training()
    generate_submission()
