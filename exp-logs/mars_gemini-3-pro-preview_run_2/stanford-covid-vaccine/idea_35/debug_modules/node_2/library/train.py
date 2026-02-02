import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import SF_DCN


def masked_mcrmse_loss(preds, targets, scored_indices, seq_scored):
    """
    Calculates MCRMSE loss using PyTorch tensors, restricted to scored positions and columns.

    Args:
        preds (torch.Tensor): Predictions (B, L, 5)
        targets (torch.Tensor): Ground truth (B, L, 5)
        scored_indices (list): Indices of columns to score (e.g., [0, 1, 3])
        seq_scored (int): Number of positions to score from the start (e.g., 68)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Slice to scored sequence length and scored columns
    # Shape: (B, seq_scored, n_scored_cols)
    preds_scored = preds[:, :seq_scored, scored_indices]
    targets_scored = targets[:, :seq_scored, scored_indices]

    # Calculate MSE per column (averaging over batch and sequence)
    mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Average RMSEs
    loss = torch.mean(rmse)

    return loss


def train_one_epoch(model, loader, optimizer, config, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, partner_indices, targets, _ in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass returns predictions from both refinement passes
        y1, y2 = model(inputs, partner_indices)

        # Calculate loss for both passes
        loss1 = masked_mcrmse_loss(
            y1, targets, config.scored_indices, config.seq_scored
        )
        loss2 = masked_mcrmse_loss(
            y2, targets, config.scored_indices, config.seq_scored
        )

        # Weighted sum
        loss = config.loss_weight_pass_2 * loss2 + config.loss_weight_pass_1 * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, config, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets, _ in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # For validation, we only care about the final refined prediction (y2)
            _, y2 = model(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Slice to scored positions for metric calculation
    # Shape: (N, seq_scored, 5)
    preds_scored = all_preds[:, : config.seq_scored, :]
    targets_scored = all_targets[:, : config.seq_scored, :]

    # Calculate MCRMSE using the utility function
    # Note: utils.MCRMSE takes scored_indices to select specific columns
    metric = MCRMSE(targets_scored, preds_scored, scored_indices=config.scored_indices)

    return metric


def generate_submission(model, loader, config, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs, partner_indices, _, ids in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            _, y2 = model(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, seq_len, 5)

    # Flatten predictions for submission format
    # Format requires one row per sequence position
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []
    target_cols = (
        config.target_cols
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (seq_len, 5)
        for seqpos in range(config.seq_length):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Ensure column order
    cols = ["id_seqpos"] + target_cols
    df_sub = df_sub[cols]

    # Save
    print(f"Saving submission to {config.submission_path}")
    df_sub.to_csv(config.submission_path, index=False)


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model
    model = SF_DCN(config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    early_stop_counter = 0
    patience = 10

    print("Starting training...")
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        val_mcrmse = validate(model, val_loader, config, device)

        scheduler.step(val_mcrmse)

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, config, device)
    print("Done.")


if __name__ == "__main__":
    run_training()
