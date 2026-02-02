import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNADilatedNet


def masked_mse_loss(preds, targets, mask):
    """
    Computes MSE loss only on scored positions defined by the mask.

    Args:
        preds: (Batch, SeqLen, Targets)
        targets: (Batch, SeqLen, Targets)
        mask: (Batch, SeqLen) - 1.0 for scored, 0.0 for unscored
    """
    # Expand mask to cover the target dimension: (Batch, SeqLen, 1)
    mask_expanded = mask.unsqueeze(-1)

    # Compute squared error
    squared_error = (preds - targets) ** 2

    # Apply mask
    masked_squared_error = squared_error * mask_expanded

    # Compute mean: Sum of errors / Number of valid elements
    # Number of valid elements = sum(mask) * num_targets
    # We add a small epsilon to denominator for safety, though mask should not be all zero
    loss = masked_squared_error.sum() / (mask_expanded.sum() * preds.shape[-1] + 1e-8)

    return loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        seq = batch["seq"].to(device)
        struct = batch["struct"].to(device)
        loop = batch["loop"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, struct, loop)

        # Compute Loss
        loss = masked_mse_loss(preds, targets, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seq.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    # We only care about scored positions for the metric,
    # but the metric function handles shape alignment.
    # We will extract the first 68 positions (SCORED_LEN) for metric calculation
    # to match the ground truth provided in validation set.

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            targets = batch["targets"].to(device)  # (B, 107, 5)

            preds = model(seq, struct, loop)  # (B, 107, 5)

            # Extract scored positions for metric calculation
            # targets in val set are padded with zeros, but we only score the first 68
            preds_scored = preds[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(preds_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = mcrmse_metric(all_targets, all_preds)
    return score


def generate_submission(model, loader, device):
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            ids = batch["id"]

            preds = model(seq, struct, loop)  # (B, 107, 5)

            ids_list.extend(ids)
            preds_list.append(preds.cpu().numpy())

    # Concatenate all predictions: (N_samples, 107, 5)
    full_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for CSV
    # We need to flatten: id_seqpos, val1, val2, val3, val4, val5
    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = full_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Save
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = RNADilatedNet(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val MCRMSE: {best_score:.6f}")

    # 5. Inference
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    run_training()
