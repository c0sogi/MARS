import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import DeepBiGRUNet


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # Loss function: MSE. We calculate MCRMSE manually for logging if needed,
    # but standard MSE is sufficient for optimization gradients.
    criterion = nn.MSELoss()

    for batch in loader:
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(features, pair_indices, pair_masks)

        # Slice to scored sequence length for loss calculation
        # We only have ground truth for the first 68 bases.
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]
        targets_sliced = targets[:, : Config.SEQ_SCORED, :]

        loss = criterion(preds_sliced, targets_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU for accumulation

            preds = model(features, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE (Handles slicing internally)
    score = calculate_mcrmse(all_preds, all_targets)

    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids = []
    preds_list = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            # Output shape: (B, 107, 5)
            preds = model(features, pair_indices, pair_masks)
            preds = preds.cpu().numpy()

            ids.extend(batch_ids)
            preds_list.append(preds)

    # Concatenate predictions: (N_test, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    # We need to flatten the predictions to match the format:
    # id_seqpos, reactivity, deg_Mg_pH10, ...

    submission_data = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = DeepBiGRUNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  New best model saved! Score: {best_mcrmse}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    # 6. Submission
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run_training()
