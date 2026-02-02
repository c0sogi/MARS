import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Seq_Len, 3)
        preds = model(sequence, loop_type, pair_dist)

        # Slicing: Only the first 68 positions are scored/trained
        # targets shape is (Batch, 107, 3), but valid data is only in first 68 indices
        # We strictly slice both preds and targets to [:, :68, :]
        preds_sliced = preds[:, : Config.PRED_LEN, :]
        targets_sliced = targets[:, : Config.PRED_LEN, :]

        # Compute Loss (MSE)
        loss = criterion(preds_sliced, targets_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer Step
        optimizer.step()

        running_loss += loss.item() * sequence.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(sequence, loop_type, pair_dist)

            # Slice to scored positions (first 68)
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced)
            all_targets.append(targets_sliced)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = MCRMSE(all_targets, all_preds)

    return score.item()


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Forward pass
            # Output shape: (Batch, 107, 3)
            # We need predictions for all 107 positions for the submission file structure,
            # even though only 68 are scored.
            preds = model(sequence, loop_type, pair_dist)

            preds_list.append(preds.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate predictions: (N_samples, 107, 3)
    preds_array = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Submission columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model outputs: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # Missing columns: deg_pH10, deg_50C (fill with 0)

    submission_rows = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_array[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Extract predicted values
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill zeros for ignored columns
            deg_pH10 = 0.0
            deg_50C = 0.0

            row = {
                "id_seqpos": f"{sample_id}_{seqpos}",
                "reactivity": reactivity,
                "deg_Mg_pH10": deg_Mg_pH10,
                "deg_pH10": deg_pH10,
                "deg_Mg_50C": deg_Mg_50C,
                "deg_50C": deg_50C,
            }
            submission_rows.append(row)

    df_sub = pd.DataFrame(submission_rows)

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")


def run_training():
    """
    Main controller for training, validation, and submission.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.initialize_workspace()
    device = torch.device(Config.DEVICE)

    print(f"Running Experiment: {Config.EXPERIMENT_ID}")
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = RNAModel().to(device)

    # 4. Optimizer & Scheduler
    # AdamW with low weight decay (1e-4) to preserve recurrent signal
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # MSE Loss (L2)
    criterion = nn.MSELoss()

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 5. Training Loop
    best_mcrmse = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with MCRMSE: {best_mcrmse:.10f}")

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")

    # 6. Inference
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    run_training()
