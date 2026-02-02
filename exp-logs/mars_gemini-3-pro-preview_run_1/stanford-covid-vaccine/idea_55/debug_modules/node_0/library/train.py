import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import ScaledResidualWideStreamBiGRU


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Enable gradient clipping
    clip_value = Config.GRAD_CLIP

    for batch in loader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Seq_Len, 3)
        preds = model(sequence, loop_type, pair_dist)

        # Masking: Calculate loss only on first 68 positions (Config.PRED_LEN)
        # Targets are already shape (Batch, 107, 3), but we only care about first 68
        if Config.MASK_LOSS:
            preds_masked = preds[:, : Config.PRED_LEN, :]
            targets_masked = targets[:, : Config.PRED_LEN, :]
        else:
            preds_masked = preds
            targets_masked = targets

        loss = criterion(preds_masked, targets_masked)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

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
            targets = batch["targets"].to(device)

            preds = model(sequence, loop_type, pair_dist)

            # For validation metric, we strictly look at the scored positions (first 68)
            preds_masked = preds[:, : Config.PRED_LEN, :]
            targets_masked = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_masked.cpu().numpy())
            all_targets.append(targets_masked.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse_metric(all_targets, all_preds)
    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            batch_ids = batch["id"]

            # Forward pass (Batch, 107, 3)
            preds = model(sequence, loop_type, pair_dist)

            preds_list.append(preds.cpu().numpy())
            ids.extend(batch_ids)

    # Concatenate all predictions: (N_samples, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model outputs: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # Missing: deg_pH10, deg_50C -> Fill with 0.0

    submission_rows = []

    # Mapping model output indices to column names
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Construct row ID
            row_id = f"{sample_id}_{seqpos}"

            # Get predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    df_submission = pd.DataFrame(submission_rows)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_submission = df_submission[cols]

    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)
    print("Submission saved.")


def train_model(debug_subset=None, epochs=None):
    """
    Main training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    if epochs is None:
        epochs = Config.EPOCHS

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug_subset=debug_subset
    )

    # 3. Model
    print("Initializing model...")
    model = ScaledResidualWideStreamBiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model! Saved to {best_model_path}")

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
