import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import HCTDBiGRU


def train_epoch(model, loader, optimizer, device, criterion_fn):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices, pair_mask)

        # Slicing: Slice predictions/targets to the first 68 positions (seq_scored)
        # before metric calculation as per strategy.
        # Targets shape: (Batch, 107, 5) -> (Batch, 68, 5)
        outputs_sliced = outputs[:, : Config.PRED_LEN, :]
        targets_sliced = targets[:, : Config.PRED_LEN, :]

        # Calculate Loss (MCRMSE on all 5 targets)
        loss = criterion_fn(targets_sliced, outputs_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Validates the model.
    Calculates MCRMSE on the 3 scored columns for the first 68 positions.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # 1. Slice to sequence length 68
    y_pred = y_pred[:, : Config.PRED_LEN, :]
    y_true = y_true[:, : Config.PRED_LEN, :]

    # 2. Select scored columns indices
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Calculate Validation Metric
    val_score = mcrmse(y_true, y_pred, scored_indices=scored_indices)

    return val_score.item()


def run_training(debug=False):
    """
    Main execution function for training the model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=debug,
    )

    # 3. Model
    print("Initializing Model (HC-TD-BiGRU)...")
    model = HCTDBiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss function wrapper using mcrmse
    def criterion(y_true, y_pred):
        return mcrmse(
            y_true, y_pred, scored_indices=None
        )  # Use all columns for training loss

    # 5. Training Loop
    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Printing full precision
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_val_score})")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_val_score}")

    # 6. Inference & Submission
    generate_submission(test_loader, device)


def generate_submission(test_loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Generating submission...")

    # Load best model
    model = HCTDBiGRU().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["ids"]  # List of strings

            # Forward pass
            # Output shape: (Batch, 107, 5)
            outputs = model(inputs, pair_indices, pair_mask)
            outputs = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(outputs)

    # Concatenate predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(target_cols):
                row_dict[col] = float(row_values[j])

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
