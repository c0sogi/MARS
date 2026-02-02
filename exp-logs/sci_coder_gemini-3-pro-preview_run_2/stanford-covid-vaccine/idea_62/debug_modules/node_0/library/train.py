import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import (
    WORKING_DIR,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SCORED_LEN,
    SEQ_LEN,
    TARGET_COLS,
    SCORED_TARGETS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SEED,
)
from library.utils import MCRMSELoss, MCRMSE
from library.model import HS_GFN
from library.data import get_loaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_scored_indices():
    """Returns the indices of the target columns that are scored."""
    return [i for i, col in enumerate(TARGET_COLS) if col in SCORED_TARGETS]


def train_one_epoch(model, loader, criterion, optimizer, device, scored_indices):
    """
    Trains the model for one epoch using the 2-pass recycling strategy.
    """
    model.train()
    running_loss = 0.0

    for x, partner_indices, targets in loader:
        x = x.to(device)  # (B, 19, L)
        partner_indices = partner_indices.to(device)  # (B, L)
        targets = targets.to(device)  # (B, 5, L)

        B, _, L = targets.shape

        # Prepare targets for Loss: (B, L, 5)
        targets_perm = targets.permute(0, 2, 1)

        # Create sequence mask: 1 for pos < SCORED_LEN, 0 otherwise
        # Shape (B, L)
        seq_mask = torch.zeros((B, L), device=device)
        seq_mask[:, :SCORED_LEN] = 1.0

        optimizer.zero_grad()

        # --- Pass 1: Zero Feedback ---
        # y_prev is None, so feedback is zero
        preds_1 = model(x, partner_indices, y_prev=None)  # (B, L, 5)

        # --- Pass 2: Feedback from Pass 1 ---
        # Prepare feedback: (B, 5, L)
        # Detach to stop gradients flowing through the feedback generation itself in this step
        y_feedback = preds_1.detach().permute(0, 2, 1)
        preds_2 = model(x, partner_indices, y_prev=y_feedback)  # (B, L, 5)

        # --- Loss Calculation ---
        # Select scored columns for metric optimization (reactivity, deg_Mg_pH10, deg_Mg_50C)
        preds_1_scored = preds_1[:, :, scored_indices]
        preds_2_scored = preds_2[:, :, scored_indices]
        targets_scored = targets_perm[:, :, scored_indices]

        # Compute loss on valid positions only
        loss_2 = criterion(preds_2_scored, targets_scored, seq_mask)
        loss_1 = criterion(preds_1_scored, targets_scored, seq_mask)

        # Combined loss
        loss = loss_2 + 0.5 * loss_1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device, scored_indices):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_fn = MCRMSE()

    with torch.no_grad():
        for x, partner_indices, targets in loader:
            x = x.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)  # (B, 5, L)

            B, _, L = targets.shape

            # Pass 1
            preds_1 = model(x, partner_indices, y_prev=None)

            # Pass 2
            y_feedback = preds_1.permute(0, 2, 1)
            preds_2 = model(x, partner_indices, y_prev=y_feedback)  # (B, L, 5)

            # Prepare for metric
            targets_perm = targets.permute(0, 2, 1)

            # Slice scored columns
            preds_scored = preds_2[:, :, scored_indices]
            targets_scored = targets_perm[:, :, scored_indices]

            # Mask
            seq_mask = torch.zeros((B, L), device=device)
            seq_mask[:, :SCORED_LEN] = 1.0

            metric_fn.update(preds_scored, targets_scored, seq_mask)

    return metric_fn.compute()


def predict(model, loader, device):
    """
    Runs inference on the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # RNADataset returns x, p_idx, y. We ignore y for test.
            x, partner_indices, _ = batch
            x = x.to(device)
            partner_indices = partner_indices.to(device)

            # Pass 1
            preds_1 = model(x, partner_indices, y_prev=None)

            # Pass 2
            y_feedback = preds_1.permute(0, 2, 1)
            preds_2 = model(x, partner_indices, y_prev=y_feedback)  # (B, L, 5)

            all_preds.append(preds_2.cpu().numpy())

    return np.concatenate(all_preds, axis=0), loader.dataset.ids


def generate_submission(model, test_loader, device):
    """
    Generates the submission CSV file.
    """
    print("Generating submission...")
    preds, ids = predict(model, test_loader, device)
    # preds shape: (N_samples, 107, 5)

    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)
        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            # Create dict
            row = {"id_seqpos": row_id}
            for j, col_name in enumerate(TARGET_COLS):
                row[col_name] = row_values[j]
            submission_data.append(row)

    df_sub = pd.DataFrame(submission_data)

    # Ensure column order matches requirements
    cols = ["id_seqpos"] + TARGET_COLS
    df_sub = df_sub[cols]

    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run_training(debug=False):
    """
    Main training loop.
    """
    set_seed()

    # Loaders
    # Note: load_cached_data=True ensures we use the cache mechanism defined in library/data.py
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=True, debug=debug
    )

    # Model
    model = HS_GFN().to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Reduce LR when validation score plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = MCRMSELoss()

    scored_indices = get_scored_indices()

    best_score = float("inf")
    early_stop_count = 0
    patience = 10

    print(f"Starting training on device: {device}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scored_indices
        )
        val_score = validate(model, val_loader, device, scored_indices)

        # Print full precision
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.10f} - Val MCRMSE: {val_score:.10f}"
        )

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score:.10f}")
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

    # Load best model for submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device)
