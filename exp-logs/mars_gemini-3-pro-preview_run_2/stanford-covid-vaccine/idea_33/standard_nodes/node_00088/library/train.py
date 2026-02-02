import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, get_device, mcrmse_numpy
from library.data import get_dataloaders
from library.model import PFR_DN
from library.loss import MCRMSELoss


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training using the Two-Stage Forward Pass strategy.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, partner_indices, targets) in enumerate(loader):
        features = features.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # --- Two-Stage Forward Pass ---

        # Pass 1: Cold Start (Recycling initialized to zeros internally)
        y_hat_1 = model(features, partner_indices, recycling=None)

        # Detach Pass 1 output to prevent gradient explosion/instability in the feedback loop
        recycling_input = y_hat_1.detach()

        # Pass 2: Refinement (Use Pass 1 output as recycling input)
        y_hat_2 = model(features, partner_indices, recycling=recycling_input)

        # --- Loss Calculation ---

        # Primary loss on refined predictions
        loss_2 = criterion(y_hat_2, targets)

        # Auxiliary loss on initial predictions (weighted 0.5)
        loss_1 = criterion(y_hat_1, targets)

        total_loss = loss_2 + 0.5 * loss_1

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Validates the model using the MCRMSE metric on the scored columns.
    Uses the Two-Stage inference strategy.
    """
    model.eval()
    running_loss = 0.0

    # Lists to store flattened predictions and targets for metric calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, partner_indices, targets in loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # --- Two-Stage Inference ---

            # Pass 1: Cold Start
            y_hat_1 = model(features, partner_indices, recycling=None)

            # Pass 2: Refinement
            # In inference, we strictly use the output of Pass 1 as input to Pass 2
            y_hat_2 = model(features, partner_indices, recycling=y_hat_1)

            # Compute Loss (for monitoring purposes)
            loss = criterion(y_hat_2, targets)
            running_loss += loss.item()

            # Prepare data for MCRMSE calculation
            # 1. Slice to scored length (68)
            # 2. Move to CPU numpy
            preds_scored = y_hat_2[:, : Config.SCORED_SEQ_LENGTH, :].cpu().numpy()
            targets_scored = targets[:, : Config.SCORED_SEQ_LENGTH, :].cpu().numpy()

            # 3. Flatten Batch and Sequence dimensions: (Batch * Scored_Len, 5)
            # This aligns with the expectation of mcrmse_numpy which takes (N, 5)
            preds_flat = preds_scored.reshape(-1, Config.OUTPUT_DIM)
            targets_flat = targets_scored.reshape(-1, Config.OUTPUT_DIM)

            all_preds.append(preds_flat)
            all_targets.append(targets_flat)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    metric_score = mcrmse_numpy(all_targets, all_preds)
    avg_loss = running_loss / len(loader)

    return avg_loss, metric_score


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, partner_indices, _ in test_loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # --- Two-Stage Inference ---
            y_hat_1 = model(features, partner_indices, recycling=None)
            y_hat_2 = model(features, partner_indices, recycling=y_hat_1)

            # Store predictions: (Batch, Seq_Len, 5)
            all_preds.append(y_hat_2.cpu().numpy())

    # Concatenate all batches: (Total_Test_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Get Test IDs
    test_ids = test_loader.dataset.ids

    # Prepare submission data
    submission_rows = []

    # Iterate through each sample
    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # Shape (107, 5)

        # Iterate through each sequence position
        for seqpos in range(Config.SEQ_LENGTH):
            # Row ID: id_sequence_pos
            row_id = f"{sample_id}_{seqpos}"

            # Get values for this position
            vals = sample_preds[seqpos]

            row_data = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_rows.append(row_data)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = PFR_DN().to(device)
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 4. Training Loop
    best_metric = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_metric)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_metric:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! MCRMSE: {best_metric:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 5. Load Best Model & Generate Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    generate_submission(model, test_loader, device)
