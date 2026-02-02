import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE,
    TARGET_COLS,
    SEQ_LEN,
    PRED_LEN,
    SEED,
    BATCH_SIZE,
)
from library.utils import seed_everything, calculate_mcrmse
from library.loss import MaskedMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(sequence, loop_type, pair_dist)

        # Compute loss (MaskedMSELoss handles the slicing internally)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set and calculates MCRMSE.
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

            # We need to slice predictions and targets to the scored length (68)
            # for MCRMSE calculation, as the metric is defined on scored positions.
            # MaskedMSELoss does this internally, but calculate_mcrmse expects raw tensors.
            preds_scored = preds[:, :PRED_LEN, :]
            targets_scored = targets[:, :PRED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    if not all_preds:
        return 0.0

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    mcrmse = calculate_mcrmse(all_preds, all_targets)
    return mcrmse.item()


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
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
            sample_ids = batch["id"]

            # Forward pass (Batch, 107, 3)
            preds = model(sequence, loop_type, pair_dist)
            preds = preds.cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                # Shape: (107, 3)
                sample_preds = preds[i]

                # We need to expand this to 107 rows per sample
                for seq_pos in range(SEQ_LEN):
                    row_id = f"{sample_id}_{seq_pos}"

                    # Get predicted values for this position
                    # Predicted columns: reactivity, deg_Mg_pH10, deg_Mg_50C
                    val_reactivity = sample_preds[seq_pos, 0]
                    val_deg_Mg_pH10 = sample_preds[seq_pos, 1]
                    val_deg_Mg_50C = sample_preds[seq_pos, 2]

                    # Store row data
                    # Required columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    # We fill deg_pH10 and deg_50C with 0.0 as they are not predicted
                    ids_list.append(row_id)
                    preds_list.append(
                        [
                            val_reactivity,
                            val_deg_Mg_pH10,
                            0.0,  # deg_pH10 (not predicted)
                            val_deg_Mg_50C,
                            0.0,  # deg_50C (not predicted)
                        ]
                    )

    # Create DataFrame
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    df_sub = pd.DataFrame(preds_list, columns=cols)
    df_sub.insert(0, "id_seqpos", ids_list)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug_sample_size=None, epochs=EPOCHS):
    """
    Main execution function for training and inference.
    """
    seed_everything(SEED)

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug_sample_size=debug_sample_size
    )

    # 2. Initialize Model
    print("Initializing model...")
    model = RNAModel().to(DEVICE)

    # 3. Setup Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MaskedMSELoss()

    best_mcrmse = float("inf")

    # 4. Training Loop
    print(f"Starting training for {epochs} epochs on {DEVICE}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_mcrmse = validate(model, val_loader, DEVICE)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | Train Loss (MSE): {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved! (MCRMSE: {best_mcrmse})")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    generate_submission(model, test_loader, DEVICE, SUBMISSION_FILE)
