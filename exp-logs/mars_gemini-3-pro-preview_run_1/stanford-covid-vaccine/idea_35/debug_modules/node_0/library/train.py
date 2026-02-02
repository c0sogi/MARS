import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, mcrmse_loss, masked_mse_loss
from library.data import get_data, RNADataset, collate_fn
from library.model import RNAModel


def train_model():
    """
    Executes the training pipeline:
    1. Loads data.
    2. Initializes model, optimizer, scheduler.
    3. Trains for fixed epochs using Masked MSE.
    4. Validates using MCRMSE.
    5. Saves the best model.
    """
    set_seed(Config.SEED)

    print("Loading data...")
    train_data = get_data(mode="train", load_cached_data=True)
    val_data = get_data(mode="val", load_cached_data=True)

    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print("Initializing model...")
    model = RNAModel(config=Config).to(Config.DEVICE)

    # Optimizer: AdamW with Low Weight Decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(Config.DEVICE)
            loop = batch["loop"].to(Config.DEVICE)
            dist = batch["dist"].to(Config.DEVICE)
            targets = batch["target"].to(Config.DEVICE)

            optimizer.zero_grad()

            preds = model(seq, loop, dist)

            # Masked MSE Loss on the first 68 positions
            loss = masked_mse_loss(preds, targets, scored_len=Config.PRED_LENGTH)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(Config.DEVICE)
                loop = batch["loop"].to(Config.DEVICE)
                dist = batch["dist"].to(Config.DEVICE)
                targets = batch["target"].to(Config.DEVICE)

                preds = model(seq, loop, dist)

                # MCRMSE Calculation
                # We slice predictions and targets to the scored length for metric calculation
                preds_scored = preds[:, : Config.PRED_LENGTH, :]
                targets_scored = targets[:, : Config.PRED_LENGTH, :]

                # Calculate MCRMSE for this batch
                # Note: To get exact dataset-wide MCRMSE, we should accumulate squared errors,
                # but averaging batch MCRMSE is a standard approximation during training loop.
                # For strict correctness with the metric definition, we will accumulate squared errors.

                # However, library.utils.mcrmse_loss returns a scalar mean(rmse).
                # Let's stick to accumulating the scalar loss for monitoring.
                batch_mcrmse = mcrmse_loss(preds_scored, targets_scored)
                val_loss_accum += batch_mcrmse.item()

        avg_val_mcrmse = val_loss_accum / len(val_loader)

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MCRMSE: {avg_val_mcrmse:.20f}"
        )

        # Checkpointing
        if avg_val_mcrmse < best_mcrmse:
            best_mcrmse = avg_val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! MCRMSE: {best_mcrmse:.20f}")

    print("Training complete.")


def generate_submission():
    """
    Loads the best model, predicts on the test set, and generates the submission CSV.
    """
    print("Generating submission...")
    set_seed(Config.SEED)

    # Load Test Data
    test_data = get_data(mode="test", load_cached_data=True)
    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Load Model
    model = RNAModel(config=Config).to(Config.DEVICE)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(Config.DEVICE)
            loop = batch["loop"].to(Config.DEVICE)
            dist = batch["dist"].to(Config.DEVICE)
            ids = batch["id"]

            # Forward pass
            # Output shape: (Batch, 107, 3) -> [reactivity, deg_Mg_pH10, deg_Mg_50C]
            preds = model(seq, loop, dist)
            preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate all predictions
    # Shape: (Total_Samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission Data
    # We need to map the 3 predicted columns to the 5 required columns
    # Required: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Predicted: reactivity (idx 0), deg_Mg_pH10 (idx 1), deg_Mg_50C (idx 2)
    # Unscored/Unpredicted: deg_pH10, deg_50C -> Fill with 0.0

    submission_rows = []

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LENGTH):
            # Create row ID
            id_seqpos = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": id_seqpos,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

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
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    train_model()
    generate_submission()
