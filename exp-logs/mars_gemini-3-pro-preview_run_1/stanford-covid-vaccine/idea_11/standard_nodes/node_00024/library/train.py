import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything
from library.data import load_and_process_data
from library.model import RNAResidualBiGRU


def train_model():
    """
    Trains the RNAResidualBiGRU and saves the best checkpoint.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    datasets = load_and_process_data(load_cached_data=True)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model, Optimizer, Scheduler
    model = RNAResidualBiGRU(config=Config).to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # We use MSE Loss for regression
    criterion = nn.MSELoss()

    best_val_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

            optimizer.zero_grad()

            # Forward pass (returns B, 107, 3)
            preds = model(seq, loop, pair)

            # Crop to scored length (68)
            preds_scored = preds[:, : Config.SCORABLE_LENGTH, :]

            # Compute Loss (MSE)
            loss = criterion(preds_scored, targets)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        scheduler.step()

        # --- Validation ---
        model.eval()

        # Accumulators for exact MCRMSE calculation
        # We need sum of squared errors for each of the (68 * 3) columns
        total_squared_error = torch.zeros(Config.SCORABLE_LENGTH, 3).to(device)
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                pair = batch["pair_index"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, pair)
                preds_scored = preds[:, : Config.SCORABLE_LENGTH, :]

                # Squared error per element: (B, 68, 3)
                squared_errors = (preds_scored - targets) ** 2

                # Sum over batch dimension -> (68, 3)
                total_squared_error += squared_errors.sum(dim=0)
                total_samples += seq.size(0)

        # Calculate RMSE per column: sqrt(sum_sq_err / N)
        rmse_per_col = torch.sqrt(total_squared_error / total_samples)

        # MCRMSE is the mean of these RMSEs
        val_mcrmse = rmse_per_col.mean().item()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpoint
        if val_mcrmse < best_val_mcrmse:
            best_val_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best model saved! ({val_mcrmse})")

    print(f"Training finished. Best Validation MCRMSE: {best_val_mcrmse}")


def generate_submission():
    """
    Loads the best model, predicts on test set, and generates submission.csv.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Cannot generate submission.")
        return

    print("Loading best model for inference...")
    model = RNAResidualBiGRU(config=Config).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Test Data
    datasets = load_and_process_data(load_cached_data=True)
    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print("Generating predictions on test set...")

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair = batch["pair_index"].to(device)
            ids = batch["id"]  # List of strings

            # Forward pass (B, 107, 3)
            preds = model(seq, loop, pair)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions: (N_test, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission Data
    # Columns in preds: 0: reactivity, 1: deg_Mg_pH10, 2: deg_Mg_50C
    # Submission requires: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_rows = []

    print("Formatting submission file...")
    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            reactivity = float(sample_preds[seqpos, 0])
            deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Unscored columns set to 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_pipeline():
    train_model()
    generate_submission()
