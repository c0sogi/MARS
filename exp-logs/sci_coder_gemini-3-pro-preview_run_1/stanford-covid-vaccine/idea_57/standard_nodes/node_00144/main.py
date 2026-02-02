import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_dataloaders
from library.model import VectorScaledWideStreamBiGRU
from library.train import train_epoch, validate


def generate_submission(model, test_loader, device):
    """
    Generates the submission file for the test set.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_enc = batch["pair_enc"].to(device)

            # Predict full length (B, 107, 3)
            preds = model(seq, loop, pair_enc)

            all_preds.append(preds.cpu().numpy())

            # IDs are collated as a tuple of strings by the DataLoader
            if "id" in batch:
                all_ids.extend(batch["id"])
            else:
                raise ValueError("Batch does not contain IDs.")

    # Concatenate all predictions: (N_test, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    submission_rows = []

    # Map model output channels to submission columns
    # Model outputs: 0: reactivity, 1: deg_Mg_pH10, 2: deg_Mg_50C

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LENGTH):
            id_seqpos = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = float(sample_preds[seqpos, 0])
            deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Fill unscored targets with 0.0
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

    df_sub = pd.DataFrame(submission_rows)

    # Ensure correct column order
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = df_sub[cols]

    df_sub.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")


def run_pipeline():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Using cached data for speed as requested
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = VectorScaledWideStreamBiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.MSELoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mcrmse = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation Reporting
    # REQUIRED FORMAT: Final Validation Metric: <value>
    print(f"Final Validation Metric: {best_mcrmse}")

    # 7. Failure Analysis
    print("Performing failure analysis...")

    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Compute Validation Predictions
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_enc = batch["pair_enc"].to(device)
            targets = batch["target"].to(device)

            preds = model(seq, loop, pair_enc)

            # Slice to scored length for metric calculation
            preds_scored = preds[:, : Config.SCORED_LENGTH, :]
            targets_scored = targets[:, : Config.SCORED_LENGTH, :]

            val_preds.append(preds_scored.cpu().numpy())
            val_targets.append(targets_scored.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)  # (N, 68, 3)
    val_targets = np.concatenate(val_targets, axis=0)  # (N, 68, 3)

    # Calculate RMSE per sample (averaged over the 3 target columns)
    # MSE per sample per target
    mse_per_sample_target = np.mean((val_preds - val_targets) ** 2, axis=1)  # (N, 3)
    rmse_per_sample_target = np.sqrt(mse_per_sample_target)  # (N, 3)
    mean_rmse_per_sample = np.mean(rmse_per_sample_target, axis=1)  # (N,)

    # Load Metadata for correlation
    # We assume the order in val.parquet matches the validation loader (shuffle=False)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    if len(df_val) != len(mean_rmse_per_sample):
        print("Warning: Validation set size mismatch. Skipping correlation analysis.")
    else:
        print("\nCorrelation between Error (RMSE) and Features:")
        features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]

        for feat in features_to_check:
            if feat in df_val.columns:
                values = df_val[feat]
                if values.std() == 0:
                    print(f"  {feat}: N/A (Constant)")
                else:
                    # Handle potential NaNs
                    valid_mask = ~np.isnan(values)
                    corr, _ = pearsonr(
                        mean_rmse_per_sample[valid_mask], values[valid_mask]
                    )
                    print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not found in metadata")

    # 8. Submission Logic
    THRESHOLD = 0.6176461577
    if best_mcrmse < THRESHOLD:
        print(
            f"\nValidation metric {best_mcrmse} is better than threshold {THRESHOLD}."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric {best_mcrmse} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
