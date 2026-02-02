import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.dataset import RNADataset
from library.model import SpectralTopologicalBiGRU
from library.engine import set_seed, train_fn, eval_fn, generate_submission
from library.utils import mcrmse_loss


def main():
    # 1. Configuration Overrides and Setup
    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # Override Config paths to match requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    # Ensure we use the working directory for intermediate files
    Config.WORKING_DIR = "./working"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Using load_cached_data=True as requested
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    val_dataset = RNADataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SpectralTopologicalBiGRU()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 4. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        train_loss = train_fn(model, train_loader, optimizer, device)

        # Evaluate
        val_score = eval_fn(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

        # Simple log
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

    print("-" * 30)
    print(f"Final Validation Metric: {best_score}")
    print("-" * 30)

    # 5. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")

    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and targets manually to calculate per-sample error
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            lpe = batch["lpe"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(sequence, loop_type, pair_dist, lpe)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()

    # Calculate RMSE per sample
    # Error = (y - y_hat)^2
    sq_diff = (all_targets - all_preds) ** 2
    # Apply mask: set invalid positions to nan or 0, but here we want to average over valid
    # Expand mask for broadcasting: (N, 107) -> (N, 107, 3)
    mask_expanded = np.expand_dims(all_masks, axis=-1)

    # Zero out invalid positions
    sq_diff = sq_diff * mask_expanded

    # Sum squared errors per sample (sum over seq_len and targets)
    sum_sq_error_per_sample = np.sum(sq_diff, axis=(1, 2))

    # Count valid entries per sample
    count_per_sample = np.sum(mask_expanded, axis=(1, 2))

    # RMSE per sample
    rmse_per_sample = np.sqrt(sum_sq_error_per_sample / (count_per_sample + 1e-8))

    # Load Validation Metadata for correlations
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Extract features for correlation
    # 1. Signal to Noise
    if "signal_to_noise" in df_val.columns:
        sn_ratio = df_val["signal_to_noise"].values
    else:
        sn_ratio = np.zeros(len(df_val))

    # 2. GC Content
    gc_content = (
        df_val["sequence"]
        .apply(lambda x: (x.count("G") + x.count("C")) / len(x))
        .values
    )

    # 3. Paired Percentage
    paired_pct = (
        df_val["structure"]
        .apply(lambda x: (x.count("(") + x.count(")")) / len(x))
        .values
    )

    # Calculate Correlations
    # Ensure lengths match (dataset should not shuffle val)
    if len(rmse_per_sample) == len(df_val):
        corr_sn, _ = pearsonr(rmse_per_sample, sn_ratio)
        corr_gc, _ = pearsonr(rmse_per_sample, gc_content)
        corr_pair, _ = pearsonr(rmse_per_sample, paired_pct)

        print(f"Correlation (Error vs Signal-to-Noise): {corr_sn:.4f}")
        print(f"Correlation (Error vs GC Content):      {corr_gc:.4f}")
        print(f"Correlation (Error vs Paired %):        {corr_pair:.4f}")
    else:
        print(
            "Warning: Mismatch in validation sample counts, skipping correlation analysis."
        )

    # 6. Conditional Submission
    THRESHOLD = 0.6176461577
    if best_score < THRESHOLD:
        print(f"\nValidation metric {best_score} is better than threshold {THRESHOLD}.")
        # generate_submission loads the best model from disk automatically
        generate_submission(device)
    else:
        print(
            f"\nValidation metric {best_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
