import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, get_device, mcrmse
from library.data import get_dataloaders
from library.model import RNARegressor
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # Using debug=False to ensure we train on the full dataset for the best possible score.
    # The dataset is small enough (~2k samples) to train quickly within the time limit.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=False,
    )

    # 3. Model Initialization
    model = RNARegressor().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing over fixed epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Final Evaluation
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Run inference on validation set to get predictions for analysis
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            seq_ids = batch["seq_ids"].to(device)
            loop_ids = batch["loop_ids"].to(device)
            pair_emb = batch["pair_emb"].to(device)
            pos_emb = batch["pos_emb"].to(device)
            targets = batch["targets"].cpu().numpy()

            preds = model(seq_ids, loop_ids, pair_emb, pos_emb)
            preds = preds.cpu().numpy()

            # Slice to scored length (68)
            val_preds_list.append(preds[:, : Config.PRED_LEN, :])
            val_targets_list.append(targets[:, : Config.PRED_LEN, :])

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Compute Final Metric
    final_metric = mcrmse(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Load validation metadata to correlate errors with features
    df_val = pd.read_parquet(Config.VAL_FILE)

    # Ensure alignment
    if len(df_val) != len(val_preds):
        df_val = df_val.iloc[: len(val_preds)]

    # Calculate RMSE per sample (averaged over positions and channels)
    # Shape: (N, 68, 3)
    mse_per_sample = np.mean((val_targets - val_preds) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    print("Failure Analysis (Correlation with Error Magnitude):")

    # Feature 1: Signal to Noise
    if "signal_to_noise" in df_val.columns:
        corr, _ = pearsonr(df_val["signal_to_noise"], rmse_per_sample)
        print(f"  signal_to_noise: {corr:.4f}")

    # Feature 2: SN_filter
    if "SN_filter" in df_val.columns:
        corr, _ = pearsonr(df_val["SN_filter"], rmse_per_sample)
        print(f"  SN_filter: {corr:.4f}")

    # Feature 3: Nucleotide Counts (A, G, C, U)
    for nuc in ["A", "G", "C", "U"]:
        count = df_val["sequence"].apply(lambda x: x.count(nuc))
        corr, _ = pearsonr(count, rmse_per_sample)
        print(f"  len_{nuc}: {corr:.4f}")

    # 8. Submission Generation
    threshold = 0.6176461577
    if final_metric < threshold:
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(
            f"Metric {final_metric} is not lower than {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
