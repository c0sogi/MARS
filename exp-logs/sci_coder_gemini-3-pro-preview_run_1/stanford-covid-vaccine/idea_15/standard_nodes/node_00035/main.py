import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.model import RNAModel
from library.dataset import prepare_data
from library.utils import set_seed, mcrmse_metric, build_submission_df


def main():
    # 1. Configuration and Setup
    config = Config()

    # Fast Baseline Overrides
    config.EPOCHS = 10  # Reduced from 25 for quick baseline execution

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Preparation
    print("Loading data...")
    datasets = prepare_data(config, load_cached_data=True)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNAModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    # 4. Training Loop
    print(f"Starting training for {config.EPOCHS} epochs on {device}...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 3)

            optimizer.zero_grad()

            # Forward pass
            preds = model(seq, loop, dist)  # (B, 107, 3)

            # Slice to scored positions
            preds_scored = preds[:, : config.SEQ_SCORED, :]

            loss = criterion(preds_scored, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation Step
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, dist)
                preds_scored = preds[:, : config.SEQ_SCORED, :]

                all_preds.append(preds_scored.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_mcrmse = mcrmse_metric(all_targets, all_preds)

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation & Metric
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    val_ids = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)
            preds_scored = preds[:, : config.SEQ_SCORED, :]

            val_preds.append(preds_scored.cpu())
            val_targets.append(targets.cpu())
            val_ids.extend(ids)

    val_preds_t = torch.cat(val_preds, dim=0)
    val_targets_t = torch.cat(val_targets, dim=0)

    final_metric = mcrmse_metric(val_targets_t, val_preds_t)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate RMSE per sample
    # val_preds_t: (N, 68, 3), val_targets_t: (N, 68, 3)
    diff = (val_preds_t - val_targets_t).numpy()
    mse_per_sample = np.mean(diff**2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create DataFrame for analysis
    error_df = pd.DataFrame({"id": val_ids, "rmse": rmse_per_sample})

    # Load metadata to get features
    if os.path.exists(config.VAL_PARQUET):
        val_meta_df = pd.read_parquet(config.VAL_PARQUET)
        # Merge on ID
        analysis_df = pd.merge(error_df, val_meta_df, on="id", how="inner")

        # Calculate correlations
        features_to_check = ["signal_to_noise", "SN_filter"]
        print("Correlation between Error (RMSE) and Features:")

        for feat in features_to_check:
            if feat in analysis_df.columns:
                # Drop NaNs if any
                valid_data = analysis_df[[feat, "rmse"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                    print(f"  {feat}: {corr:.4f}")
                else:
                    print(f"  {feat}: Not enough data")
            else:
                print(f"  {feat}: Column not found")
    else:
        print("Validation metadata not found, skipping detailed correlation analysis.")

    # 7. Submission Generation
    threshold = 0.6226052641868591
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        test_loader = DataLoader(
            datasets["test"],
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        all_test_preds = []
        all_test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                ids = batch["id"]

                # Predict full sequence
                preds = model(seq, loop, dist)  # (B, 107, 3)

                all_test_preds.append(preds.cpu())
                all_test_ids.extend(ids)

        all_test_preds = torch.cat(all_test_preds, dim=0)

        submission_df = build_submission_df(
            all_test_ids, all_test_preds, seq_len=config.SEQ_LENGTH
        )
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
