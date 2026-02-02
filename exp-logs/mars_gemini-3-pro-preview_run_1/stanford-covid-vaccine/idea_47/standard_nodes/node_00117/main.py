import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.stats as stats

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.model import RNAModel
from library.data import get_dataloaders
from library.train import train_one_epoch, validate, generate_submission
from library.utils import set_seed


def run_pipeline():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Load cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel(config=Config).to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Execute training for the defined number of epochs
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.CLIP_GRAD
        )
        val_mcrmse = validate(model, val_loader, device)
        scheduler.step()

        # Save best model based on validation metric
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Metric Reporting
    print(f"Final Validation Metric: {best_mcrmse}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Calculate per-sample error on validation set
    val_errors = []
    val_ids_check = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)
            targets = batch["target"].to(device)

            outputs = model(seq, loop, dist)
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            # Calculate MCRMSE per sample: Mean of RMSEs across the 3 target columns
            diff_sq = (outputs_scored - targets) ** 2
            mse_per_col = torch.mean(diff_sq, dim=1)  # Average over sequence length
            rmse_per_col = torch.sqrt(mse_per_col)
            mcrmse_per_sample = torch.mean(rmse_per_col, dim=1)

            val_errors.extend(mcrmse_per_sample.cpu().numpy())
            val_ids_check.extend(batch["id"])

    # Load validation metadata to correlate errors with features
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Map calculated errors to the dataframe
    error_map = dict(zip(val_ids_check, val_errors))
    df_val["error"] = df_val["id"].map(error_map)

    # Define features for correlation analysis
    features_to_check = ["signal_to_noise", "SN_filter"]

    # Add derived sequence composition features
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))
    features_to_check.extend(["len_A", "len_G", "len_C", "len_U"])

    print("Correlations with Error:")
    for feat in features_to_check:
        if feat in df_val.columns:
            valid_data = df_val[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.6176461577
    if best_mcrmse < THRESHOLD:
        print(
            f"Validation metric {best_mcrmse} < {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(f"Validation metric {best_mcrmse} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
