import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNADilatedNet
from library.train import train_one_epoch, validate, generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_ids = []
    all_errors = []

    # 1. Calculate per-sample error
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            targets = batch["targets"].to(device)  # (B, 107, 5)
            ids = batch["id"]

            preds = model(seq, struct, loop)  # (B, 107, 5)

            # Focus on scored positions
            preds_scored = preds[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            # Calculate RMSE per sample (averaging over positions and targets)
            # MSE per sample: (B, 68, 5) -> (B,)
            mse_per_sample = torch.mean(
                (preds_scored - targets_scored) ** 2, dim=(1, 2)
            )
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_ids.extend(ids)
            all_errors.extend(rmse_per_sample.cpu().numpy())

    # 2. Create Error DataFrame
    error_df = pd.DataFrame({"id": all_ids, "rmse": all_errors})

    # 3. Load Metadata
    if not os.path.exists(Config.VAL_PATH):
        print("Validation metadata not found. Skipping correlation analysis.")
        return

    val_meta = pd.read_parquet(Config.VAL_PATH)

    # 4. Merge
    analysis_df = pd.merge(error_df, val_meta, on="id", how="inner")

    # 5. Calculate Correlations
    # Select numeric columns of interest
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]
    # Add derived features
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))

    features_to_check.extend(["len_A", "len_G", "len_U", "len_C"])

    print("-" * 40)
    print("Correlation between Error (RMSE) and Features:")
    print("-" * 40)

    correlations = {}
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Handle potential NaN or non-numeric issues
            try:
                corr = analysis_df["rmse"].corr(analysis_df[feat].astype(float))
                correlations[feat] = corr
                print(f"{feat:<20}: {corr:.4f}")
            except Exception as e:
                print(f"{feat:<20}: Could not calculate (Error: {e})")

    print("-" * 40)


def main():
    # 1. Configuration & Setup
    # Override epochs for a faster baseline run
    Config.EPOCHS = 15
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNADilatedNet(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)
        scheduler.step()

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

    print("Training finished.")

    # 5. Final Validation & Metric
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.7462618350982666
    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
