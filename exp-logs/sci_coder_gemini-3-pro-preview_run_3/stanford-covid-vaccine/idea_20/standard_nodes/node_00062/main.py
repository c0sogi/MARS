import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, global_mcrmse
from library.data import get_loaders
from library.model import CGSRBiGRU
from library.train import train_epoch, validate, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline and Path Requirements
    Config.epochs = 10  # Reduced for fast baseline
    Config.submission_path = "./submission/submission.csv"
    Config.model_save_path = "./working/best_model_runfile.pth"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Set Seed
    seed_everything(Config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = CGSRBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    criterion = MCRMSELoss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_score = float("inf")

    print(f"Starting training for {Config.epochs} epochs...")
    for epoch in range(Config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    print(f"Training complete. Best Validation Score: {best_score}")

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\nRunning Final Evaluation and Failure Analysis...")

    # Load Best Model
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.eval()

    # Re-run validation to get predictions for analysis
    all_preds = []
    all_targets = []

    # We assume val_loader is not shuffled (shuffle=False in library/data.py)
    # This aligns with metadata/val.parquet
    with torch.no_grad():
        for batch in val_loader:
            features, adjacency, targets, masks = batch

            features = features.to(device)
            adjacency = adjacency.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(features, adjacency)

            # Keep data on CPU for analysis
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 107, 5)

    # Compute Global Metric (Masked)
    # We need to apply the mask logic consistent with the metric definition
    # The masks are generated in data.py based on seq_scored (68)
    # We can slice the first 68 positions
    seq_scored = Config.seq_scored
    preds_scored = all_preds[:, :seq_scored, :]
    targets_scored = all_targets[:, :seq_scored, :]

    # Compute MCRMSE manually to ensure correctness for printing
    mse = np.mean((preds_scored - targets_scored) ** 2, axis=0)  # Mean over N*SeqScored
    rmse = np.sqrt(mse)
    final_metric = np.mean(rmse)  # Mean over 5 targets

    # Flatten for global calculation if needed, but the above is correct for (N, L, C) -> mean over N, L
    # Actually, MCRMSE formula is: average over columns of (sqrt(mean((y-yhat)^2)))
    # We flatten N and L dimensions together
    preds_flat = preds_scored.reshape(-1, 5)
    targets_flat = targets_scored.reshape(-1, 5)
    mse_flat = np.mean((preds_flat - targets_flat) ** 2, axis=0)
    rmse_flat = np.sqrt(mse_flat)
    final_metric = np.mean(rmse_flat)

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Load Metadata
    try:
        val_meta_path = os.path.join(Config.metadata_dir, "val.parquet")
        val_df = pd.read_parquet(val_meta_path)

        # Calculate Sample-wise Error (Mean RMSE across targets and positions)
        # Shape: (N, 68, 5)
        diff = preds_scored - targets_scored
        sq_diff = diff**2
        # Mean over positions (axis 1) and targets (axis 2)
        sample_mse = np.mean(sq_diff, axis=(1, 2))
        sample_rmse = np.sqrt(sample_mse)

        val_df["model_error"] = sample_rmse

        # Feature Engineering for Correlation
        val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
        val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
        val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))
        val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))

        analysis_cols = [
            "signal_to_noise",
            "SN_filter",
            "pct_A",
            "pct_G",
            "pct_C",
            "pct_U",
        ]

        print("\nFailure Analysis (Correlation with Model Error):")
        print(f"{'Feature':<20} {'Correlation':<10}")
        print("-" * 35)

        for col in analysis_cols:
            if col in val_df.columns:
                # Drop NaNs just in case
                valid_data = val_df[[col, "model_error"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[col], valid_data["model_error"])
                    print(f"{col:<20} {corr:.4f}")
                else:
                    print(f"{col:<20} N/A")

    except Exception as e:
        print(f"Failure analysis skipped due to error: {e}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader, device)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
