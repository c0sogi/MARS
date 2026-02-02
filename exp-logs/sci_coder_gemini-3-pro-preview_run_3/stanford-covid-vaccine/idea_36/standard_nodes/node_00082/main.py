import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided library
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_loaders
from library.model import DDPNBiGRU
from library.train import train_epoch, validate, generate_submission_file

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline and Path Requirements
    Config.EPOCHS = 15  # Limit epochs for speed
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set reproducible seed
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Load loaders with caching enabled
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = DDPNBiGRU().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"Starting training for {Config.EPOCHS} epochs...")

    best_mcrmse = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_mcrmse, val_scores = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\nRunning Final Evaluation & Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    # A. Calculate Final Metric
    final_mcrmse, final_scores = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_mcrmse}")

    # B. Failure Analysis
    # We need sample-wise predictions and targets to correlate with metadata
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            y = batch["y"]  # CPU
            ids = batch["id"]

            preds = model(X, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(y)
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 107, 5)

    # Calculate sample-wise RMSE for scored columns
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    # Slice to scored sequence length
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, scored_indices]
    targets_sliced = all_targets[:, : Config.SEQ_SCORED, scored_indices]

    # Squared error: (N, 68, 3)
    sq_error = (preds_sliced - targets_sliced) ** 2
    # Mean over seq_len and targets -> (N,)
    mse_per_sample = torch.mean(sq_error, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Metadata to correlate
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    # Ensure alignment by ID
    val_df = val_df.set_index("id").loc[all_ids].reset_index()

    # Add error to dataframe
    val_df["model_rmse"] = rmse_per_sample

    # Calculate correlations
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = {}

    if "signal_to_noise" in val_df.columns:
        corr = val_df["model_rmse"].corr(val_df["signal_to_noise"])
        correlations["signal_to_noise"] = corr

    if "SN_filter" in val_df.columns:
        corr = val_df["model_rmse"].corr(val_df["SN_filter"])
        correlations["SN_filter"] = corr

    # Sequence length is constant (107), so no correlation needed

    for feature, corr in correlations.items():
        print(f"  Correlation with {feature}: {corr:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.5978901386

    if final_mcrmse < THRESHOLD:
        print(f"\nMetric {final_mcrmse:.6f} < {THRESHOLD}. Generating submission...")
        generate_submission_file(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {final_mcrmse:.6f} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
