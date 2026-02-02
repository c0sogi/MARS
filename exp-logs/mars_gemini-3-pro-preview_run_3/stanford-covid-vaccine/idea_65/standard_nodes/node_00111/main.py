import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders
from library.model import HC_WG_BiGRU
from library.train import train_one_epoch, validate, generate_submission, MCRMSELoss


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for fast baseline execution
    # The dataset is small (1728 train), so we use the full dataset but limit epochs.
    Config.EPOCHS = 15
    Config.DEBUG = False

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print(f"Device: {device}")
    print(f"Training for {Config.EPOCHS} epochs on full dataset.")

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = HC_WG_BiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f} | Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 5. Final Validation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    model.eval()
    val_preds = []
    val_targets = []

    # Collect predictions and targets
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU

            outputs = model(features, pair_indices, pair_mask)
            val_preds.append(outputs.cpu())
            val_targets.append(targets)

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    # Calculate sample-wise MCRMSE contribution
    # Slice to scored length and columns
    vp = val_preds[:, : Config.PRED_LEN, Config.SCORED_INDICES]
    vt = val_targets[:, : Config.PRED_LEN, Config.SCORED_INDICES]

    # MSE per column per sample (average over sequence length)
    mse_per_col = torch.mean((vp - vt) ** 2, dim=1)  # Shape: (N, 3)
    # RMSE per column per sample
    rmse_per_col = torch.sqrt(mse_per_col)  # Shape: (N, 3)
    # Average RMSE across columns for each sample
    sample_errors = torch.mean(rmse_per_col, dim=1).numpy()  # Shape: (N,)

    # Load Metadata for correlation
    val_df = pd.read_parquet(Config.VAL_METADATA)
    val_df["error_metric"] = sample_errors

    # Feature Engineering for Analysis
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # Calculate Correlations
    analysis_cols = ["error_metric", "signal_to_noise", "SN_filter", "gc_content"]
    correlations = val_df[analysis_cols].corr()["error_metric"].drop("error_metric")

    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission Generation
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        print(
            f"\nValidation Score ({final_val_score}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        submission_df = generate_submission(model, test_loader, device)

        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nValidation Score ({final_val_score}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
