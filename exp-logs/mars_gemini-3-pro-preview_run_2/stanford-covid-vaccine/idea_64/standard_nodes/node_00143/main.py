import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import ML_GFN
from library.loss import MaskedMCRMSELoss
from library.train import train_one_epoch, validate, generate_submission


def run_pipeline():
    # =========================================================================
    # 1. Setup & Configuration Override
    # =========================================================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Override Config for Fast Baseline
    # We reduce the number of epochs to ensure the script completes quickly
    # while still allowing for convergence on this small dataset.
    Config.NUM_EPOCHS = 15

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing Data Loaders...")
    # load_cached_data=True ensures we use preprocessed .npz files if available
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing ML-GFN Model...")
    model = ML_GFN().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = MaskedMCRMSELoss().to(device)

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")

    print(f"Starting Training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score, val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        # Logging (minimal)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

    # =========================================================================
    # 5. Final Validation & Metric Reporting
    # =========================================================================
    print("-" * 30)
    print(f"Final Validation Metric: {best_score}")
    print("-" * 30)

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis on Validation Set...")

    # Load the best model for analysis
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_ids = []
    val_errors = []

    # Identify scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Two-pass Inference
            z = model.encode_static(inputs)
            y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)
            y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1)

            # Compute RMSE per sample for correlation analysis
            preds_scored = y2[..., scored_indices]
            targets_scored = targets[..., scored_indices]

            # Iterate over batch to calculate per-sample error
            for i in range(len(ids)):
                # Apply mask for this sample
                m = mask[i] > 0
                p = preds_scored[i][m]
                t = targets_scored[i][m]

                if p.shape[0] > 0:
                    mse = torch.mean((p - t) ** 2)
                    rmse = torch.sqrt(mse).item()
                    val_ids.append(ids[i])
                    val_errors.append(rmse)
                else:
                    # Should not happen given data filtering, but safe fallback
                    val_ids.append(ids[i])
                    val_errors.append(0.0)

    # Load validation metadata to retrieve features
    val_metadata = pd.read_csv(Config.VAL_CSV)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": val_ids, "rmse_error": val_errors})

    # Merge with metadata
    analysis_df = val_metadata.merge(error_df, on="id")

    # Calculate GC Content
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0
    )

    # Calculate Correlations
    # 1. Error vs Signal-to-Noise
    if "signal_to_noise" in analysis_df.columns:
        corr_sn, _ = pearsonr(analysis_df["rmse_error"], analysis_df["signal_to_noise"])
        print(f"Correlation (Error vs Signal_to_Noise): {corr_sn:.4f}")

    # 2. Error vs GC Content
    corr_gc, _ = pearsonr(analysis_df["rmse_error"], analysis_df["gc_content"])
    print(f"Correlation (Error vs GC_Content): {corr_gc:.4f}")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    THRESHOLD = 0.47142532743789534

    if best_score < THRESHOLD:
        print(
            f"Validation Score ({best_score}) is better than threshold ({THRESHOLD}). Generating Submission..."
        )
        # Use the provided function in library/train.py
        generate_submission()
    else:
        print(
            f"Validation Score ({best_score}) did not meet threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    run_pipeline()
