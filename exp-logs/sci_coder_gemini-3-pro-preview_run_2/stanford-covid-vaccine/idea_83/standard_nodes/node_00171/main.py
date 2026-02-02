import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats as stats

# Ensure library is in path
sys.path.append("./library")

from library.config import Config
from library.utils import set_seed
from library.loss import MaskedMCRMSE
from library.data import get_loader
from library.model import HC_HIGFN
from library.train import train_one_epoch, validate, generate_submission


def run():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for rapid execution
    Config.EPOCHS = 5
    Config.PATIENCE = 5

    # 2. Data Loading
    # Using the provided loaders which handle caching and preprocessing
    train_loader = get_loader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_loader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    model = HC_HIGFN().to(device)
    criterion = MaskedMCRMSE()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.GRAD_CLIP
        )

        # Validate
        val_mcrmse, val_metrics = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_mcrmse)

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("Training complete.")

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on full validation set
    final_mcrmse, final_metrics = validate(model, val_loader, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_mcrmse}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    # Calculate per-sample RMSE
    all_sample_rmses = []

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward pass
            y_pred = model(inputs, partner_indices)

            # Apply masking logic manually to get per-sample error
            # 1. Sequence Masking
            valid_len = min(y_pred.shape[1], Config.SCORED_SEQ_LEN)
            pred_valid = y_pred[:, :valid_len, :]
            target_valid = targets[:, :valid_len, :]

            # 2. Column Masking (Select scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C)
            pred_scored = pred_valid[:, :, Config.SCORED_TARGET_INDICES]
            target_scored = target_valid[:, :, Config.SCORED_TARGET_INDICES]

            # Compute MSE per sample (average over sequence length and columns)
            # Shape: (Batch, Seq, Cols) -> (Batch,)
            mse_per_sample = torch.mean((pred_scored - target_scored) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_sample_rmses.extend(rmse_per_sample.cpu().numpy())

    # Load metadata to correlate errors with features
    val_metadata = pd.read_csv(Config.VAL_METADATA)

    # Add errors to dataframe
    # Note: val_loader is shuffle=False, so order matches metadata
    if len(all_sample_rmses) == len(val_metadata):
        val_metadata["sample_rmse"] = all_sample_rmses

        # Correlation with Signal to Noise
        if "signal_to_noise" in val_metadata.columns:
            corr_sn, _ = stats.pearsonr(
                val_metadata["signal_to_noise"], val_metadata["sample_rmse"]
            )
            print(f"Correlation (Error vs Signal_to_Noise): {corr_sn:.4f}")

        # Correlation with Mean Reactivity
        if "mean_reactivity" in val_metadata.columns:
            corr_mr, _ = stats.pearsonr(
                val_metadata["mean_reactivity"], val_metadata["sample_rmse"]
            )
            print(f"Correlation (Error vs Mean_Reactivity): {corr_mr:.4f}")
    else:
        print(
            "Warning: Mismatch between validation set size and metadata rows. Skipping correlation analysis."
        )

    # 7. Submission
    THRESHOLD = 0.47142532743789534

    if final_mcrmse < THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Metric {final_mcrmse} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
