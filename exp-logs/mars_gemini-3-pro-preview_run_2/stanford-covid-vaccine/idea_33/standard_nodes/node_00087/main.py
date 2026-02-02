import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device, mcrmse_numpy
from library.data import get_dataloaders
from library.model import PFR_DN
from library.loss import MCRMSELoss
from library.train import train_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for features, partner_indices, targets in val_loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Two-Stage Inference
            y_hat_1 = model(features, partner_indices, recycling=None)
            y_hat_2 = model(features, partner_indices, recycling=y_hat_1)

            # Slice to scored length
            preds_scored = y_hat_2[:, : Config.SCORED_SEQ_LENGTH, :].cpu().numpy()
            targets_scored = targets[:, : Config.SCORED_SEQ_LENGTH, :].cpu().numpy()

            # Filter to scored columns (indices [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C)
            scored_indices = [
                i
                for i, col in enumerate(Config.TARGET_COLS)
                if col in Config.SCORED_COLS
            ]

            preds_scored = preds_scored[:, :, scored_indices]
            targets_scored = targets_scored[:, :, scored_indices]

            all_preds.append(preds_scored)
            all_targets.append(targets_scored)

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 68, 3)
    all_targets = np.concatenate(all_targets, axis=0)  # (N_samples, 68, 3)
    all_ids = val_loader.dataset.ids

    # 2. Calculate MCRMSE per sample
    # Error per sample = mean(sqrt(mean((y-y_hat)^2, axis=0))) -- roughly
    # We calculate RMSE per column per sample, then mean across columns
    mse_per_sample_col = np.mean((all_targets - all_preds) ** 2, axis=1)  # (N, 3)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)  # (N, 3)
    mcrmse_per_sample = np.mean(rmse_per_sample_col, axis=1)  # (N,)

    # 3. Load Metadata to correlate
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment by ID
    analysis_df = pd.DataFrame({"id": all_ids, "error": mcrmse_per_sample})

    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # 4. Calculate Correlations
    # We look at signal_to_noise, SN_filter, and potentially sequence composition
    # Helper to get numeric columns
    numeric_cols = ["signal_to_noise", "SN_filter", "mean_reactivity", "error"]
    corr_df = merged_df[numeric_cols].corr()

    print("Correlation of Error with Metadata Features:")
    print(corr_df["error"].drop("error").sort_values(ascending=False))

    return mcrmse_per_sample.mean()


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for Fast Baseline & Submission Path
    Config.EPOCHS = 15  # Reduced from 50 for speed
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = PFR_DN().to(device)
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_metric = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_metric)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:02d} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val MCRMSE: {val_metric:.8f}"
        )

        # Save Best
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))

    # Calculate Final Metric on full validation set
    _, final_val_metric = validate(model, val_loader, criterion, device)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5417620723771521

    if final_val_metric < THRESHOLD:
        print(
            f"\nMetric ({final_val_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_val_metric:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
