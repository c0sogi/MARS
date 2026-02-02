import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data import load_data
from library.model import RCRDN, generate_submission
from library.loss import MCRMSELoss
from library.train import set_seed, train_epoch, validate


def pearson_corr(x, y):
    """Calculate Pearson correlation coefficient using numpy."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline constraints
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32

    # Ensure submission directory exists and update path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Set device and reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Use cached data if available
    train_dataset = load_data("train", load_cached_data=True)
    val_dataset = load_data("val", load_cached_data=True)

    # num_workers=0 for safe execution in single script file
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = RCRDN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Weighted loss for recycling steps [pass_1, pass_2]
    criterion = MCRMSELoss(weights=[0.5, 1.0])

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_loss = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Calculate final metric on full validation set
    final_metric = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")

    # Load validation metadata to access 'signal_to_noise'
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Collect all predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, p_idx, p_mask, y in val_loader:
            x, p_idx, p_mask = x.to(device), p_idx.to(device), p_mask.to(device)

            preds_list = model(x, p_idx, p_mask)
            final_pred = preds_list[-1]  # (B, 5, L)

            all_preds.append(final_pred.cpu().numpy())
            all_targets.append(y.numpy())

    # Concatenate batches
    preds_np = np.concatenate(all_preds, axis=0)  # (N, 5, L)
    targets_np = np.concatenate(all_targets, axis=0)  # (N, L, 5)

    # Transpose preds to (N, L, 5) to match targets
    preds_np = preds_np.transpose(0, 2, 1)

    # Slice to scored region and scored columns
    # Scored Length: 68, Scored Indices: [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    preds_scored = preds_np[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]
    targets_scored = targets_np[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]

    # Calculate RMSE per sample
    # 1. MSE per sample per column (mean over sequence length axis=1)
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=1)  # (N, 3)
    # 2. RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)  # (N, 3)
    # 3. Mean RMSE per sample (average over the 3 scored columns)
    sample_errors = np.mean(rmse_per_col, axis=1)  # (N,)

    # Correlation with Signal to Noise
    if "signal_to_noise" in val_df.columns:
        sn_values = val_df["signal_to_noise"].values
        if len(sn_values) == len(sample_errors):
            corr = pearson_corr(sample_errors, sn_values)
            print(
                f"Failure Analysis: Correlation between Error and Signal-to-Noise: {corr}"
            )
        else:
            print("Warning: Metadata length mismatch for failure analysis.")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, device)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
