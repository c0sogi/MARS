import sys
import os
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.data import process_data, RNADataset
from library.model import REIDFN
from library.loss import MaskedMCRMSELoss
from library.train import train_one_epoch, generate_submission


def validate_mcrmse(model, loader, device):
    """
    Computes the exact Mean Columnwise Root Mean Squared Error (MCRMSE)
    on the validation set, strictly adhering to the competition metric.
    """
    model.eval()

    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    num_scored_cols = len(scored_indices)

    # Accumulators for Squared Error per column
    total_se_per_col = torch.zeros(num_scored_cols, device=device)
    total_count_per_col = torch.zeros(num_scored_cols, device=device)

    with torch.no_grad():
        for x, pairs, y, _ in loader:
            x = x.to(device)
            pairs = pairs.to(device)
            y = y.to(device)

            # Two-pass inference
            pred1 = model(x, pairs, prev_preds=None)
            pred2 = model(x, pairs, prev_preds=pred1)

            # Slice to scored length (68) and scored columns
            pred_scored = pred2[:, : Config.PRED_LEN, scored_indices]
            target_scored = y[:, : Config.PRED_LEN, scored_indices]

            # Compute Squared Error (B, L, 3)
            se = (pred_scored - target_scored) ** 2

            # Sum over Batch and Length dimensions -> (3,)
            total_se_per_col += se.sum(dim=(0, 1))

            # Count valid elements (B * L)
            batch_size = x.shape[0]
            valid_elements = batch_size * Config.PRED_LEN
            total_count_per_col += valid_elements

    # Compute RMSE per column
    mse_per_col = total_se_per_col / (total_count_per_col + 1e-12)
    rmse_per_col = torch.sqrt(mse_per_col)

    # MCRMSE is the mean of the column RMSEs
    mcrmse = rmse_per_col.mean().item()

    return mcrmse


def analyze_failures(model, loader, device):
    """
    Performs failure analysis by calculating per-sample error and correlating
    it with metadata features like Signal-to-Noise Ratio.
    """
    print("Performing failure analysis...")
    model.eval()

    sample_errors = []
    sample_ids = []
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for x, pairs, y, batch_ids in loader:
            x = x.to(device)
            pairs = pairs.to(device)
            y = y.to(device)

            pred1 = model(x, pairs, prev_preds=None)
            pred2 = model(x, pairs, prev_preds=pred1)

            pred_scored = pred2[:, : Config.PRED_LEN, scored_indices]
            target_scored = y[:, : Config.PRED_LEN, scored_indices]

            # Calculate MCRMSE per sample: Mean(Sqrt(Mean((y-y_hat)^2)))
            # Here we approximate 'error magnitude' as the RMSE averaged over columns for that sample
            mse = (pred_scored - target_scored) ** 2
            # Mean over length (dim 1) -> (B, 3)
            mse_per_col = mse.mean(dim=1)
            rmse_per_col = torch.sqrt(mse_per_col)
            # Mean over columns -> (B,)
            mcrmse_per_sample = rmse_per_col.mean(dim=1)

            sample_errors.extend(mcrmse_per_sample.cpu().numpy())
            sample_ids.extend(batch_ids)

    # Load Validation Metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping correlation analysis.")
        return

    val_df = pd.read_csv(val_meta_path)

    # Merge errors with metadata
    error_df = pd.DataFrame({"id": sample_ids, "error": sample_errors})
    merged_df = pd.merge(error_df, val_df, on="id")

    # Calculate Correlations
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "mean_reactivity",
        "seq_length",
    ]

    for feature in features_to_check:
        if feature in merged_df.columns:
            # Drop NaNs if any
            valid_data = merged_df[["error", feature]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data["error"], valid_data[feature])
                print(f"Correlation (Error vs {feature}): {corr:.6f}")


def run():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast Baseline Settings
    Config.EPOCHS = 10  # Sufficient for small dataset (1728 samples)
    Config.BATCH_SIZE = 16

    # 2. Data Loading
    # We use the full dataset as it is small enough for the time limit.
    train_dict, val_dict, test_dict = process_data(load_cached_data=True)

    train_ds = RNADataset(train_dict)
    val_ds = RNADataset(val_dict)
    test_ds = RNADataset(test_dict, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = REIDFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MaskedMCRMSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model_runfile.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate_mcrmse(model, val_loader, device)

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_val_score = validate_mcrmse(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.47142532743789534

    if final_val_score < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score {final_val_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
