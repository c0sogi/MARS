import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config, set_seed
from library.data import get_loaders
from library.model import RNA_ResBiLSTM
from library.engine import run_training, eval_fn
from library.utils import mcrmse_metric

# Suppress warnings
warnings.filterwarnings("ignore")


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    # 1. Collect Predictions and Targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, dist)

            # Slice to scored length
            preds_scored = preds[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # (N_samples, 68, 3)
    all_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate Per-Sample Error (RMSE per sample, averaged over columns/positions)
    # Shape: (N_samples,)
    mse_per_sample = torch.mean((all_preds - all_targets) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata for Correlation
    val_parquet_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if not os.path.exists(val_parquet_path):
        print("Validation metadata not found. Skipping detailed correlation analysis.")
        return

    df_val = pd.read_parquet(val_parquet_path)

    # Ensure alignment: The loader preserves order, and df_val is the source.
    # However, we must ensure the lengths match.
    if len(df_val) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(df_val)}) != Predictions length ({len(rmse_per_sample)}). Skipping correlation."
        )
        return

    # 4. Correlate Error with Features
    features_to_check = []

    # Existing metadata columns
    if "signal_to_noise" in df_val.columns:
        features_to_check.append("signal_to_noise")
    if "SN_filter" in df_val.columns:
        features_to_check.append("SN_filter")

    # Derived sequence features
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))
    features_to_check.extend(["len_A", "len_G", "len_C", "len_U"])

    print(f"{'Feature':<20} | {'Correlation with Error':<25} | {'P-Value':<10}")
    print("-" * 60)

    for feat in features_to_check:
        if feat in df_val.columns:
            # Handle potential NaNs in metadata
            valid_mask = ~df_val[feat].isna()
            if valid_mask.sum() > 1:
                corr, p_val = pearsonr(
                    df_val.loc[valid_mask, feat], rmse_per_sample[valid_mask]
                )
                print(f"{feat:<20} | {corr:<25.4f} | {p_val:<10.4f}")


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating Submission...")
    model.eval()
    all_preds = []

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)

            preds = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 3)

    # Load Test Metadata for IDs
    test_parquet_path = os.path.join(Config.METADATA_DIR, "test.parquet")
    df_test = pd.read_parquet(test_parquet_path)
    test_ids = df_test["id"].values

    submission_data = []

    # Config.TARGET_COLS are ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"] (indices 0, 1, 2)
    # Submission requires: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(test_ids):
        sample_pred = test_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Predicted values
            val_reactivity = sample_pred[seqpos, 0]
            val_deg_Mg_pH10 = sample_pred[seqpos, 1]
            val_deg_Mg_50C = sample_pred[seqpos, 2]

            # Unpredicted values (fill with 0)
            val_deg_pH10 = 0.0
            val_deg_50C = 0.0

            submission_data.append(
                [
                    row_id,
                    val_reactivity,
                    val_deg_Mg_pH10,
                    val_deg_pH10,
                    val_deg_Mg_50C,
                    val_deg_50C,
                ]
            )

    submission_df = pd.DataFrame(
        submission_data, columns=["id_seqpos"] + Config.ALL_SUBMISSION_COLS
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNA_ResBiLSTM(Config).to(device)

    # 4. Training Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    criterion = nn.MSELoss()

    # 5. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.EPOCHS,  # Run full schedule as per idea
        save_path=best_model_path,
    )

    # 6. Final Validation
    print("Evaluating on Validation Set...")
    val_metric = eval_fn(model, val_loader, device)

    # STRICT OUTPUT FORMAT REQUIRED
    print(f"Final Validation Metric: {val_metric}")

    # 7. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 8. Submission
    threshold = 0.6199890971183777
    if val_metric < threshold:
        print(
            f"Validation metric {val_metric:.6f} is better than threshold {threshold:.6f}. Generating submission."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {val_metric:.6f} did not meet threshold {threshold:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
