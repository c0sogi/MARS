import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import ZoneoutWideResBiGRU
from library.engine import train_model, validate, generate_submission


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("Performing Failure Analysis...")
    model.eval()

    # 1. Collect Predictions and Targets
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            batch_ids = batch["id"]

            outputs = model(sequence, loop_type, pair_dist)

            # Slice to scored length
            scored_preds = outputs[:, : Config.SCORED_LEN, :]
            scored_targets = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(scored_preds.cpu().numpy())
            all_targets.append(scored_targets.cpu().numpy())
            all_ids.extend(batch_ids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # 2. Calculate RMSE per sample
    # Shape: (N_samples, Seq_Len, N_targets)
    # MSE per sample: mean over seq_len and targets
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Load Metadata
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment by ID
    # Create a mapping from ID to RMSE
    rmse_map = dict(zip(all_ids, rmse_per_sample))

    # Add RMSE to dataframe
    df_val["sample_rmse"] = df_val["id"].map(rmse_map)

    # Filter out any missing rows (should not happen if data is consistent)
    df_val = df_val.dropna(subset=["sample_rmse"])

    # 4. Feature Engineering for Correlation
    # Signal to Noise
    if "signal_to_noise" not in df_val.columns:
        # Fallback if column missing, though it should be there
        print("Warning: 'signal_to_noise' column missing in metadata.")
        df_val["signal_to_noise"] = 0.0

    # Nucleotide Counts
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # 5. Calculate Correlations
    features = ["signal_to_noise", "len_A", "len_G", "len_C", "len_U"]
    print("\nCorrelation between Sample RMSE and Features:")
    for feat in features:
        if feat in df_val.columns:
            # Check if feature is constant
            if df_val[feat].std() == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(df_val[feat], df_val["sample_rmse"])
            print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Adjust Config for Fast Baseline
    # 15 epochs is sufficient for convergence on this small dataset
    Config.EPOCHS = 15

    print(f"Initializing run on {device}...")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = ZoneoutWideResBiGRU().to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training
    train_model(model, train_loader, val_loader, optimizer, device, scheduler)

    # 6. Final Validation
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    threshold = 0.6199890971183777
    if val_score < threshold:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation score {val_score} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
