import sys
import os
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import train_model, generate_submission, validate
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import set_seed


def run_pipeline():
    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    # Modify Config for a fast baseline execution
    # Config.EPOCHS = 15 # Removed override to use Config default (25)
    Config.PATIENCE = 5

    # Ensure reproducible results
    set_seed(Config.SEED)

    print("Starting pipeline...")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # train_model handles data loading, training loop, and saving the best model
    train_model(Config)

    # --------------------------------------------------------------------------
    # 3. Final Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Performing final validation...")
    device = torch.device(Config.DEVICE)

    # Load best model
    model = RNAModel(Config).to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get loaders (train, val, test)
    _, val_loader, _ = get_dataloaders(Config, load_cached_data=True)

    # Calculate metric using the library function which implements MCRMSE
    final_metric = validate(model, val_loader, device, Config)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming failure analysis...")
    analyze_failures(model, val_loader, device)

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    # Threshold check as per requirements
    THRESHOLD = 0.7462618350982666

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(Config)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


def analyze_failures(model, val_loader, device):
    """
    Computes per-sample error and correlates with metadata features.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            distance = batch["distance"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(sequence, loop, distance)

            # Slice to scored length (first 68 positions)
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Calculate RMSE per sample (averaging over the 5 targets and 68 positions)
    # MSE per sample
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    # RMSE per sample
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create DataFrame for analysis
    df_error = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Load metadata to get features
    # Note: Metadata generation script ensures these files exist
    df_val_meta = pd.read_parquet(Config.VAL_METADATA)

    # Merge error data with metadata
    df_analysis = pd.merge(df_error, df_val_meta, on="id", how="left")

    # Compute derived features for correlation
    df_analysis["len_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
    df_analysis["len_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    df_analysis["len_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))
    df_analysis["len_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))

    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    print("Correlation between Model Error (RMSE) and Features:")
    for feat in features:
        if feat in df_analysis.columns:
            # Drop NaNs just in case, though data should be clean
            valid_data = df_analysis[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")


if __name__ == "__main__":
    run_pipeline()
