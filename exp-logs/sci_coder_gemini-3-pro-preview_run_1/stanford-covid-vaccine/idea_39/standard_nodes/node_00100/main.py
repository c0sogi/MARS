import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.train import train, generate_submission
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import seed_everything, calculate_mcrmse


def run_failure_analysis(val_preds, val_targets, val_ids, config):
    """
    Performs failure analysis by correlating sample-wise errors with metadata features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Calculate Sample-wise Error (Mean of RMSEs per target for each sample)
    # val_preds: (N_samples, 68, 3)
    # val_targets: (N_samples, 68, 3)

    # Squared Error: (N, 68, 3)
    se = (val_preds - val_targets) ** 2

    # MSE per sample per target: (N, 3) -> Mean over seq dim (68)
    mse_per_sample_target = torch.mean(se, dim=1)

    # RMSE per sample per target: (N, 3)
    rmse_per_sample_target = torch.sqrt(mse_per_sample_target)

    # Mean RMSE per sample (aggregating targets): (N,)
    sample_errors = torch.mean(rmse_per_sample_target, dim=1).numpy()

    # Create a DataFrame for errors
    error_df = pd.DataFrame({"id": val_ids, "error": sample_errors})

    # 2. Load Metadata
    # We need to merge with val.parquet to get features like signal_to_noise
    val_meta_df = pd.read_parquet(config.val_file)

    # Merge
    analysis_df = pd.merge(error_df, val_meta_df, on="id", how="left")

    # 3. Define Features for Correlation
    # Extract sequence features
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    print(f"Correlations with Model Error (Sample-wise MCRMSE):")
    print("-" * 50)
    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs just in case
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat:<20} | {corr:+.4f}       | {p_val:.4e}")
            else:
                print(f"{feat:<20} | N/A (Not enough data)")
        else:
            print(f"{feat:<20} | Not found in metadata")
    print("-" * 50)


def main():
    # 1. Configuration
    config = Config()

    # Adjust config for a robust baseline within time limits
    # The default is 20 epochs, which is fast enough for this dataset size (~1700 samples)
    # on the provided hardware (A100). We will stick to the defaults in Config
    # as they are tuned for the "Wide-Stream" architecture.

    # Ensure reproducibility
    seed_everything(config.seed)

    print(f"Working Directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # 2. Train the Model
    # The train function handles training loop, validation monitoring, and saving best model.
    best_model_path = train(config)

    # 3. Final Validation Assessment
    print("\nRunning Final Validation Assessment...")

    # Load Validation Data
    _, val_loader, _ = get_dataloaders(config=config)

    # Load Best Model
    model = RNAModel(config).to(config.device)
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            seqs = batch["seq"].to(config.device)
            loops = batch["loop"].to(config.device)
            dists = batch["dist"].to(config.device)
            targets = batch["targets"].to(config.device)
            ids = batch["id"]

            preds = model(seqs, loops, dists)

            # Slice to scored length (68)
            preds_scored = preds[:, : config.pred_len, :]
            targets_scored = targets[:, : config.pred_len, :]

            val_preds_list.append(preds_scored.cpu())
            val_targets_list.append(targets_scored.cpu())
            val_ids_list.extend(ids)

    val_preds_all = torch.cat(val_preds_list, dim=0)
    val_targets_all = torch.cat(val_targets_list, dim=0)

    # Compute Metric
    final_mcrmse = calculate_mcrmse(val_preds_all, val_targets_all).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcrmse}")

    # 4. Failure Analysis
    run_failure_analysis(val_preds_all, val_targets_all, val_ids_list, config)

    # 5. Submission Generation
    # Threshold from instructions: 0.6199890971183777
    THRESHOLD = 0.6199890971183777

    if final_mcrmse < THRESHOLD:
        print(
            f"\nMetric ({final_mcrmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(config)
    else:
        print(
            f"\nMetric ({final_mcrmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
