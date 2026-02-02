import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.engine import train_fn, validate_fn, generate_submission
from library.dataset import get_dataset
from library.model import RNAModel


def failure_analysis(model, val_loader, val_df):
    """
    Performs failure analysis by correlating prediction errors with metadata features.
    """
    # print("\n==== Failure Analysis ====")
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # Get predictions
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            y = batch["y"].to(Config.DEVICE)
            ids = batch["id"]

            pred = model(x, adj)
            # Slice to scored length
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_ids.extend(ids)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE per sample (averaged over the 3 scored columns: 0, 1, 3)
    # columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    scored_indices = [0, 1, 3]

    # Error per sample: sqrt(mean((y-y_hat)^2)) averaged over columns
    # Shape: (N, 68, 3)
    diff_sq = (preds[:, :, scored_indices] - targets[:, :, scored_indices]) ** 2
    # Mean over sequence length (axis 1) -> (N, 3)
    mse_per_col = np.mean(diff_sq, axis=1)
    rmse_per_col = np.sqrt(mse_per_col)
    # Mean over columns -> (N,)
    mcrmse_per_sample = np.mean(rmse_per_col, axis=1)

    # Create DataFrame for analysis
    error_df = pd.DataFrame({"id": all_ids, "error": mcrmse_per_sample})

    # Merge with metadata
    # val_df has 'id', 'signal_to_noise', 'SN_filter', 'sequence', 'structure'
    merged_df = pd.merge(error_df, val_df, on="id", how="left")

    # Feature Engineering for correlation
    merged_df["seq_len"] = merged_df["sequence"].apply(len)
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda s: s.count("A") / len(s))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda s: s.count("G") / len(s))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda s: s.count("C") / len(s))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda s: s.count("U") / len(s))
    merged_df["pct_paired"] = merged_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    # Correlations
    features = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_paired",
    ]
    print("Correlation between Error and Features:")
    for feat in features:
        if feat in merged_df.columns:
            # Drop NaNs if any
            valid = merged_df[[feat, "error"]].dropna()
            if len(valid) > 1:
                corr, _ = pearsonr(valid[feat], valid["error"])
                print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Configuration Override
    # Limit epochs for speed as per requirements
    Config.EPOCHS = 15
    # Override submission path to match requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs("./submission", exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 2. Training
    # train_fn() handles the full training loop and saves the best model
    train_fn()

    # 3. Validation & Metrics
    # Load validation data
    val_ds = get_dataset("val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load best model
    model = RNAModel().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    else:
        print("Error: Model file not found.")
        return

    # Calculate final metric
    _, val_mcrmse_scored = validate_fn(model, val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mcrmse_scored}")

    # 4. Failure Analysis
    val_df = pd.read_parquet(Config.VAL_META)
    failure_analysis(model, val_loader, val_df)

    # 5. Submission
    # Threshold: 0.5978901386
    threshold = 0.5978901386

    if val_mcrmse_scored < threshold:
        # print(f"Validation metric {val_mcrmse_scored} < {threshold}. Generating submission...")
        generate_submission()
    else:
        print(
            f"Validation metric {val_mcrmse_scored} >= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
