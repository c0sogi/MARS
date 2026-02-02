import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.train import train_model, generate_submission
from library.data import get_dataset, RNADataset
from library.model import HighCapacityBiGRU
from torch.utils.data import DataLoader


def run_failure_analysis(model_path, device):
    """
    Performs failure analysis on the validation set by correlating
    model error with input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Load Validation Metadata (for features like signal_to_noise)
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping failure analysis.")
        return

    val_df = pd.read_parquet(val_meta_path)

    # 2. Load Processed Validation Data (for inference)
    val_inputs, val_targets = get_dataset("val", load_cached_data=True)

    # 3. Load Model
    model = HighCapacityBiGRU(Config).to(device)
    if not os.path.exists(model_path):
        print("Model file not found for failure analysis.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Inference
    val_dataset = RNADataset(val_inputs, val_targets)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            item, y = batch
            x = item["X"].to(device)
            adj = item["adj"].to(device)
            mask = item["mask"].to(device)

            # Forward pass
            pred = model(x, adj, mask)

            # Slice to scored length (68)
            pred = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()  # Shape: (N, 68, 5)
    targets = torch.cat(all_targets, dim=0).numpy()  # Shape: (N, 68, 5)

    # 5. Calculate Per-Sample Error
    # We focus on the scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    preds_scored = preds[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Compute RMSE per column for each sample, then average over columns
    # diff_sq: (N, 68, 3)
    diff_sq = (preds_scored - targets_scored) ** 2
    # Mean over sequence length (axis 1) -> MSE per column per sample: (N, 3)
    mse_per_col = np.mean(diff_sq, axis=1)
    # RMSE per column per sample: (N, 3)
    rmse_per_col = np.sqrt(mse_per_col)
    # Average over the 3 scored columns -> Single scalar error per sample: (N,)
    sample_errors = np.mean(rmse_per_col, axis=1)

    val_df["model_error"] = sample_errors

    # 6. Feature Engineering for Correlation Analysis
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_paired"] = val_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    # 7. Compute and Print Correlations
    analysis_cols = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_paired",
    ]
    print("Correlation between Model Error and Features:")

    for col in analysis_cols:
        if col in val_df.columns:
            # Drop NaNs to ensure pearsonr works
            valid_data = val_df[[col, "model_error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[col], valid_data["model_error"])
                print(f"  {col}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train Model
    # We limit epochs to 15 for a fast baseline execution.
    # The full config suggests 240, but 15 is sufficient for initial validation within time limits.
    print("Starting Training...")
    best_val_score = train_model(epochs=15, patience=5)

    # 3. Print Final Metric
    # Must print exactly "Final Validation Metric: <value>"
    print(f"Final Validation Metric: {best_val_score}")

    # 4. Perform Failure Analysis
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    run_failure_analysis(best_model_path, device)

    # 5. Generate Submission
    # Only submit if the validation score is better (lower) than the threshold.
    THRESHOLD = 0.5884495377540588

    if best_val_score < THRESHOLD:
        print(
            f"Validation score ({best_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Validation score ({best_val_score}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
