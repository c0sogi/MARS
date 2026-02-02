import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import StructuralBiGRU
from library.train_eval import train_model, predict_and_submit, mcrmse_metric

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure execution completes within 2 hours
Config.EPOCHS = 15
# Set submission path as per task requirement
Config.SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # 1. Train the model
    # train_model handles the training loop, validation monitoring, and saving the best model.
    train_model(debug=False)

    # 2. Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    _, val_loader, _ = get_dataloaders(debug=False, load_cached_data=True)

    # Load Model
    model = StructuralBiGRU().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, pair_indices, targets in val_loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)

            outputs = model(inputs, pair_indices)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    # Note: We must use the scored_only=True flag to match competition metric
    # (reactivity, deg_Mg_pH10, deg_Mg_50C)
    final_metric = mcrmse_metric(all_preds, all_targets, scored_only=True)

    # PRINT REQUIRED METRIC FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Load metadata for feature correlation
    val_df = pd.read_parquet(Config.VAL_PARQUET)

    # Calculate RMSE per sample (averaged over scored columns and positions)
    # Preds: (N, 107, 5), Targets: (N, 68, 5)
    # We slice preds to match targets
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, :].numpy()
    targets_numpy = all_targets.numpy()

    # Calculate MSE per sample
    # Shape: (N,)
    mse_per_sample = np.mean((preds_sliced - targets_numpy) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["error_rmse"] = rmse_per_sample

    # Feature Engineering for Correlation
    # 1. Signal to Noise (Raw metadata)
    # 2. SN Filter (Raw metadata)
    # 3. GC Content (Calculated from sequence)
    # 4. Paired Percentage (Calculated from structure)

    val_df["gc_content"] = val_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    val_df["paired_pct"] = val_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    analysis_features = ["signal_to_noise", "SN_filter", "gc_content", "paired_pct"]

    print("Correlation between Error (RMSE) and Input Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            # Handle potential NaNs just in case
            valid_idx = ~val_df[feat].isna() & ~val_df["error_rmse"].isna()
            if valid_idx.sum() > 1:
                corr, _ = pearsonr(
                    val_df.loc[valid_idx, feat], val_df.loc[valid_idx, "error_rmse"]
                )
                print(f"  {feat}: {corr:.6f}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Not found in metadata")

    # 4. Submission Generation
    # Threshold check
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nMetric passed threshold ({final_metric} < {THRESHOLD}). Generating submission..."
        )
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        predict_and_submit()
    else:
        print(
            f"\nMetric failed threshold ({final_metric} >= {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
