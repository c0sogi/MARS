"""
Orchestration script for RNA Degradation Prediction.
Implements training, validation, failure analysis, and submission.
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import GEHN
from library.train import Trainer
from library.predict import generate_submission


def main():
    # 1. Setup
    seed_everything(42)
    config = Config()

    # Configure for fast baseline execution
    # The dataset is small (1728 train samples), so we use the full dataset
    # but limit epochs to ensure execution finishes quickly (approx 5-10 mins).
    config.debug = False
    config.epochs = 15

    print(f"Configuration: Device={config.device}, Epochs={config.epochs}")

    # 2. Data Loading
    print("\n--- Data Loading ---")
    # load_cached_data=True will use existing .npy files in working/idea_2 if available
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Training
    print("\n--- Training ---")
    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)

    # 4. Final Validation & Metric Calculation
    print("\n--- Final Validation ---")

    # Load best model
    model = GEHN(config).to(config.device)
    if not os.path.exists(config.best_model_path):
        print("Error: Best model not found.")
        return

    model.load_state_dict(
        torch.load(config.best_model_path, map_location=config.device)
    )
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(config.device)
            adj = batch["adj"].to(config.device)
            targets = batch["targets"].to(config.device)
            ids = batch["id"]

            preds = model(inputs, adj)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 107, 5)

    # Calculate MCRMSE on Scored Columns
    # Scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Indices in target_cols: 0, 1, 3
    scored_indices = [0, 1, 3]

    # Slice to pred_len (68)
    preds_scored = all_preds[:, : config.pred_len, :]
    targets_scored = all_targets[:, : config.pred_len, :]

    # Calculate RMSE per column
    # MSE: Mean over (Batch, Seq)
    mse_per_col = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))
    rmse_per_col = torch.sqrt(mse_per_col)

    # Average RMSE over scored columns
    final_metric = torch.mean(rmse_per_col[scored_indices]).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate RMSE per sample (averaged over scored columns and positions)
    # Shape: (N, 68, 3)
    preds_s = preds_scored[:, :, scored_indices]
    targs_s = targets_scored[:, :, scored_indices]

    # Mean Squared Error per sample
    sample_mse = torch.mean((preds_s - targs_s) ** 2, dim=(1, 2))
    sample_rmse = sample_mse.sqrt().numpy()

    # Load metadata for features
    val_df = pd.read_csv(config.val_file)

    # Ensure alignment: val_loader is shuffle=False, so order should match CSV.
    # But to be safe, reindex based on IDs collected during inference.
    val_df = val_df.set_index("id").reindex(all_ids).reset_index()

    # Feature 1: Signal to Noise
    sn = val_df["signal_to_noise"].fillna(0).values

    # Feature 2: GC Content
    # Helper to calc GC
    def calc_gc(seq):
        if not isinstance(seq, str):
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)

    gc_content = val_df["sequence"].apply(calc_gc).values

    # Feature 3: Paired Base Ratio (Structure Density)
    def calc_paired(struc):
        if not isinstance(struc, str):
            return 0.0
        return (struc.count("(") + struc.count(")")) / len(struc)

    paired_ratio = val_df["structure"].apply(calc_paired).values

    # Calculate Correlations using Numpy
    corr_sn = np.corrcoef(sample_rmse, sn)[0, 1]
    corr_gc = np.corrcoef(sample_rmse, gc_content)[0, 1]
    corr_struc = np.corrcoef(sample_rmse, paired_ratio)[0, 1]

    print("Correlation between Model Error (RMSE) and Input Features:")
    print(f"  Signal-to-Noise: {corr_sn:.6f}")
    print(f"  GC Content: {corr_gc:.6f}")
    print(f"  Structure Density: {corr_struc:.6f}")

    # 6. Submission
    print("\n--- Submission Generation ---")
    threshold = 0.6795554161071777

    if final_metric < threshold:
        print(
            f"Metric {final_metric} meets threshold ({threshold}). Generating submission..."
        )

        # Generate to default path (./working/idea_2/submission.csv)
        generate_submission(config, load_cached_data=True)

        # Move to required path (./submission/submission.csv)
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        if os.path.exists(config.submission_path):
            shutil.move(config.submission_path, target_path)
            print(f"Submission saved to {target_path}")
        else:
            print(f"Error: Generated submission not found at {config.submission_path}")
    else:
        print(
            f"Metric {final_metric} does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
