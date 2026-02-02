import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, global_mcrmse_metric
from library.data import get_loaders
from library.engine import train_engine, predict_submission
from library.model import RNAModel


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Use cached data if available for speed
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Training
    # Run for 20 epochs for a fast baseline
    print("Starting training...")
    best_val_score = train_engine(train_loader, val_loader, epochs=20)

    # 4. Validation & Metric Calculation
    print("Loading best model for validation and analysis...")
    model = RNAModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found.")
        return

    model.eval()

    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["sequence"].to(device)
            pair_indices = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_indices)

            val_preds.append(preds.cpu())
            val_targets.append(targets.cpu())
            val_ids.extend(ids)

    # Compute Global Metric
    final_metric = global_mcrmse_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n==== Failure Analysis ====")

    # Load metadata to get features
    val_df = pd.read_parquet(Config.VAL_FILE)
    meta_map = val_df.set_index("id").to_dict("index")

    # Concatenate predictions and targets
    preds_cat = torch.cat(val_preds, dim=0).numpy()  # (N, 107, 5)
    targets_cat = torch.cat(val_targets, dim=0).numpy()  # (N, 68, 5)

    # Identify scored columns indices
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices: 0, 1, 3
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    sample_errors = []
    feat_sn = []
    feat_sn_filter = []
    feat_gc = []
    feat_paired = []

    for i, sample_id in enumerate(val_ids):
        # Slice to scored length and columns
        p = preds_cat[i, : Config.SEQ_SCORED, scored_indices]
        t = targets_cat[i, :, scored_indices]

        # Calculate MCRMSE for this sample
        mse = np.mean((p - t) ** 2, axis=0)
        rmse = np.sqrt(mse)
        error = np.mean(rmse)
        sample_errors.append(error)

        # Get Metadata Features
        row = meta_map.get(sample_id)
        if row:
            # Signal to Noise
            feat_sn.append(row.get("signal_to_noise", 0))
            # SN Filter
            feat_sn_filter.append(row.get("SN_filter", 0))

            # GC Content
            seq = row.get("sequence", "")
            if len(seq) > 0:
                gc = (seq.count("G") + seq.count("C")) / len(seq)
            else:
                gc = 0
            feat_gc.append(gc)

            # Paired Percentage
            struct = row.get("structure", "")
            if len(struct) > 0:
                # '.' is unpaired
                paired = 1.0 - (struct.count(".") / len(struct))
            else:
                paired = 0
            feat_paired.append(paired)
        else:
            # Fallback if id not found (should not happen)
            feat_sn.append(0)
            feat_sn_filter.append(0)
            feat_gc.append(0)
            feat_paired.append(0)

    # Calculate Correlations
    corr_sn, _ = pearsonr(sample_errors, feat_sn)
    corr_filter, _ = pearsonr(sample_errors, feat_sn_filter)
    corr_gc, _ = pearsonr(sample_errors, feat_gc)
    corr_paired, _ = pearsonr(sample_errors, feat_paired)

    print("Correlation between Error and Input Features:")
    print(f"  Signal_to_Noise: {corr_sn:.4f}")
    print(f"  SN_Filter:       {corr_filter:.4f}")
    print(f"  GC_Content:      {corr_gc:.4f}")
    print(f"  Pct_Paired:      {corr_paired:.4f}")

    # 6. Submission
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        predict_submission(test_loader, output_file=submission_path)
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
