import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import load_data
from library.model import (
    ScalarAggregatedBiGRU,
    generate_submission,
    set_seed,
    compute_mcrmse,
)
from library.trainer import Trainer


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating sample-wise error with metadata features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_ids = []

    # 1. Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["id"]

            outputs = model(sequences, loop_types, pair_dists)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())
            all_ids.extend(ids)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # 2. Compute Sample-wise RMSE
    # We compute RMSE per sample across the 3 targets and valid positions
    # Shape: (N, L, 3)
    diff_sq = (preds - targets) ** 2

    # Mask out invalid positions
    # masks: (N, L) -> (N, L, 3)
    mask_expanded = masks.unsqueeze(-1).expand_as(diff_sq)

    # Zero out invalid errors
    diff_sq = diff_sq * mask_expanded.float()

    # Sum errors per sample (dim 1 and 2)
    sum_sq_per_sample = diff_sq.sum(dim=(1, 2))

    # Count valid positions per sample (dim 1 and 2)
    count_per_sample = mask_expanded.sum(dim=(1, 2)) + 1e-8

    # MSE per sample
    mse_per_sample = sum_sq_per_sample / count_per_sample
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata for Correlation
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Ensure alignment by ID (though loader order should be preserved, safe to merge)
    df_errors = pd.DataFrame({"id": all_ids, "error_rmse": rmse_per_sample})

    # Merge with metadata
    df_analysis = pd.merge(df_val, df_errors, on="id", how="inner")

    # 4. Compute Correlations
    features_to_check = []

    # Signal to Noise
    if "signal_to_noise" in df_analysis.columns:
        features_to_check.append("signal_to_noise")

    # SN Filter
    if "SN_filter" in df_analysis.columns:
        features_to_check.append("SN_filter")

    # Sequence Composition (e.g., G content)
    df_analysis["G_content"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    features_to_check.append("G_content")

    # Ground Truth Statistics (Mean Reactivity)
    # df_analysis["mean_reactivity"] = df_analysis["reactivity"].apply(lambda x: np.mean(x))
    # features_to_check.append("mean_reactivity")

    print(f"Correlations with Model Error (RMSE):")
    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Drop NaNs if any
            valid_data = df_analysis[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error_rmse"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()

    # Adjust Config for Fast Baseline
    # 20 epochs is sufficient for this small dataset and fits within time limits
    Config.EPOCHS = 20

    print(f"Running Fast Baseline on Device: {Config.DEVICE}")

    # 2. Training
    trainer = Trainer(Config)
    # We use debug=False to train on the full dataset as it is small (1.7k samples)
    # and we need maximum performance to beat the threshold.
    best_score = trainer.fit(debug=False)

    print(f"Final Validation Metric: {best_score}")

    # 3. Load Best Model for Analysis & Inference
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model = ScalarAggregatedBiGRU(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # 4. Failure Analysis
    val_dataset = load_data("val", debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    run_failure_analysis(model, val_loader, Config.DEVICE)

    # 5. Submission Logic
    THRESHOLD = 0.6209375959946717

    if best_score < THRESHOLD:
        print(
            f"\nValidation score ({best_score:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        test_dataset = load_data("test", debug=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        generate_submission(model, test_loader, Config.DEVICE)
    else:
        print(
            f"\nValidation score ({best_score:.6f}) did NOT meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
