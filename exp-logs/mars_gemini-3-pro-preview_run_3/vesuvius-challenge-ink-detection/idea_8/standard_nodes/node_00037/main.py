import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from pathlib import Path

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    fbeta_score,
    load_inklabels,
    load_volume,
    load_mask,
)
from library.train import train_model
from library.inference import predict_full_mask, generate_submission
from library.model import DilatedFCN


def main():
    # 1. Setup
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Ensure the submission directory exists as per prompt requirement
    submission_dir = Path("./submission")
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch Config to save submission to the correct path required by the prompt
    Config.SUBMISSION_PATH = submission_dir / "submission.csv"

    # 2. Train
    # Run the training loop. This will save 'best_model.pth' in the working directory.
    # We use load_cached_data=True to speed up data loading if cache exists.
    train_model(load_cached_data=True)

    # 3. Validation & Metric Calculation
    # We need to manually calculate the metric to print it in the exact format required.
    device = torch.device(Config.DEVICE)
    model = DilatedFCN().to(device)

    if not Config.BEST_MODEL_PATH.exists():
        print("Error: Best model not found after training.")
        return

    # Load the best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Run inference on the validation set
    # predict_full_mask handles tiling and aggregation
    preds_map, masks_map = predict_full_mask(model, split="val", load_cached_data=True)

    # Load Validation Metadata
    if not Config.VAL_METADATA_PATH.exists():
        print("Validation metadata missing.")
        return

    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Containers for metric calculation and failure analysis
    all_preds = []
    all_labels = []
    all_intensities = []

    # Iterate over validation fragments to collect pixel data
    for _, row in df_val.iterrows():
        fid = str(row["fragment_id"])
        if fid not in preds_map:
            continue

        # Load Ground Truth Label
        label = load_inklabels(fid, "val", df_val, load_cached_data=True)
        if label is None:
            continue

        # Load Volume for Failure Analysis (Feature: Mean Pixel Intensity)
        vol = load_volume(fid, "val", df_val, load_cached_data=True)
        # Calculate mean intensity across Z-depth (H, W)
        mean_intensity = np.mean(vol, axis=0)

        mask = masks_map[fid]
        valid_indices = mask > 0

        # Flatten and filter by valid mask
        p_flat = preds_map[fid][valid_indices]
        l_flat = label[valid_indices]
        i_flat = mean_intensity[valid_indices]

        all_preds.append(p_flat)
        all_labels.append(l_flat)
        all_intensities.append(i_flat)

    if not all_preds:
        print("No validation data found for evaluation.")
        return

    # Concatenate all fragments
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_intensities = np.concatenate(all_intensities)

    # Optimize Threshold to find the best F0.5 score
    best_score = -1.0
    best_th = 0.5
    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )

    for th in thresholds:
        score = fbeta_score(all_preds, all_labels, beta=0.5, threshold=th)
        if score > best_score:
            best_score = score
            best_th = th

    # Print the required metric
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    # Calculate correlation between prediction error and pixel intensity
    # Error is defined as absolute difference between probability and binary label
    errors = np.abs(all_preds - all_labels)

    if np.std(errors) > 0 and np.std(all_intensities) > 0:
        corr_intensity, _ = pearsonr(errors, all_intensities)
        print(f"Correlation (Error vs Intensity): {corr_intensity}")
    else:
        print("Correlation (Error vs Intensity): 0.0")

    # 5. Submission
    # Generate submission only if the score beats the baseline
    target_metric = 0.4064630960392697
    if best_score > target_metric:
        # generate_submission uses Config.SUBMISSION_PATH which we updated earlier
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation score {best_score} does not exceed {target_metric}. Submission skipped."
        )


if __name__ == "__main__":
    main()
