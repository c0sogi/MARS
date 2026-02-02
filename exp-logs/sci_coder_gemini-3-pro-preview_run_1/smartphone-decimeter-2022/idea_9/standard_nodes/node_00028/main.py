import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Import from library
from library.config import Config

# Monkey-patch Config for fast baseline execution
Config.NUM_EPOCHS = 5
Config.BATCH_SIZE = 32

from library.train import train_model
from library.dataset import get_dataloaders
from library.model import HybridResUNetGRU
from library.utils import cartesian_to_wgs84, haversine_distance
from library.inference import generate_submission


def run():
    # --- 1. Training Phase ---
    print("=" * 40)
    print("STARTING TRAINING PHASE")
    print("=" * 40)
    # This will load data (computing features if needed), train the model, and save 'best_model.pth'
    train_model()

    # --- 2. Evaluation & Failure Analysis Phase ---
    print("\n" + "=" * 40)
    print("STARTING EVALUATION & ANALYSIS PHASE")
    print("=" * 40)

    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We only need val_loader here
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the best model trained in step 1
    model = HybridResUNetGRU().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Containers for metrics and analysis
    phone_errors = {}  # phone_name -> list of distance errors
    feature_stats = {"error": [], "cn0_mean": [], "elev_mean": [], "sat_count": []}

    print("Evaluating on Validation Set...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            baselines = batch["baseline"]
            timestamps = batch["timestamps"]
            phone_names = batch["phone_name"]

            # Forward pass
            outputs = model(features)  # (B, 2, T)

            batch_size = features.size(0)
            for i in range(batch_size):
                t_len = len(timestamps[i])
                p_name = phone_names[i]

                # Get valid mask for this sequence
                # mask is (B, T)
                valid_mask = mask[i, :t_len].cpu().numpy().astype(bool)

                if valid_mask.sum() == 0:
                    continue

                # Extract Predictions (Cartesian Offsets)
                pred_offsets = outputs[i, :, :t_len].cpu().numpy().T

                # Extract Baseline
                base_pos = baselines[i]

                # Reconstruct WGS84 Predictions
                pred_lat, pred_lon = cartesian_to_wgs84(
                    pred_offsets[:, 0],
                    pred_offsets[:, 1],
                    base_pos[:, 0],
                    base_pos[:, 1],
                )

                # Extract Targets (Cartesian Offsets) and Reconstruct Ground Truth
                # Note: We reconstruct GT from offsets to ensure alignment with the mask
                target_offsets = targets[i, :, :t_len].cpu().numpy().T
                gt_lat, gt_lon = cartesian_to_wgs84(
                    target_offsets[:, 0],
                    target_offsets[:, 1],
                    base_pos[:, 0],
                    base_pos[:, 1],
                )

                # Filter by valid mask
                p_lat_valid = pred_lat[valid_mask]
                p_lon_valid = pred_lon[valid_mask]
                gt_lat_valid = gt_lat[valid_mask]
                gt_lon_valid = gt_lon[valid_mask]

                # Calculate Haversine Distance
                dists = haversine_distance(
                    gt_lat_valid, gt_lon_valid, p_lat_valid, p_lon_valid
                )

                # Store for Metric Calculation
                if p_name not in phone_errors:
                    phone_errors[p_name] = []
                phone_errors[p_name].extend(dists)

                # Extract Features for Failure Analysis
                # Features shape: (Channels, Time)
                # Indices based on data_processing.py:
                # 22: CN0 Mean, 25: Elev Mean, 26: Sat Count
                feats_np = features[i, :, :t_len].cpu().numpy()

                cn0_means = feats_np[22, valid_mask]
                elev_means = feats_np[25, valid_mask]
                sat_counts = feats_np[26, valid_mask]

                feature_stats["error"].extend(dists)
                feature_stats["cn0_mean"].extend(cn0_means)
                feature_stats["elev_mean"].extend(elev_means)
                feature_stats["sat_count"].extend(sat_counts)

    # Compute Final Metric
    # Mean of (50th + 95th) / 2 averaged across phones
    phone_scores = []
    for p_name, errors in phone_errors.items():
        if len(errors) == 0:
            continue
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score = (p50 + p95) / 2.0
        phone_scores.append(score)

    if phone_scores:
        final_metric = np.mean(phone_scores)
    else:
        final_metric = 0.0

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    print("\nFailure Analysis (Correlation between Error and Features):")
    if len(feature_stats["error"]) > 1:
        for feat_name in ["cn0_mean", "elev_mean", "sat_count"]:
            if len(feature_stats[feat_name]) == len(feature_stats["error"]):
                corr, _ = pearsonr(feature_stats["error"], feature_stats[feat_name])
                print(f"  Correlation with {feat_name}: {corr:.4f}")
            else:
                print(f"  Skipping {feat_name} due to length mismatch.")
    else:
        print("  Not enough data for correlation analysis.")

    # --- 3. Submission Phase ---
    THRESHOLD = 3.802240262877392
    if final_metric < THRESHOLD:
        print("\n" + "=" * 40)
        print(f"METRIC PASSED ({final_metric} < {THRESHOLD})")
        print("GENERATING SUBMISSION")
        print("=" * 40)
        generate_submission(load_cached_data=True)
    else:
        print("\n" + "=" * 40)
        print(f"METRIC FAILED ({final_metric} >= {THRESHOLD})")
        print("SKIPPING SUBMISSION")
        print("=" * 40)


if __name__ == "__main__":
    run()
