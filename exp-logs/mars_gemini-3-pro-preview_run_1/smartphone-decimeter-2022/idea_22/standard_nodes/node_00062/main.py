import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import train_model, generate_submission
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import StratifiedResUNet1D
from library.utils import load_checkpoint, cartesian_to_wgs84, haversine_distance


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline and Path Alignment
    Config.setup()

    # Point cache to ./working where the parquet files likely exist based on file listing
    # This avoids reprocessing raw CSVs if possible
    Config.CACHE_DIR = Config.WORKING_DIR

    # Fast training settings
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32
    Config.DEBUG = False  # Use full dataset for valid metrics

    print(f"Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Cache Dir: {Config.CACHE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Training Model...")
    # train_model handles dataset loading, training loop, and saving best model
    best_model_path = train_model(load_cached_data=True)
    print(f"Training complete. Best model: {best_model_path}")

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Validating...")

    # Load Validation Data
    val_dataset = GNSSSequenceDataset(
        split="val", load_cached_data=True, debug=Config.DEBUG
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=4,
    )

    # Load Model
    model = StratifiedResUNet1D().to(device)
    load_checkpoint(model, None, best_model_path, device=device)
    model.eval()

    results = []

    # Features for Failure Analysis
    # We'll collect: Error, Cn0_mean (Channel 0), SignalCount (Channel -4)
    fa_data = {"error": [], "cn0_mean": [], "signal_count": []}

    with torch.no_grad():
        for features, targets, mask, metadata_list in val_loader:
            features = features.to(device)
            # features shape: (B, C, L)

            # Forward pass
            outputs = model(features)
            preds = outputs["main"].cpu().numpy()  # (B, 2, L)

            # Move inputs to CPU for analysis
            features_cpu = features.cpu().numpy()
            targets_cpu = targets.cpu().numpy()  # (B, 2, L)

            # Iterate over batch
            for i in range(len(metadata_list)):
                meta = metadata_list[i]
                seq_len = meta["timestamps"].shape[0]

                # Extract valid sequence
                # Predictions: (2, L) -> (L, 2)
                local_preds = preds[i, :, :seq_len].transpose(1, 0)
                d_east_pred = local_preds[:, 0]
                d_north_pred = local_preds[:, 1]

                # Targets (Ground Truth Offsets): (2, L) -> (L, 2)
                local_targets = targets_cpu[i, :, :seq_len].transpose(1, 0)
                d_east_gt = local_targets[:, 0]
                d_north_gt = local_targets[:, 1]

                # Baseline WLS
                wls_pos = meta["wls_pos"].numpy()
                wls_lat = wls_pos[:, 0]
                wls_lon = wls_pos[:, 1]

                # Convert both Pred and GT offsets to Lat/Lon
                # Note: We reconstruct GT from WLS + Target Delta to ensure consistency
                pred_lat, pred_lon = cartesian_to_wgs84(
                    d_east_pred, d_north_pred, wls_lat, wls_lon
                )
                gt_lat, gt_lon = cartesian_to_wgs84(
                    d_east_gt, d_north_gt, wls_lat, wls_lon
                )

                # Calculate Haversine Distance
                dists = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

                # Store results for Metric
                phone_id = f"{meta['drive_id']}_{meta['phone_name']}"
                for d in dists:
                    results.append({"phone": phone_id, "error": d})

                # Store data for Failure Analysis
                # Feature 0: S1_Cn0DbHz_mean
                # Feature -4: SignalCount (Assuming Config.GLOBAL_FEATURES order)
                # features_cpu shape: (B, C, L)
                cn0 = features_cpu[i, 0, :seq_len]
                sig_count = features_cpu[i, -4, :seq_len]

                fa_data["error"].extend(dists)
                fa_data["cn0_mean"].extend(cn0)
                fa_data["signal_count"].extend(sig_count)

    # Compute Final Metric
    df_res = pd.DataFrame(results)

    # "The 50th and 95th percentile errors are then averaged for each phone."
    # "Lastly, the mean of these averaged values is calculated across all phones"
    phone_stats = df_res.groupby("phone")["error"].quantile([0.50, 0.95]).unstack()
    phone_stats["avg_50_95"] = (phone_stats[0.50] + phone_stats[0.95]) / 2
    final_metric = phone_stats["avg_50_95"].mean()

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Failure Analysis...")
    df_fa = pd.DataFrame(fa_data)

    # Correlation
    corr_cn0 = df_fa["error"].corr(df_fa["cn0_mean"])
    corr_sig = df_fa["error"].corr(df_fa["signal_count"])

    print(f"Correlation between Error and Signal Strength (Cn0): {corr_cn0:.4f}")
    print(f"Correlation between Error and Signal Count: {corr_sig:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Submission Check...")
    THRESHOLD = 3.802240262877392

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric:.4f} is better than threshold {THRESHOLD:.4f}.")
        print("Generating submission file...")
        generate_submission(best_model_path, load_cached_data=True)
    else:
        print(f"Metric {final_metric:.4f} did not meet threshold {THRESHOLD:.4f}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
