import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

# Import from provided library files
from library.config import Config
from library.dataset import SmartphoneLocationDataset, collate_fn
from library.model import HR1DResNet
from library.trainer import Trainer
from library.inference import run_inference
from library.utils import enu_to_ecef, ecef_to_wgs84, wgs84_to_ecef


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


def calculate_competition_metric(df_results):
    """
    Computes the competition metric:
    Mean of (50th percentile error + 95th percentile error) / 2 averaged across phones.
    """
    phones = df_results["phone_name"].unique()
    phone_scores = []

    for phone in phones:
        phone_data = df_results[df_results["phone_name"] == phone]

        # Calculate distances
        dists = haversine_distance(
            phone_data["LatitudeDegrees"].values,
            phone_data["LongitudeDegrees"].values,
            phone_data["PredLat"].values,
            phone_data["PredLon"].values,
        )

        p50 = np.percentile(dists, 50)
        p95 = np.percentile(dists, 95)

        score = (p50 + p95) / 2
        phone_scores.append(score)

    final_metric = np.mean(phone_scores)
    return final_metric, dists


def perform_failure_analysis(val_loader, model, device):
    """
    Correlates prediction error with input features to identify failure modes.
    """
    model.eval()
    all_errors = []
    all_features = []
    feature_names = Config.get_feature_names()

    print("\n--- Failure Analysis ---")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # [B, C, T]
            wls = batch["wls"].numpy()
            meta = batch["meta"]

            # Forward pass
            preds = model(features)
            if isinstance(preds, list):
                preds = preds[0]  # High-res output [B, 2, T]

            preds = preds.cpu().numpy()
            features_np = features.cpu().numpy()

            # Iterate through batch
            for i in range(preds.shape[0]):
                length = meta[i]["orig_length"]

                # Get valid sequence
                pred_n = preds[i, 0, :length]
                pred_e = preds[i, 1, :length]
                curr_wls = wls[i, :length, :]
                curr_feats = features_np[i, :, :length]  # [C, T]

                # Reconstruct Predictions
                pred_lats = []
                pred_lons = []
                gt_lats = []  # We need GT. It's not in batch directly in simple mode,
                # but targets are dN, dE from WLS.
                # We can approximate error magnitude directly from regression targets vs preds.

                # Use regression targets for error calculation to be faster/simpler
                # (Euclidean distance in ENU frame is close approximation to Haversine for small errors)
                if "targets" in batch:
                    targets = batch["targets"][i, :, :length].cpu().numpy()  # [2, T]
                    target_n = targets[0]
                    target_e = targets[1]

                    # Error magnitude (Euclidean distance in meters)
                    diff_n = pred_n - target_n
                    diff_e = pred_e - target_e
                    errors = np.sqrt(diff_n**2 + diff_e**2)

                    all_errors.append(errors)
                    all_features.append(curr_feats.T)  # [T, C]

    if not all_errors:
        print("No validation data for analysis.")
        return

    # Concatenate
    y_err = np.concatenate(all_errors)  # [N]
    X_feat = np.concatenate(all_features)  # [N, C]

    # Correlation
    correlations = {}
    for idx, name in enumerate(feature_names):
        if idx < X_feat.shape[1]:
            # Handle constant features
            if np.std(X_feat[:, idx]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(X_feat[:, idx], y_err)[0, 1]
            correlations[name] = corr

    # Sort and print
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")


def validate_and_score(model, val_loader, device):
    """
    Runs inference on validation set, reconstructs coords, and computes metric.
    """
    model.eval()
    results = []

    # We need ground truth. The dataset loads processed features.
    # To get GT Lat/Lon, we need to look up the metadata or reconstruction.
    # The Dataset's collate_fn doesn't pass raw GT Lat/Lon.
    # However, we can reconstruct the GT position from WLS + Targets (since Target = GT - WLS).
    # GT_pos = WLS + Target.

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            wls = batch["wls"].numpy()
            meta = batch["meta"]

            # Get targets if available
            if "targets" not in batch:
                continue

            targets = batch["targets"].numpy()  # [B, 2, T]

            # Forward
            preds = model(features)
            if isinstance(preds, list):
                preds = preds[0]
            preds = preds.cpu().numpy()

            for i in range(preds.shape[0]):
                length = meta[i]["orig_length"]
                phone_name = meta[i]["phone_name"]

                pred_n = preds[i, 0, :length]
                pred_e = preds[i, 1, :length]

                targ_n = targets[i, 0, :length]
                targ_e = targets[i, 1, :length]

                curr_wls = wls[i, :length, :]

                for t in range(length):
                    wx, wy, wz = curr_wls[t]
                    w_lat, w_lon, w_alt = ecef_to_wgs84(wx, wy, wz)

                    # Reconstruct Pred
                    pred_x, pred_y, pred_z = enu_to_ecef(
                        pred_e[t], pred_n[t], 0, w_lat, w_lon, w_alt
                    )
                    p_lat, p_lon, _ = ecef_to_wgs84(pred_x, pred_y, pred_z)

                    # Reconstruct GT (Target is offset from WLS)
                    gt_x, gt_y, gt_z = enu_to_ecef(
                        targ_e[t], targ_n[t], 0, w_lat, w_lon, w_alt
                    )
                    g_lat, g_lon, _ = ecef_to_wgs84(gt_x, gt_y, gt_z)

                    results.append(
                        {
                            "phone_name": phone_name,
                            "LatitudeDegrees": g_lat,
                            "LongitudeDegrees": g_lon,
                            "PredLat": p_lat,
                            "PredLon": p_lon,
                        }
                    )

    df_results = pd.DataFrame(results)
    metric, _ = calculate_competition_metric(df_results)
    return metric


def main():
    # 1. Configuration
    Config.setup()

    # Fast Baseline Settings
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size to fit in memory/time
    Config.NUM_WORKERS = 2

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on {device}")

    # 2. Dataset
    print("Loading Training Data...")
    # Limit drives for speed if needed, but we use cached data so it's fast
    train_dataset = SmartphoneLocationDataset(split="train", load_cached=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print("Loading Validation Data...")
    val_dataset = SmartphoneLocationDataset(split="val", load_cached=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 3. Model
    model = HR1DResNet()

    # 4. Training
    trainer = Trainer(model, device=device)
    # We save to a temp path first
    model = trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, save_path="baseline_model.pth"
    )

    # 5. Validation Assessment
    print("Computing Final Validation Metric...")
    val_metric = validate_and_score(model, val_loader, device)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_loader, model, device)

    # 7. Submission
    THRESHOLD = 3.7864967500302016
    if val_metric < THRESHOLD:
        print(f"Validation metric {val_metric} < {THRESHOLD}. Generating submission...")
        run_inference(
            checkpoint_path=os.path.join(Config.WORKING_DIR, "baseline_model.pth"),
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            device=Config.DEVICE,
            load_cached=True,
        )
    else:
        print(f"Validation metric {val_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
