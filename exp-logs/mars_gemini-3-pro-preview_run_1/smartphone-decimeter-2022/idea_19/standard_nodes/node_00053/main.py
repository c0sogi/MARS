import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.dataset import GNSSHeatmapDataset
from library.trainer import Trainer
from library.inference import generate_submission
from library.geo_utils import enu_to_wgs84


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    R = 6371000.0  # Radius of Earth in meters

    # Convert decimal degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config parameters to ensure fast execution and correct output paths
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    # Update Submission Directory to match requirements
    Config.SUBMISSION_DIR = "./submission"
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("=" * 40)
    print("FAST BASELINE RUN CONFIGURATION")
    print("=" * 40)
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")
    print(f"Submission Dir: {Config.SUBMISSION_DIR}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Datasets...")
    # Load cached data if available to save time
    train_dataset = GNSSHeatmapDataset(split="train", load_cached_data=True)
    val_dataset = GNSSHeatmapDataset(split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("\n[Step 2] Starting Training...")
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # -------------------------------------------------------------------------
    # 3. Validation Inference & Aggregation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Validation Inference...")
    trainer.load_best_model()
    model = trainer.model
    model.eval()
    device = torch.device(Config.DEVICE)

    # Dictionary to aggregate predictions: key=(drive_id, phone_name, timestamp)
    val_preds = {}

    # Map for failure analysis features
    feature_map = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            features = batch["features"].to(device)
            mask = batch["mask"].cpu().numpy()
            wls_pos = batch["wls_pos"].numpy()
            timestamps = batch["timestamps"].numpy()
            drive_indices = batch["drive_idx"].numpy()

            # Forward pass
            residuals = model(features).cpu().numpy()  # (B, T, 2)

            # Extract features for analysis (Mean over Azimuth for relevant channels)
            # Features: (B, C, T, A) -> (B, C, T)
            feat_cpu = batch["features"].numpy()
            feat_mean_az = np.mean(feat_cpu, axis=3)

            batch_size = features.shape[0]
            window_size = features.shape[2]

            for i in range(batch_size):
                drive_idx = drive_indices[i]
                drive_info = val_dataset.drives[drive_idx]
                drive_id = drive_info["drive_id"]
                phone_name = drive_info["phone_name"]

                for t in range(window_size):
                    # Check mask (1.0 = valid data)
                    if mask[i, t] > 0.5:
                        ts = timestamps[i, t]
                        key = (drive_id, phone_name, ts)

                        if key not in val_preds:
                            val_preds[key] = {
                                "dEast": [],
                                "dNorth": [],
                                "wls_lat": wls_pos[i, t, 0],
                                "wls_lon": wls_pos[i, t, 1],
                                "wls_alt": wls_pos[i, t, 2],
                            }

                        val_preds[key]["dEast"].append(residuals[i, t, 0])
                        val_preds[key]["dNorth"].append(residuals[i, t, 1])

                        # Store features for analysis (overwrite is fine as they are static per timestamp)
                        # Channels: 1:MeanCn0, 3:SatCount, 5:GlobalUnc
                        if key not in feature_map:
                            feature_map[key] = {
                                "Mean_Cn0": feat_mean_az[i, 1, t],
                                "Mean_SatCount": feat_mean_az[i, 3, t],
                                "Global_Unc": feat_mean_az[i, 5, t],
                            }

    # Aggregate predictions (average over overlapping windows)
    print("Aggregating validation predictions...")
    rows = []
    for key, values in val_preds.items():
        drive_id, phone_name, ts = key
        mean_dE = np.mean(values["dEast"])
        mean_dN = np.mean(values["dNorth"])

        # Convert ENU residuals + WLS baseline -> Pred Lat/Lon
        pred_lat, pred_lon, _ = enu_to_wgs84(
            mean_dE, mean_dN, 0, values["wls_lat"], values["wls_lon"], values["wls_alt"]
        )

        rows.append(
            {
                "drive_id": drive_id,
                "phone_name": phone_name,
                "UnixTimeMillis": ts,
                "Pred_Lat": pred_lat,
                "Pred_Lon": pred_lon,
            }
        )

    df_pred = pd.DataFrame(rows)

    # Load Ground Truth Metadata
    df_gt = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge Predictions with Ground Truth
    df_eval = pd.merge(
        df_gt, df_pred, on=["drive_id", "phone_name", "UnixTimeMillis"], how="inner"
    )

    # -------------------------------------------------------------------------
    # 4. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Calculating Metric...")
    # Calculate Haversine Distance Error
    df_eval["error_m"] = haversine_distance(
        df_eval["LatitudeDegrees"],
        df_eval["LongitudeDegrees"],
        df_eval["Pred_Lat"],
        df_eval["Pred_Lon"],
    )

    # Metric: Mean of (50th + 95th percentile) averaged over phones
    df_eval["phone_key"] = df_eval["drive_id"] + "_" + df_eval["phone_name"]

    phone_scores = []
    for phone, group in df_eval.groupby("phone_key"):
        p50 = np.percentile(group["error_m"], 50)
        p95 = np.percentile(group["error_m"], 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    final_metric = np.mean(phone_scores)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 5] Failure Analysis...")

    # Attach features to evaluation dataframe
    # Helper to map features back
    def get_feats(row):
        key = (row["drive_id"], row["phone_name"], row["UnixTimeMillis"])
        if key in feature_map:
            return pd.Series(feature_map[key])
        else:
            return pd.Series(
                [np.nan] * 3, index=["Mean_Cn0", "Mean_SatCount", "Global_Unc"]
            )

    feat_cols = df_eval.apply(get_feats, axis=1)
    df_analysis = pd.concat([df_eval, feat_cols], axis=1).dropna()

    # Calculate correlations
    print("Correlation between Error Magnitude and Input Features:")
    correlations = df_analysis[
        ["error_m", "Mean_Cn0", "Mean_SatCount", "Global_Unc"]
    ].corr()["error_m"]
    print(correlations.drop("error_m"))

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 3.802240262877392
    if final_metric < THRESHOLD:
        print(
            f"\n[Step 6] Metric {final_metric} < {THRESHOLD}. Generating Submission..."
        )
        # generate_submission handles test data loading, inference, and saving to Config.SUBMISSION_DIR
        generate_submission(load_cached_data=True)
    else:
        print(f"\n[Step 6] Metric {final_metric} >= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    run()
