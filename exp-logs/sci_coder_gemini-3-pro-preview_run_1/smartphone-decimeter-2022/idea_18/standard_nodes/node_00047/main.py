import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_competition_metric, haversine_distance
from library.preprocessing import PreProcessor
from library.dataset import GnssSequenceDataset
from library.model import SEResUNet1D
from library.train import train_model
from library.inference import predict_drive


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model (Fast Baseline: 1 Epoch)
    print("\n=== Starting Training (Fast Baseline) ===")
    # We use 1 epoch to satisfy the time constraint while using the full dataset structure
    train_model(debug=False, epochs=1)

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load processed validation data
    preprocessor = PreProcessor()
    # Ensure we load the same data used/generated during training
    _, val_df, _ = preprocessor.process_data(load_cached_data=True)

    # Create Dataset/Loader
    # mode='val' ensures full sequences are returned without windowing
    val_dataset = GnssSequenceDataset(
        val_df, mode="val", window_size=Config.TRAIN_WINDOW_SIZE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for variable sequence lengths in validation
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model file not found. Training might have failed.")
        return

    model = SEResUNet1D(in_channels=Config.INPUT_CHANNELS, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference on Validation Set
    all_preds_lat = []
    all_preds_lon = []
    all_gt_lat = []
    all_gt_lon = []
    all_phones = []
    all_timestamps = []

    print("Running validation inference...")
    with torch.no_grad():
        for i, (features, targets, meta) in enumerate(val_loader):
            features = features.to(device)

            # Forward pass (eval mode returns only final head)
            out_final = model(features)

            # Convert ENU offsets to Lat/Lon
            # Output: (B, 2, T) -> (T, 2) since B=1
            pred_enu = out_final.cpu().numpy()[0].T
            pred_e = pred_enu[:, 0]
            pred_n = pred_enu[:, 1]

            # Baseline WLS
            base_lat = meta["baseline_lat"].numpy()[0]
            base_lon = meta["baseline_lon"].numpy()[0]

            # Approximation for ENU -> Geodetic
            lat_scale = 111320.0
            lon_scale = 111320.0 * np.cos(np.radians(base_lat))

            pred_lat = base_lat + (pred_n / lat_scale)
            pred_lon = base_lon + (pred_e / lon_scale)

            # Ground Truth (reconstruct from targets to ensure alignment)
            target_enu = targets.numpy()[0].T
            target_e = target_enu[:, 0]
            target_n = target_enu[:, 1]

            gt_lat = base_lat + (target_n / lat_scale)
            gt_lon = base_lon + (target_e / lon_scale)

            # Metadata
            # meta['timestamp'] is a Tensor of shape (1, T)
            timestamps = meta["timestamp"][0].numpy()
            phone_name = meta["phone_name"][0]

            all_preds_lat.extend(pred_lat)
            all_preds_lon.extend(pred_lon)
            all_gt_lat.extend(gt_lat)
            all_gt_lon.extend(gt_lon)
            all_phones.extend([phone_name] * len(pred_lat))
            all_timestamps.extend(timestamps)

    # Create Results DataFrame
    df_results = pd.DataFrame(
        {
            "phone_name": all_phones,
            "UnixTimeMillis": all_timestamps,
            "LatitudeDegrees": all_preds_lat,
            "LongitudeDegrees": all_preds_lon,
            "GT_Lat": all_gt_lat,
            "GT_Lon": all_gt_lon,
        }
    )

    # Calculate Metric
    # We pass df_results as both pred and gt (using different columns)
    # calculate_competition_metric expects 'LatitudeDegrees' and 'LongitudeDegrees' in both
    df_pred_fmt = df_results[["phone_name", "LatitudeDegrees", "LongitudeDegrees"]]
    df_gt_fmt = df_results[["phone_name", "GT_Lat", "GT_Lon"]].rename(
        columns={"GT_Lat": "LatitudeDegrees", "GT_Lon": "LongitudeDegrees"}
    )

    final_metric = calculate_competition_metric(df_pred_fmt, df_gt_fmt)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error per point
    df_results["error_meters"] = haversine_distance(
        df_results["LatitudeDegrees"].values,
        df_results["LongitudeDegrees"].values,
        df_results["GT_Lat"].values,
        df_results["GT_Lon"].values,
    )

    # Merge with input features from val_df to analyze correlations
    # val_df has ['phone_name', 'UnixTimeMillis', features...]
    # Note: val_df might contain multiple drives with same phone/time?
    # No, (drive_id, phone, time) is unique. We used phone/time for merge, which is unique within val set usually.
    # To be safe, we rely on the fact that val_df was processed sequentially.

    # Let's merge on phone and time
    analysis_df = pd.merge(
        df_results, val_df, on=["phone_name", "UnixTimeMillis"], how="inner"
    )

    # Select features for correlation
    feature_cols = [
        "global_cn0_mean",
        "global_cn0_std",
        "global_elev_mean",
        "global_elev_std",
        "global_sat_count",
        "global_pr_unc_mean",
    ]

    # Check correlations
    if not analysis_df.empty:
        correlations = (
            analysis_df[feature_cols + ["error_meters"]]
            .corr()["error_meters"]
            .drop("error_meters")
        )
        print("Correlation between Input Features and Error Magnitude:")
        print(correlations.sort_values(ascending=False))
    else:
        print("Could not merge results with features for analysis.")

    # 4. Submission
    THRESHOLD = 3.802240262877392
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predict_drive(debug=False)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
