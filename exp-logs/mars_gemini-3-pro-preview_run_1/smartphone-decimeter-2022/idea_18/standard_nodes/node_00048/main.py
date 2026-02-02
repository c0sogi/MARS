import os
import shutil
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_competition_metric, haversine_distance
from library.preprocessing import PreProcessor
from library.dataset import GnssSequenceDataset
from library.model import SEResUNet1D
from library.train import train_model
from library.inference import predict_drive


def regenerate_metadata():
    print("Regenerating metadata with AltitudeMeters...")

    if not os.path.exists(Config.METADATA_DIR):
        os.makedirs(Config.METADATA_DIR)

    # 1. Process Training Data
    train_data = []
    train_base_path = os.path.join(Config.INPUT_DIR, "train")

    if os.path.exists(train_base_path):
        drives = [
            d
            for d in os.listdir(train_base_path)
            if os.path.isdir(os.path.join(train_base_path, d))
        ]

        for drive_id in drives:
            drive_path = os.path.join(train_base_path, drive_id)
            phones = [
                p
                for p in os.listdir(drive_path)
                if os.path.isdir(os.path.join(drive_path, p))
            ]

            for phone_name in phones:
                phone_path = os.path.join(drive_path, phone_name)
                gt_path = os.path.join(phone_path, "ground_truth.csv")

                if os.path.exists(gt_path):
                    df_gt = pd.read_csv(gt_path)

                    # Construct relative paths
                    rel_path_prefix = os.path.join("train", drive_id, phone_name)
                    gnss_path = os.path.join(rel_path_prefix, "device_gnss.csv")
                    imu_path = os.path.join(rel_path_prefix, "device_imu.csv")

                    df_gt["drive_id"] = drive_id
                    df_gt["phone_name"] = phone_name
                    df_gt["gnss_path"] = gnss_path
                    df_gt["imu_path"] = imu_path

                    # Include AltitudeMeters
                    cols = [
                        "drive_id",
                        "phone_name",
                        "UnixTimeMillis",
                        "LatitudeDegrees",
                        "LongitudeDegrees",
                        "AltitudeMeters",  # Added this
                        "gnss_path",
                        "imu_path",
                    ]

                    # Filter to ensure we only select existing columns
                    cols = [c for c in cols if c in df_gt.columns]
                    train_data.append(df_gt[cols])

    if train_data:
        full_train_df = pd.concat(train_data, ignore_index=True)
    else:
        full_train_df = pd.DataFrame(
            columns=[
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
                "gnss_path",
                "imu_path",
            ]
        )

    # 2. Split Train/Val
    if not full_train_df.empty:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        groups = full_train_df["drive_id"]
        train_idx, val_idx = next(splitter.split(full_train_df, groups=groups))

        train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_train_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df = full_train_df.copy()
        val_df = pd.DataFrame(columns=full_train_df.columns)

    train_df.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    val_df.to_csv(Config.VAL_METADATA_PATH, index=False)

    # 3. Process Test Data
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        test_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        parsed = test_df["tripId"].str.rsplit("-", n=1)
        test_df["drive_id"] = parsed.str[0]
        test_df["phone_name"] = parsed.str[1]

        test_df["gnss_path"] = test_df.apply(
            lambda x: os.path.join(
                "test", x["drive_id"], x["phone_name"], "device_gnss.csv"
            ),
            axis=1,
        )
        test_df["imu_path"] = test_df.apply(
            lambda x: os.path.join(
                "test", x["drive_id"], x["phone_name"], "device_imu.csv"
            ),
            axis=1,
        )

        cols = [
            "tripId",
            "drive_id",
            "phone_name",
            "UnixTimeMillis",
            "gnss_path",
            "imu_path",
        ]
        test_df = test_df[cols]
    else:
        test_df = pd.DataFrame(
            columns=[
                "tripId",
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "gnss_path",
                "imu_path",
            ]
        )

    test_df.to_csv(Config.TEST_METADATA_PATH, index=False)
    print("Metadata regeneration complete.")


def clean_cache():
    if os.path.exists(Config.CACHE_DIR):
        print(f"Clearing cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
    else:
        print(f"Cache directory {Config.CACHE_DIR} does not exist.")


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Regenerate metadata to include AltitudeMeters
    regenerate_metadata()
    # Clean cache to force reprocessing (Cite debug_lesson_13)
    clean_cache()

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
