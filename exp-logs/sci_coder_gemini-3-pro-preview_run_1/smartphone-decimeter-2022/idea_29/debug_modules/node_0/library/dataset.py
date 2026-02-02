import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library.config import Config
from library.data_preprocessing import process_dataset, load_gnss_log, align_timestamps
from library.utils import enu_to_ecef, ecef_to_wgs84


class SmartphoneLocationDataset(Dataset):
    def __init__(self, split="train", load_cached=True, max_drives=None):
        """
        PyTorch Dataset for Smartphone Location.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to load cached parquet files.
            max_drives (int, optional): Limit number of drives for debugging.
        """
        self.split = split

        # Determine paths based on split
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.CACHE_TRAIN
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
            cache_path = Config.CACHE_VAL
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
            cache_path = Config.CACHE_TEST
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load processed features (and targets for train/val)
        # This uses the provided library function which handles caching
        self.df = process_dataset(meta_path, cache_path, load_cached_data=load_cached)

        # Load metadata CSV to retrieve file paths for WLS coordinates
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        self.meta_df = pd.read_csv(meta_path)

        self.sequences = []

        if self.df.empty:
            print(f"Warning: Dataset for split {split} is empty.")
            return

        # Group data by drive and phone to form sequences
        groups = self.df.groupby(["drive_id", "phone_name"])
        keys = list(groups.groups.keys())

        if max_drives:
            keys = keys[:max_drives]

        print(f"Preparing sequences for {len(keys)} drives ({split})...")

        for k in keys:
            drive_id, phone_name = k
            # Get the sequence data, sorted by time
            group_df = groups.get_group(k).copy().sort_values("UnixTimeMillis")

            # Retrieve WLS coordinates from raw GNSS file
            # We need to find the path in the metadata
            # Filter meta_df for this drive/phone. Take the first row to get the path.
            meta_subset = self.meta_df[
                (self.meta_df["drive_id"] == drive_id)
                & (self.meta_df["phone_name"] == phone_name)
            ]

            if meta_subset.empty:
                print(f"Warning: No metadata found for {drive_id} {phone_name}")
                continue

            gnss_rel_path = meta_subset.iloc[0]["gnss_path"]
            gnss_full_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

            # Load WLS and align timestamps
            try:
                # Load only necessary columns to save time
                # load_gnss_log already filters columns, including WLS
                gnss_log = load_gnss_log(gnss_full_path)
                gnss_log = align_timestamps(gnss_log)

                # There might be multiple entries per timestamp (one per satellite)
                # We just need the WLS position of the receiver, which is repeated.
                # Group by time and take first.
                wls_df = gnss_log.groupby("UnixTimeMillis").first().reset_index()
                wls_cols = [
                    "UnixTimeMillis",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
                wls_df = wls_df[wls_cols]

                # Merge WLS into the main sequence dataframe
                # We use left join on group_df to ensure we keep the sequence structure
                group_df = pd.merge(group_df, wls_df, on="UnixTimeMillis", how="left")

                # Fill missing WLS (if any) with interpolation or 0 (shouldn't happen often if aligned correctly)
                wls_coords = group_df[wls_cols[1:]]
                if wls_coords.isnull().any().any():
                    group_df[wls_cols[1:]] = wls_coords.interpolate(
                        limit_direction="both"
                    ).fillna(0)

            except Exception as e:
                print(f"Error loading WLS for {drive_id} {phone_name}: {e}")
                # Fill with zeros if loading fails
                group_df["WlsPositionXEcefMeters"] = 0.0
                group_df["WlsPositionYEcefMeters"] = 0.0
                group_df["WlsPositionZEcefMeters"] = 0.0

            # Prepare Tensors
            # Features
            feature_cols = Config.get_feature_names()
            # Ensure columns exist (process_dataset should handle this, but double check)
            for f in feature_cols:
                if f not in group_df.columns:
                    group_df[f] = 0.0

            x = group_df[feature_cols].values.astype(np.float32)
            x = x.transpose(1, 0)  # [C, T]

            # WLS
            wls = group_df[
                [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ].values.astype(np.float64)

            # Meta
            timestamps = group_df["UnixTimeMillis"].values.astype(np.int64)

            item = {
                "features": torch.from_numpy(x),
                "wls": torch.from_numpy(wls),
                "timestamps": torch.from_numpy(timestamps),
                "drive_id": drive_id,
                "phone_name": phone_name,
            }

            # Targets (if available)
            if "Target_E" in group_df.columns and "Target_N" in group_df.columns:
                # Targets might be NaN in test set, handle that
                if split != "test":
                    # Stack as [Target_N, Target_E] (North, East)
                    y = group_df[["Target_N", "Target_E"]].values.astype(np.float32)
                    y = np.nan_to_num(y, nan=0.0)  # Safety
                    y = y.transpose(1, 0)  # [2, T]
                    item["targets"] = torch.from_numpy(y)

            self.sequences.append(item)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    """
    Collate function to pad sequences to the same length in a batch.
    """
    # Find max length in this batch
    max_len = max([b["features"].shape[1] for b in batch])

    features_list = []
    targets_list = []
    wls_list = []
    masks_list = []
    meta_list = []

    for b in batch:
        length = b["features"].shape[1]
        pad_len = max_len - length

        # Pad Features [C, T] -> Pad last dim
        f = b["features"]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, pad_len), value=0)
        features_list.append(f)

        # Create Mask [T]
        mask = torch.ones(length, dtype=torch.bool)
        if pad_len > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)
        masks_list.append(mask)

        # Pad WLS [T, 3] -> Pad dim 0
        w = b["wls"]
        if pad_len > 0:
            # Create padding of shape [pad_len, 3]
            pad_w = torch.zeros((pad_len, 3), dtype=w.dtype)
            w = torch.cat([w, pad_w], dim=0)
        wls_list.append(w)

        # Pad Targets [2, T] -> Pad last dim
        if "targets" in b:
            t = b["targets"]
            if pad_len > 0:
                t = torch.nn.functional.pad(t, (0, pad_len), value=0)
            targets_list.append(t)

        # Meta info
        meta_list.append(
            {
                "drive_id": b["drive_id"],
                "phone_name": b["phone_name"],
                "timestamps": b["timestamps"],  # numpy array
                "orig_length": length,
            }
        )

    batch_res = {
        "features": torch.stack(features_list),  # [B, C, T]
        "mask": torch.stack(masks_list),  # [B, T]
        "wls": torch.stack(wls_list),  # [B, T, 3]
        "meta": meta_list,
    }

    if targets_list:
        batch_res["targets"] = torch.stack(targets_list)  # [B, 2, T]

    return batch_res


def generate_submission(model, dataloader, output_path=Config.SUBMISSION_PATH):
    """
    Generates submission file by running inference and reconstructing coordinates.
    """
    model.eval()
    device = Config.DEVICE

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features = batch["features"].to(device)
            # mask = batch["mask"].to(device) # Not strictly needed for inference if we use length
            wls = batch["wls"].numpy()  # Keep on CPU for reconstruction
            meta = batch["meta"]

            # Forward pass
            # Model output expected: [B, 2, T] (Delta North, Delta East)
            preds = model(features)

            # Handle deep supervision output (list of outputs)
            if isinstance(preds, list) or isinstance(preds, tuple):
                preds = preds[0]  # Take High-Res output

            preds = preds.cpu().numpy()

            batch_size = preds.shape[0]

            for i in range(batch_size):
                length = meta[i]["orig_length"]
                drive_id = meta[i]["drive_id"]
                phone_name = meta[i]["phone_name"]
                timestamps = meta[i]["timestamps"]

                # Trip ID for submission
                trip_id = f"{drive_id}-{phone_name}"

                # Get valid sequence parts
                # Preds: [2, T] -> (dN, dE)
                # Note: We stacked targets as [Target_N, Target_E] in __init__
                # So index 0 is North, index 1 is East
                pred_n = preds[i, 0, :length]
                pred_e = preds[i, 1, :length]

                curr_wls = wls[i, :length, :]  # [T, 3] (X, Y, Z)

                # Reconstruct
                for t in range(length):
                    ts = timestamps[t]

                    # Baseline WLS ECEF
                    wx, wy, wz = curr_wls[t]

                    # Convert WLS to LLA to get reference Lat/Lon/Alt
                    w_lat, w_lon, w_alt = ecef_to_wgs84(wx, wy, wz)

                    # Convert Predicted ENU Deltas (East, North, Up=0) to ECEF Deltas
                    dn = pred_n[t]
                    de = pred_e[t]

                    # enu_to_ecef returns absolute ECEF coordinates
                    pred_x, pred_y, pred_z = enu_to_ecef(de, dn, 0, w_lat, w_lon, w_alt)

                    # Convert back to WGS84 Lat/Lon
                    p_lat, p_lon, _ = ecef_to_wgs84(pred_x, pred_y, pred_z)

                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": ts,
                            "LatitudeDegrees": p_lat,
                            "LongitudeDegrees": p_lon,
                        }
                    )

    # Create DataFrame
    pred_df = pd.DataFrame(results)

    # Load sample submission to ensure correct rows and order
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)

        # Create a mapping column in sample_df
        # Round to nearest second (1000ms) to match our aligned predictions
        sample_df["AlignedTime"] = (
            np.round(sample_df["UnixTimeMillis"] / 1000) * 1000
        ).astype(np.int64)

        # Rename pred_df columns for merge
        pred_df = pred_df.rename(
            columns={
                "UnixTimeMillis": "AlignedTime",
                "LatitudeDegrees": "PredLat",
                "LongitudeDegrees": "PredLon",
            }
        )

        # Merge
        # We merge on tripId and AlignedTime
        merged = pd.merge(sample_df, pred_df, on=["tripId", "AlignedTime"], how="left")

        # Update columns
        # If PredLat is NaN, keep original
        merged["LatitudeDegrees"] = merged["PredLat"].fillna(merged["LatitudeDegrees"])
        merged["LongitudeDegrees"] = merged["PredLon"].fillna(
            merged["LongitudeDegrees"]
        )

        submission_df = merged[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]
    else:
        submission_df = pred_df

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
