import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import geodetic_to_enu

# WGS84 Constants for ECEF to LLA
A = 6378137.0
B = 6356752.314245
E_SQ = 6.69437999014e-3
E_PRIME_SQ = 6.73949674228e-3


def ecef_to_lla(x, y, z):
    """
    Vectorized implementation of ECEF to Geodetic (Lat, Lon, Alt)
    Using Ferrari's solution.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + E_PRIME_SQ * B * np.sin(theta) ** 3, p - E_SQ * A * np.cos(theta) ** 3
    )

    # Convert to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg


def preprocess_drive(metadata_row, split, load_cached_data=True):
    """
    Preprocesses a single drive: aggregates GNSS, aligns with GT (if available),
    calculates residuals, and caches the result.
    """
    drive_id = metadata_row["drive_id"]
    phone_name = metadata_row["phone_name"]

    # Cache filename
    cache_path = os.path.join(
        Config.WORKING_DIR, f"{drive_id}_{phone_name}_{split}.parquet"
    )

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 1. Load Raw GNSS
    gnss_path = os.path.join(Config.INPUT_DIR, metadata_row["gnss_path"])
    if not os.path.exists(gnss_path):
        return pd.DataFrame()

    gnss_df = pd.read_csv(gnss_path)

    # 2. Temporal Quantization (1Hz)
    if "utcTimeMillis" in gnss_df.columns:
        gnss_df["UnixTimeMillis"] = (
            np.round(gnss_df["utcTimeMillis"] / 1000) * 1000
        ).astype(np.int64)
    else:
        return pd.DataFrame()

    # 3. Aggregation
    # Define aggregation dictionary
    agg_dict = {}
    for feat in Config.RAW_FEATURES:
        if feat in gnss_df.columns:
            for stat in Config.AGG_STATS:
                agg_dict[feat] = agg_dict.get(feat, []) + [stat]

    # Add SatCount (count of Svid)
    agg_dict["Svid"] = ["count"]

    # WLS position for baseline
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    for col in wls_cols:
        if col in gnss_df.columns:
            # Use mean instead of first to avoid NaNs if the first row is missing data
            # Cite debug_lesson_6
            agg_dict[col] = ["mean"]

    # Perform groupby
    grouped = gnss_df.groupby("UnixTimeMillis")
    df_agg = grouped.agg(agg_dict)

    # Flatten MultiIndex columns
    new_columns = []
    for col_tuple in df_agg.columns:
        if col_tuple[0] == "Svid" and col_tuple[1] == "count":
            new_columns.append("SatCount")
        elif col_tuple[1] == "mean" and col_tuple[0] in wls_cols:
            new_columns.append(col_tuple[0])
        elif col_tuple[1] == "first":
            new_columns.append(col_tuple[0])
        else:
            new_columns.append(f"{col_tuple[0]}_{col_tuple[1]}")

    df_agg.columns = new_columns
    df_agg = df_agg.reset_index()

    # 4. Convert WLS ECEF to LLA (Baseline)
    if all(c in df_agg.columns for c in wls_cols):
        wls_lat, wls_lon = ecef_to_lla(
            df_agg["WlsPositionXEcefMeters"].values,
            df_agg["WlsPositionYEcefMeters"].values,
            df_agg["WlsPositionZEcefMeters"].values,
        )
        df_agg["WlsLatitudeDegrees"] = wls_lat
        df_agg["WlsLongitudeDegrees"] = wls_lon
    else:
        # If WLS missing, we can't compute residuals. Skip.
        return pd.DataFrame()

    # 5. Align with Ground Truth (for Train/Val)
    if split in ["train", "val"]:
        gt_path = os.path.join(
            Config.INPUT_DIR, "train", drive_id, phone_name, "ground_truth.csv"
        )

        if os.path.exists(gt_path):
            df_gt = pd.read_csv(gt_path)

            # Normalize timestamps to 1Hz grid to match GNSS aggregation
            # Cite debug_lesson_4
            df_gt["UnixTimeMillis"] = (
                np.round(df_gt["UnixTimeMillis"] / 1000) * 1000
            ).astype(np.int64)

            # Merge Aggregated GNSS with GT
            df_proc = pd.merge(
                df_agg,
                df_gt[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
                on="UnixTimeMillis",
                how="inner",
            )

            if df_proc.empty:
                # Cite debug_lesson_5: Validate data volume
                print(
                    f"Warning: Empty intersection between GNSS and GT for {drive_id} {phone_name}"
                )
                return pd.DataFrame()

            # Calculate Residuals (Target)
            d_east, d_north = geodetic_to_enu(
                df_proc["LatitudeDegrees"].values,
                df_proc["LongitudeDegrees"].values,
                df_proc["WlsLatitudeDegrees"].values,
                df_proc["WlsLongitudeDegrees"].values,
            )

            df_proc["DeltaEast"] = d_east
            df_proc["DeltaNorth"] = d_north

        else:
            return pd.DataFrame()

    else:
        # Test mode: No GT.
        df_proc = df_agg.copy()
        df_proc["DeltaEast"] = 0.0
        df_proc["DeltaNorth"] = 0.0

    # Fill NaNs
    df_proc = df_proc.fillna(0)

    # Save to cache
    df_proc.to_parquet(cache_path)

    return df_proc


class GNSSSequenceDataset(Dataset):
    def __init__(self, processed_dfs, mode="train"):
        self.mode = mode
        self.samples = []

        # Feature columns
        self.feature_cols = []
        for feat in Config.RAW_FEATURES:
            for stat in Config.AGG_STATS:
                self.feature_cols.append(f"{feat}_{stat}")
        self.feature_cols.extend(Config.META_FEATURES)

        if processed_dfs and len(processed_dfs) > 0:
            available_cols = processed_dfs[0].columns
            self.feature_cols = [c for c in self.feature_cols if c in available_cols]

        window_size = Config.WINDOW_SIZE

        for df in processed_dfs:
            if df.empty:
                continue

            features = df[self.feature_cols].values.astype(np.float32)
            targets = df[Config.TARGET_COLS].values.astype(np.float32)

            length = len(df)

            if mode == "train":
                stride = window_size // 2
                for start in range(0, length - window_size + 1, stride):
                    end = start + window_size
                    self.samples.append(
                        {"features": features[start:end], "targets": targets[start:end]}
                    )
            elif mode == "val":
                stride = window_size
                for start in range(0, length, stride):
                    end = min(start + window_size, length)
                    feat_window = features[start:end]
                    targ_window = targets[start:end]

                    if len(feat_window) < window_size:
                        pad_len = window_size - len(feat_window)
                        feat_window = np.pad(
                            feat_window, ((0, pad_len), (0, 0)), "constant"
                        )
                        targ_window = np.pad(
                            targ_window, ((0, pad_len), (0, 0)), "constant"
                        )

                    self.samples.append(
                        {"features": feat_window, "targets": targ_window}
                    )
            else:
                # Test: Full sequence
                self.samples.append(
                    {
                        "features": features,
                        "targets": targets,
                        "wls_lat": df["WlsLatitudeDegrees"].values,
                        "wls_lon": df["WlsLongitudeDegrees"].values,
                        "timestamps": df["UnixTimeMillis"].values,
                        "drive_id": df.get("drive_id", "unknown"),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = torch.tensor(sample["features"].transpose(1, 0), dtype=torch.float32)
        targets = torch.tensor(sample["targets"].transpose(1, 0), dtype=torch.float32)

        if self.mode == "test":
            return {
                "features": features,
                "wls_lat": sample["wls_lat"],
                "wls_lon": sample["wls_lon"],
                "timestamps": sample["timestamps"],
            }

        return features, targets


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test.
    """
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Train
    train_drives = train_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()
    if Config.DEBUG:
        train_drives = train_drives.head(Config.DEBUG_SAMPLE_SIZE)

    train_dfs = []
    for _, row in train_drives.iterrows():
        df = preprocess_drive(row, "train", load_cached_data)
        if not df.empty:
            train_dfs.append(df)

    # Val
    val_drives = val_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()
    if Config.DEBUG:
        val_drives = val_drives.head(Config.DEBUG_SAMPLE_SIZE)

    val_dfs = []
    for _, row in val_drives.iterrows():
        df = preprocess_drive(row, "val", load_cached_data)
        if not df.empty:
            val_dfs.append(df)

    # Test
    test_drives = test_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()
    test_dfs = []
    for _, row in test_drives.iterrows():
        df = preprocess_drive(row, "test", load_cached_data)
        if not df.empty:
            df["drive_id"] = row["drive_id"]
            df["phone_name"] = row["phone_name"]
            test_dfs.append(df)

    train_dataset = GNSSSequenceDataset(train_dfs, mode="train")
    val_dataset = GNSSSequenceDataset(val_dfs, mode="val")
    test_dataset = GNSSSequenceDataset(test_dfs, mode="test")

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
