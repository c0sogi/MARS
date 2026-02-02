import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import WGS84Utils


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA) using WGS84 constants.
    Vectorized implementation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2  # eccentricity

    asq = a**2
    esq = e**2

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th) ** 3), (p - esq * a * np.cos(th) ** 3))
    N = a / np.sqrt(1 - esq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    return np.degrees(lat), np.degrees(lon), alt


def preprocess_drive(drive_id, phone_name, gnss_path, gt_df=None):
    """
    Reads raw GNSS data, aggregates it to 1Hz, computes features,
    and aligns with ground truth (if provided).
    """
    full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(full_gnss_path):
        print(f"Warning: GNSS file not found: {full_gnss_path}")
        return pd.DataFrame()

    # Load GNSS data
    # Use only necessary columns to save memory
    cols_to_load = [
        "utcTimeMillis",
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeUncertaintyMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    try:
        gnss_df = pd.read_csv(full_gnss_path, usecols=lambda c: c in cols_to_load)
    except ValueError:
        # Fallback if some columns are missing
        gnss_df = pd.read_csv(full_gnss_path)

    # 1. Temporal Quantization
    # Align timestamps to nearest second (1000ms)
    gnss_df["UnixTimeMillis"] = (
        np.round(gnss_df["utcTimeMillis"] / 1000.0) * 1000
    ).astype(np.int64)

    # 2. Feature Engineering & Aggregation
    # Pre-calculate Azimuth components
    gnss_df["SvAzimuthRad"] = np.deg2rad(gnss_df["SvAzimuthDegrees"].fillna(0))
    gnss_df["SvAzimuthDegrees_sin"] = np.sin(gnss_df["SvAzimuthRad"])
    gnss_df["SvAzimuthDegrees_cos"] = np.cos(gnss_df["SvAzimuthRad"])

    # Define aggregations
    aggs = {
        "Cn0DbHz": ["mean", "std", "min", "max"],
        "SvElevationDegrees": ["mean", "std", "min", "max"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "SvAzimuthDegrees_sin": ["mean"],
        "SvAzimuthDegrees_cos": ["mean"],
        "WlsPositionXEcefMeters": ["mean"],
        "WlsPositionYEcefMeters": ["mean"],
        "WlsPositionZEcefMeters": ["mean"],
        "utcTimeMillis": ["count"],  # SatCount
    }

    # Group by timestamp
    grouped = gnss_df.groupby("UnixTimeMillis")
    df_agg = grouped.agg(aggs)

    # Flatten MultiIndex columns
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()

    # Rename specific columns to match Config
    df_agg = df_agg.rename(
        columns={
            "utcTimeMillis_count": "SatCount",
            "SvAzimuthDegrees_sin_mean": "SvAzimuthDegrees_sin_mean",
            "SvAzimuthDegrees_cos_mean": "SvAzimuthDegrees_cos_mean",
        }
    )

    # 3. Compute Baseline (WLS) Lat/Lon
    # We use the mean ECEF position for that second
    wls_lat, wls_lon, _ = ecef_to_lla(
        df_agg["WlsPositionXEcefMeters_mean"],
        df_agg["WlsPositionYEcefMeters_mean"],
        df_agg["WlsPositionZEcefMeters_mean"],
    )
    df_agg["WlsLat"] = wls_lat
    df_agg["WlsLon"] = wls_lon

    # Fill NaNs in features (e.g. std is NaN if count=1)
    # Features starting with Cn0, SvElev, RawPseudo, SvAzim
    feat_cols = [
        c for c in df_agg.columns if any(x in c for x in ["Cn0", "Sv", "Raw", "Sat"])
    ]
    df_agg[feat_cols] = df_agg[feat_cols].fillna(0)

    # 4. Merge with Ground Truth (Training Mode)
    if gt_df is not None:
        # Align GT timestamps to nearest second to match GNSS aggregation
        # Cite debug_lesson_4: Normalize Timestamp Precision Before Merging Time-Series Data
        gt_df = gt_df.copy()
        gt_df["UnixTimeMillis"] = (
            np.round(gt_df["UnixTimeMillis"] / 1000.0) * 1000
        ).astype(np.int64)

        # gt_df has columns: UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
        df_merged = pd.merge(df_agg, gt_df, on="UnixTimeMillis", how="inner")

        # Compute Targets (Delta Meters)
        delta_north, delta_east = WGS84Utils.degrees_to_meters(
            df_merged["LatitudeDegrees"].values,
            df_merged["LongitudeDegrees"].values,
            df_merged["WlsLat"].values,
            df_merged["WlsLon"].values,
        )
        df_merged["DeltaNorth"] = delta_north
        df_merged["DeltaEast"] = delta_east

        final_df = df_merged
    else:
        # Inference Mode
        final_df = df_agg
        # Add dummy targets for consistency
        final_df["DeltaNorth"] = 0.0
        final_df["DeltaEast"] = 0.0

    # Add identifiers
    final_df["drive_id"] = drive_id
    final_df["phone_name"] = phone_name
    final_df["trip_id"] = f"{drive_id}-{phone_name}"

    return final_df


def load_data(metadata_path, cache_path, load_cached_data=True):
    """
    Loads and preprocesses data. Uses parquet caching.
    Returns a single concatenated DataFrame of all trips.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # 3. Identify Unique Trips
    # We process by unique (drive_id, phone_name) pairs
    unique_trips = meta_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    processed_dfs = []

    print(f"Processing {len(unique_trips)} trips from {metadata_path}...")

    for _, row in unique_trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        # Get GT for this trip if available
        trip_gt = meta_df[
            (meta_df["drive_id"] == drive_id) & (meta_df["phone_name"] == phone_name)
        ]

        # If metadata has Lat/Lon columns, pass them as GT
        if "LatitudeDegrees" in trip_gt.columns:
            gt_subset = trip_gt[
                ["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
            ].copy()
        else:
            gt_subset = None

        df_trip = preprocess_drive(drive_id, phone_name, gnss_path, gt_subset)

        if not df_trip.empty:
            processed_dfs.append(df_trip)

    if not processed_dfs:
        print("Warning: No data processed!")
        return pd.DataFrame()

    master_df = pd.concat(processed_dfs, ignore_index=True)

    # 4. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    master_df.to_parquet(cache_path, index=False)
    print(f"Saved processed data to {cache_path}")

    return master_df


class GNSSDataset(Dataset):
    """
    PyTorch Dataset for GNSS sequences.
    Groups the master DataFrame by trip_id and yields full drive sequences.
    """

    def __init__(self, data_df):
        self.data_df = data_df
        # Cite debug_lesson_5: Validate Data Volume After Filtering
        if self.data_df.empty:
            raise ValueError(
                "GNSSDataset received an empty DataFrame. Check data loading and preprocessing."
            )

        self.feature_cols = Config.INPUT_FEATURES
        self.target_cols = ["DeltaEast", "DeltaNorth"]  # Order matters: x, y

        # Create an index of trips
        # We group by trip_id and store the start/end indices or the group itself
        # Storing indices into the main dataframe is memory efficient
        if "trip_id" not in self.data_df.columns:
            raise KeyError(
                "Column 'trip_id' missing from data_df. Ensure preprocessing adds identifiers."
            )

        self.trip_ids = self.data_df["trip_id"].unique()
        self.trip_indices = {}

        # Sort dataframe by trip and time to ensure sequential order
        self.data_df = self.data_df.sort_values(
            ["trip_id", "UnixTimeMillis"]
        ).reset_index(drop=True)

        # Build index map: trip_id -> (start_idx, end_idx)
        # This assumes data is sorted by trip_id
        for trip_id, group in self.data_df.groupby("trip_id"):
            self.trip_indices[trip_id] = group.index.values

    def __len__(self):
        return len(self.trip_ids)

    def __getitem__(self, idx):
        trip_id = self.trip_ids[idx]
        indices = self.trip_indices[trip_id]

        # Extract data for this trip
        trip_data = self.data_df.iloc[indices]

        # Features
        features = trip_data[self.feature_cols].values.astype(np.float32)

        # Targets
        targets = trip_data[self.target_cols].values.astype(np.float32)

        # Metadata needed for inference reconstruction
        # We return the WLS baseline and timestamps
        wls_pos = trip_data[["WlsLat", "WlsLon"]].values.astype(np.float64)
        timestamps = trip_data["UnixTimeMillis"].values.astype(np.int64)

        return {
            "features": torch.tensor(features),
            "targets": torch.tensor(targets),
            "wls_pos": torch.tensor(wls_pos),
            "timestamps": torch.tensor(timestamps),
            "trip_id": trip_id,
        }


def gnss_collate_fn(batch):
    """
    Collate function to pad variable length sequences.
    """
    features_list = [item["features"] for item in batch]
    targets_list = [item["targets"] for item in batch]
    wls_list = [item["wls_pos"] for item in batch]
    time_list = [item["timestamps"] for item in batch]
    trip_ids = [item["trip_id"] for item in batch]

    # Pad sequences (Batch, Seq, Feat)
    # batch_first=True
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)

    # Create padding mask (Batch, Seq)
    # 1 for valid data, 0 for padding
    lengths = torch.tensor([len(f) for f in features_list])
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # For metadata, we can just keep them as lists or pad them if needed for tensor ops
    # Usually metadata is used in loop, not batch ops, but WLS might be needed for loss if we used lat/lon loss
    # Here we stick to lists for non-model inputs to avoid float precision loss in padding

    return {
        "features": features_padded,
        "targets": targets_padded,
        "mask": mask,
        "wls_pos": wls_list,  # List of tensors
        "timestamps": time_list,  # List of tensors
        "trip_ids": trip_ids,
    }
