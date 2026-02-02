import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.geo_utils import wgs84_to_enu, ecef_to_wgs84


class DataProcessor:
    """
    Handles the conversion of raw GNSS logs into Spatio-Temporal Sky Heatmaps
    and generates regression targets relative to WLS baselines.
    """

    def __init__(self):
        self.azimuth_bins = Config.AZIMUTH_BINS
        self.bin_width = 360.0 / self.azimuth_bins
        self.cache_dir = Config.CACHE_DIR

    def _get_metadata(self, split):
        if split == "train":
            return pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif split == "val":
            return pd.read_csv(Config.VAL_METADATA_PATH)
        elif split == "test":
            return pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid split: {split}")

    def _process_drive(self, drive_id, phone_name, df_meta_drive, is_test=False):
        """
        Process a single drive: create heatmap features and targets.
        """
        # 1. Load Raw GNSS Data
        # We take the first row's path because the file is the same for the whole drive/phone
        gnss_rel_path = df_meta_drive.iloc[0]["gnss_path"]
        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found: {gnss_path}")
            return None

        # Load only necessary columns to save memory
        use_cols = [
            "utcTimeMillis",
            "SvAzimuthDegrees",
            "SvElevationDegrees",
            "Cn0DbHz",
            "RawPseudorangeUncertaintyMeters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        try:
            df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in use_cols)
        except ValueError:
            # Fallback if some columns are missing (though they should be there based on description)
            df_gnss = pd.read_csv(gnss_path)

        # 2. Temporal Alignment using merge_asof
        # Ensure timestamps are sorted for merge_asof
        df_gnss = df_gnss.sort_values("utcTimeMillis")
        targets = (
            df_meta_drive[["UnixTimeMillis"]]
            .drop_duplicates()
            .sort_values("UnixTimeMillis")
        )

        # Use merge_asof to find the nearest Target for each GNSS observation
        # Tolerance: 1000 ms (1 second)
        # We want to keep GNSS rows and attach the nearest target timestamp
        df_merged = pd.merge_asof(
            df_gnss,
            targets,
            left_on="utcTimeMillis",
            right_on="UnixTimeMillis",
            direction="nearest",
            tolerance=1000,
        )

        # Drop rows where no target was found (UnixTimeMillis is NaN)
        df_gnss = df_merged.dropna(subset=["UnixTimeMillis"]).copy()

        if df_gnss.empty:
            return None

        # 3. Handle WLS Baseline
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        # Ensure columns exist, create if missing
        for col in wls_cols:
            if col not in df_gnss.columns:
                df_gnss[col] = np.nan

        # Interpolate WLS to fill gaps within the drive
        # We interpolate on the full GNSS dataframe before grouping
        df_gnss[wls_cols] = df_gnss[wls_cols].interpolate(
            method="linear", limit_direction="both"
        )

        # If still NaN (e.g. all missing), fill with 0.0
        df_gnss[wls_cols] = df_gnss[wls_cols].fillna(0.0)

        # 4. Calculate WLS Baseline Positions (Lat/Lon) per timestamp
        # We average WLS positions per timestamp if there are multiple entries
        df_wls = df_gnss.groupby("UnixTimeMillis")[wls_cols].mean().reset_index()

        # Convert WLS ECEF to WGS84
        wls_lat, wls_lon, wls_alt = ecef_to_wgs84(
            df_wls["WlsPositionXEcefMeters"].values,
            df_wls["WlsPositionYEcefMeters"].values,
            df_wls["WlsPositionZEcefMeters"].values,
        )
        df_wls["wls_lat"] = wls_lat
        df_wls["wls_lon"] = wls_lon
        df_wls["wls_alt"] = wls_alt

        # 5. Construct Spatio-Temporal Heatmap
        # Fill NaNs in critical columns to prevent loss of data during aggregation
        # Cite debug_lesson_2: Sanitize Raw Sensor Inputs
        if "RawPseudorangeUncertaintyMeters" in df_gnss.columns:
            df_gnss["RawPseudorangeUncertaintyMeters"] = df_gnss[
                "RawPseudorangeUncertaintyMeters"
            ].fillna(100.0)
        if "Cn0DbHz" in df_gnss.columns:
            df_gnss["Cn0DbHz"] = df_gnss["Cn0DbHz"].fillna(0.0)
        if "SvElevationDegrees" in df_gnss.columns:
            df_gnss["SvElevationDegrees"] = df_gnss["SvElevationDegrees"].fillna(0.0)

        # Filter out invalid Azimuth values (NaN/Inf) which cause casting errors
        # Cite debug_lesson_2: Sanitize Raw Sensor Inputs
        valid_az = np.isfinite(df_gnss["SvAzimuthDegrees"])
        if not valid_az.all():
            df_gnss = df_gnss[valid_az].copy()

        if df_gnss.empty:
            return None

        # Bin Azimuth
        df_gnss["azimuth_bin"] = (df_gnss["SvAzimuthDegrees"] // self.bin_width).astype(
            int
        )
        df_gnss["azimuth_bin"] = df_gnss["azimuth_bin"].clip(0, self.azimuth_bins - 1)

        # Global Statistics per timestamp
        global_stats = (
            df_gnss.groupby("UnixTimeMillis")
            .agg(
                global_sat_count=("Cn0DbHz", "count"),
                global_mean_pr_unc=("RawPseudorangeUncertaintyMeters", "mean"),
            )
            .reset_index()
        )

        # Bin Statistics
        # We aggregate multiple signals/satellites falling into the same bin
        bin_stats = (
            df_gnss.groupby(["UnixTimeMillis", "azimuth_bin"])
            .agg(
                max_cn0=("Cn0DbHz", "max"),
                mean_cn0=("Cn0DbHz", "mean"),
                mean_elev=("SvElevationDegrees", "mean"),
                sat_count=("Cn0DbHz", "count"),
            )
            .reset_index()
        )

        # Create a complete grid (Time x Azimuth)
        # We use pivot to create the 2D structure
        # Resulting DataFrame columns will be MultiIndex
        pivot_df = bin_stats.pivot(
            index="UnixTimeMillis",
            columns="azimuth_bin",
            values=["max_cn0", "mean_cn0", "mean_elev", "sat_count"],
        )

        # Reindex to ensure all bins 0..35 exist and all timestamps exist
        pivot_df = pivot_df.reindex(
            columns=pd.MultiIndex.from_product(
                [
                    ["max_cn0", "mean_cn0", "mean_elev", "sat_count"],
                    range(self.azimuth_bins),
                ]
            ),
            fill_value=0,
        )

        # Sort by timestamp to ensure alignment
        pivot_df = pivot_df.sort_index()

        # Extract arrays
        # Shape: (T, 4, 36) -> Transpose to (T, 36, 4)
        # Note: pivot columns are (Feature, Bin)

        T = len(pivot_df)
        K = self.azimuth_bins

        # Stack features: (T, K, 4)
        # Order: MaxCn0, MeanCn0, MeanElev, SatCount
        f1 = pivot_df["max_cn0"].values
        f2 = pivot_df["mean_cn0"].values
        f3 = pivot_df["mean_elev"].values
        f4 = pivot_df["sat_count"].values

        heatmap_core = np.stack([f1, f2, f3, f4], axis=2)  # (T, K, 4)

        # Add Global Context (Broadcasted)
        # Ensure global stats index matches pivot index
        global_stats = (
            global_stats.set_index("UnixTimeMillis").reindex(pivot_df.index).fillna(0)
        )

        g1 = global_stats["global_sat_count"].values[
            :, np.newaxis, np.newaxis
        ]  # (T, 1, 1)
        g2 = global_stats["global_mean_pr_unc"].values[
            :, np.newaxis, np.newaxis
        ]  # (T, 1, 1)

        # Broadcast to (T, K, 1)
        g1 = np.tile(g1, (1, K, 1))
        g2 = np.tile(g2, (1, K, 1))

        # Final Feature Tensor: (T, K, 6)
        features = np.concatenate([heatmap_core, g1, g2], axis=2)

        # Timestamps for this drive
        timestamps = pivot_df.index.values

        # Align WLS baseline
        # df_wls index is UnixTimeMillis, reindex to match
        df_wls = df_wls.set_index("UnixTimeMillis").reindex(timestamps)
        wls_pos = df_wls[["wls_lat", "wls_lon", "wls_alt"]].values

        # 6. Targets (if train/val)
        targets = None
        if not is_test:
            # Merge GT with timestamps
            # df_meta_drive contains GT Lat/Lon
            df_gt = df_meta_drive.set_index("UnixTimeMillis").reindex(timestamps)

            # Calculate ENU residuals
            # Target = GT - WLS (in meters)
            gt_lat = df_gt["LatitudeDegrees"].values
            gt_lon = df_gt["LongitudeDegrees"].values

            # Handle potential NaNs if GT is missing for some timestamps (shouldn't happen with inner join logic, but safe to check)
            # We assume metadata generation ensured existence, but reindexing might introduce NaNs if timestamps mismatch slightly.
            # However, we filtered GNSS by `target_timestamps` earlier, so alignment should be exact.

            # WLS is reference
            ref_lat = wls_pos[:, 0]
            ref_lon = wls_pos[:, 1]
            ref_alt = wls_pos[:, 2]  # Use WLS altitude as reference altitude

            e, n, u = wgs84_to_enu(gt_lat, gt_lon, ref_lat, ref_lon, ref_alt)

            # We predict East and North offsets. Up is usually ignored or less critical for horizontal accuracy.
            targets = np.stack([e, n], axis=1)  # (T, 2)

            # Robust Filtering: NaNs, Infs, and Extreme Outliers
            # 1. Finite Targets (No NaN, No Inf)
            mask_target_finite = np.isfinite(targets).all(axis=1)

            # 2. Reasonable Magnitude (< 20,000km error)
            # Relaxed to allow training even with poor baselines (e.g. 0,0,0)
            mask_target_mag = (np.abs(targets) < 20000000).all(axis=1)

            # 3. Finite Features (Just in case)
            # features shape: (T, K, C) -> reshape to check all
            mask_feat_finite = np.isfinite(features.reshape(features.shape[0], -1)).all(
                axis=1
            )

            # Combine masks
            valid_mask = mask_target_finite & mask_target_mag & mask_feat_finite

            if not np.all(valid_mask):
                features = features[valid_mask]
                targets = targets[valid_mask]
                wls_pos = wls_pos[valid_mask]
                timestamps = timestamps[valid_mask]

            if len(features) == 0:
                return None

        return {
            "drive_id": drive_id,
            "phone_name": phone_name,
            "features": features.astype(np.float32),  # (T, 36, 6)
            "targets": (
                targets.astype(np.float32) if targets is not None else None
            ),  # (T, 2)
            "wls_pos": wls_pos.astype(
                np.float64
            ),  # (T, 3) - Keep float64 for precision
            "timestamps": timestamps,  # (T,)
        }

    def process_data(self, split="train", load_cached_data=True, save_cache=True):
        """
        Main method to process a dataset split.
        """
        cache_file = os.path.join(self.cache_dir, f"{split}_processed.npz")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} data from cache: {cache_file}")
            try:
                data = np.load(cache_file, allow_pickle=True)
                # Reconstruct list of dicts
                processed_data = []
                # We store keys as 'drive_0_features', 'drive_0_meta', etc.
                # To simplify, we store a list of dicts in the object structure if pickle were allowed,
                # but with npz we iterate keys.

                # Get number of drives
                num_drives = data["num_drives"].item()

                for i in range(num_drives):
                    item = {
                        "drive_id": str(data[f"drive_{i}_id"]),
                        "phone_name": str(data[f"drive_{i}_phone"]),
                        "features": data[f"drive_{i}_features"],
                        "wls_pos": data[f"drive_{i}_wls_pos"],
                        "timestamps": data[f"drive_{i}_timestamps"],
                    }
                    if f"drive_{i}_targets" in data:
                        item["targets"] = data[f"drive_{i}_targets"]
                    else:
                        item["targets"] = None
                    processed_data.append(item)

                print(f"Successfully loaded {len(processed_data)} drives.")
                return processed_data
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from Scratch
        print(f"Processing {split} data from raw files...")
        df_meta = self._get_metadata(split)

        # Group by drive and phone
        grouped = df_meta.groupby(["drive_id", "phone_name"])

        processed_data = []

        # Use tqdm for progress
        for (drive_id, phone_name), group in tqdm(grouped, desc=f"Processing {split}"):
            drive_data = self._process_drive(
                drive_id, phone_name, group, is_test=(split == "test")
            )
            if drive_data is not None:
                processed_data.append(drive_data)

        # 3. Save Cache
        if save_cache and processed_data:
            print(f"Saving {split} data to cache: {cache_file}")
            save_dict = {"num_drives": len(processed_data)}
            for i, item in enumerate(processed_data):
                save_dict[f"drive_{i}_id"] = item["drive_id"]
                save_dict[f"drive_{i}_phone"] = item["phone_name"]
                save_dict[f"drive_{i}_features"] = item["features"]
                save_dict[f"drive_{i}_wls_pos"] = item["wls_pos"]
                save_dict[f"drive_{i}_timestamps"] = item["timestamps"]
                if item["targets"] is not None:
                    save_dict[f"drive_{i}_targets"] = item["targets"]

            np.savez_compressed(cache_file, **save_dict)

        return processed_data
