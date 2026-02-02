import os
import json
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WINDOW_SIZE,
    KINEMATIC_FEATURES,
    CONTEXT_FEATURES,
    TARGET_COLS,
    DEBUG,
)
from library.utils import wgs84_to_enu, ecef_to_lla

# Constants for caching
SCALER_PATH = os.path.join(WORKING_DIR, "scaler_stats.json")


class GNSSPreprocessor:
    def __init__(self):
        self.scaler_stats = {}
        self.feature_means = None
        self.feature_stds = None

    def _load_metadata(self, split):
        if split == "train":
            return pd.read_csv(TRAIN_METADATA_PATH)
        elif split == "validation":
            return pd.read_csv(VAL_METADATA_PATH)
        elif split == "test":
            return pd.read_csv(TEST_METADATA_PATH)
        else:
            raise ValueError(f"Unknown split: {split}")

    def _aggregate_gnss_epoch(self, df_gnss):
        """
        Aggregates raw GNSS measurements into epoch-level statistics.
        """
        # Define aggregation dictionary
        agg_dict = {
            "WlsPositionXEcefMeters": "first",
            "WlsPositionYEcefMeters": "first",
            "WlsPositionZEcefMeters": "first",
            "SvElevationDegrees": ["mean", "std"],
            "Cn0DbHz": "mean",
            "RawPseudorangeUncertaintyMeters": "mean",
        }

        # Group by timestamp
        df_epoch = df_gnss.groupby("utcTimeMillis").agg(agg_dict)

        # Flatten MultiIndex columns
        df_epoch.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in df_epoch.columns.values
        ]

        # Rename for clarity
        rename_map = {
            "WlsPositionXEcefMeters_first": "WlsX",
            "WlsPositionYEcefMeters_first": "WlsY",
            "WlsPositionZEcefMeters_first": "WlsZ",
            "SvElevationDegrees_mean": "mean_sv_elevation",
            "SvElevationDegrees_std": "std_sv_elevation",
            "Cn0DbHz_mean": "mean_cn0",
            "RawPseudorangeUncertaintyMeters_mean": "mean_uncertainty",
        }
        df_epoch = df_epoch.rename(columns=rename_map)

        # Fill NaNs in std (occurs if only 1 satellite)
        df_epoch["std_sv_elevation"] = df_epoch["std_sv_elevation"].fillna(0)

        return df_epoch.sort_index()

    def _add_kinematic_features(self, df):
        """
        Computes LLA from ECEF and adds velocity/delta features.
        """
        # Convert WLS ECEF to LLA
        lat, lon, alt = ecef_to_lla(
            df["WlsX"].values, df["WlsY"].values, df["WlsZ"].values
        )

        df["wls_lat"] = lat
        df["wls_lon"] = lon
        df["wls_alt"] = alt

        # Calculate deltas (dynamics) - Simple difference implies velocity per second if 1Hz
        # We assume 1Hz data roughly.
        df["d_lat"] = df["wls_lat"].diff().fillna(0)
        df["d_lon"] = df["wls_lon"].diff().fillna(0)
        df["d_alt"] = df["wls_alt"].diff().fillna(0)

        # Convert deltas to meters approx for scaling consistency
        # 1 deg lat ~ 111320m
        meters_per_deg = 111320.0
        df["d_lat_m"] = df["d_lat"] * meters_per_deg
        df["d_lon_m"] = df["d_lon"] * meters_per_deg * np.cos(np.radians(df["wls_lat"]))
        df["d_alt_m"] = df["d_alt"]  # Already in meters

        # Map raw signal metrics to kinematic feature names expected by model
        # We use mean values as the representative signal quality for the epoch
        df["Cn0DbHz"] = df["mean_cn0"]
        df["Uncertainty"] = df["mean_uncertainty"]

        return df

    def _create_windows_and_targets(self, df_trip, trip_meta, is_train):
        """
        Creates sliding windows and calculates targets for requested timestamps.
        """
        X_kinematic_list = []
        X_context_list = []
        y_list = []
        valid_indices = []

        # Ensure trip data is sorted by time
        df_trip = df_trip.sort_index()

        # Reindex to handle gaps (1 second intervals)
        # Find min and max time in both GNSS and Metadata to cover full range
        min_t = min(df_trip.index.min(), trip_meta["UnixTimeMillis"].min())
        max_t = max(df_trip.index.max(), trip_meta["UnixTimeMillis"].max())

        # Create 1s grid
        full_grid = np.arange(min_t, max_t + 1000, 1000)
        df_trip = (
            df_trip.reindex(full_grid).interpolate(method="linear").bfill().ffill()
        )

        # Half window size
        half_window = WINDOW_SIZE // 2

        # Pre-compute LLA for the whole trip for speed
        wls_lats = df_trip["wls_lat"].values
        wls_lons = df_trip["wls_lon"].values
        wls_alts = df_trip["wls_alt"].values

        # Feature arrays
        # Features: rel_lat_m, rel_lon_m, rel_alt_m, d_lat_m, d_lon_m, d_alt_m, Cn0, Unc
        # Note: rel_lat/lon/alt depend on the window center, so we extract raw first
        feat_cols = [
            "wls_lat",
            "wls_lon",
            "wls_alt",
            "d_lat_m",
            "d_lon_m",
            "d_alt_m",
            "Cn0DbHz",
            "Uncertainty",
        ]
        raw_feats = df_trip[feat_cols].values

        # Context cols
        ctx_cols = [
            "mean_sv_elevation",
            "std_sv_elevation",
            "mean_cn0",
            "mean_uncertainty",
        ]
        raw_ctx = df_trip[ctx_cols].values

        # Map timestamps to integer indices in the reindexed dataframe
        # Timestamps are indices in df_trip
        time_to_idx = {t: i for i, t in enumerate(df_trip.index)}

        meters_per_deg = 111320.0

        for idx, row in trip_meta.iterrows():
            query_time = row["UnixTimeMillis"]

            # Find closest grid timestamp
            # Since we reindexed to 1000ms, we can round
            grid_time = int(round(query_time / 1000.0) * 1000)

            if grid_time not in time_to_idx:
                continue

            center_idx = time_to_idx[grid_time]

            # Check bounds
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window + 1  # +1 for slice

            if start_idx < 0 or end_idx > len(df_trip):
                # Padding logic could go here, but for simplicity we skip edge cases in training
                # For test, we might replicate edges.
                # Given dataset size, simple edge replication for test:
                if not is_train:
                    # Clamp indices
                    indices = np.arange(start_idx, end_idx)
                    indices = np.clip(indices, 0, len(df_trip) - 1)
                    window_data = raw_feats[indices]
                    window_ctx = raw_ctx[indices]
                else:
                    continue
            else:
                window_data = raw_feats[start_idx:end_idx]
                window_ctx = raw_ctx[start_idx:end_idx]

            # --- Kinematic Feature Engineering (Relative to Center) ---
            center_lat = wls_lats[center_idx]
            center_lon = wls_lons[center_idx]
            center_alt = wls_alts[center_idx]

            # Copy window data to avoid modifying source
            # Columns: 0:lat, 1:lon, 2:alt, 3:dlat, 4:dlon, 5:dalt, 6:cn0, 7:unc
            win_feats = window_data.copy()

            # Calculate relative positions in meters
            # lat diff
            win_feats[:, 0] = (win_feats[:, 0] - center_lat) * meters_per_deg
            # lon diff (scaled by cos(center_lat))
            win_feats[:, 1] = (
                (win_feats[:, 1] - center_lon)
                * meters_per_deg
                * np.cos(np.radians(center_lat))
            )
            # alt diff
            win_feats[:, 2] = win_feats[:, 2] - center_alt

            X_kinematic_list.append(win_feats)

            # --- Context Feature Engineering ---
            # Average over the window
            X_context_list.append(np.mean(window_ctx, axis=0))

            # --- Target Generation ---
            if is_train:
                gt_lat = row["LatitudeDegrees"]
                gt_lon = row["LongitudeDegrees"]
                gt_alt = (
                    row["AltitudeDegrees"] if "AltitudeDegrees" in row else 0
                )  # Some GT might miss alt, assume 0 or WLS

                # We need target: GT - WLS_Center (in meters)
                # We use wgs84_to_enu
                # Ref: WLS Center
                # Target: GT
                e, n, u = wgs84_to_enu(
                    gt_lat, gt_lon, gt_alt, center_lat, center_lon, center_alt
                )
                y_list.append([e, n])

            valid_indices.append(idx)

        return X_kinematic_list, X_context_list, y_list, valid_indices

    def _fit_scaler(self, X_kin, X_ctx):
        # Flatten kinematic to (N*T, F) for stats
        kin_flat = np.concatenate(X_kin, axis=0)
        ctx_flat = np.array(X_ctx)

        self.feature_means = {
            "kin_mean": np.mean(kin_flat, axis=0).tolist(),
            "kin_std": np.std(kin_flat, axis=0).tolist(),
            "ctx_mean": np.mean(ctx_flat, axis=0).tolist(),
            "ctx_std": np.std(ctx_flat, axis=0).tolist(),
        }

        # Avoid div by zero
        self.feature_means["kin_std"] = [
            s if s > 1e-6 else 1.0 for s in self.feature_means["kin_std"]
        ]
        self.feature_means["ctx_std"] = [
            s if s > 1e-6 else 1.0 for s in self.feature_means["ctx_std"]
        ]

        # Save
        with open(SCALER_PATH, "w") as f:
            json.dump(self.feature_means, f)

    def _load_scaler(self):
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler stats not found. Run training split first.")
        with open(SCALER_PATH, "r") as f:
            self.feature_means = json.load(f)

    def _transform(self, X_kin_list, X_ctx_list):
        if self.feature_means is None:
            self._load_scaler()

        k_mean = np.array(self.feature_means["kin_mean"])
        k_std = np.array(self.feature_means["kin_std"])
        c_mean = np.array(self.feature_means["ctx_mean"])
        c_std = np.array(self.feature_means["ctx_std"])

        X_kin_norm = []
        for x in X_kin_list:
            X_kin_norm.append((x - k_mean) / k_std)

        X_ctx_norm = (np.array(X_ctx_list) - c_mean) / c_std

        return np.array(X_kin_norm, dtype=np.float32), X_ctx_norm.astype(np.float32)

    def process_data(self, split="train", load_cached_data=True):
        """
        Main driver function to process data for a specific split.
        """
        # Cache paths
        cache_kin = os.path.join(WORKING_DIR, f"{split}_X_kinematic.npy")
        cache_ctx = os.path.join(WORKING_DIR, f"{split}_X_context.npy")
        cache_y = os.path.join(WORKING_DIR, f"{split}_y.npy")
        cache_meta = os.path.join(WORKING_DIR, f"{split}_meta.parquet")

        # 1. Try Loading Cache
        if load_cached_data:
            if (
                os.path.exists(cache_kin)
                and os.path.exists(cache_ctx)
                and os.path.exists(cache_meta)
            ):
                if split != "test" and not os.path.exists(cache_y):
                    pass  # Train/Val must have y
                else:
                    print(f"Loading cached data for {split}...")
                    X_kin = np.load(cache_kin)
                    X_ctx = np.load(cache_ctx)
                    meta = pd.read_parquet(cache_meta)
                    y = np.load(cache_y) if split != "test" else None
                    return X_kin, X_ctx, y, meta

        # 2. Compute from Scratch
        print(f"Processing {split} data from scratch...")
        meta_df = self._load_metadata(split)

        if DEBUG:
            print("DEBUG MODE: Sampling subset of trips.")
            trips = meta_df["tripId"].unique()[:2]
            meta_df = meta_df[meta_df["tripId"].isin(trips)].copy()

        # Group by trip
        trips = meta_df["tripId"].unique()

        all_X_kin = []
        all_X_ctx = []
        all_y = []
        all_meta_indices = []

        for trip_id in trips:
            trip_meta = meta_df[meta_df["tripId"] == trip_id]

            # Get file path from first row
            gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
            gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)

            if not os.path.exists(gnss_path):
                print(f"Warning: GNSS file not found: {gnss_path}")
                continue

            # Load and aggregate GNSS
            df_gnss = pd.read_csv(gnss_path)
            df_epoch = self._aggregate_gnss_epoch(df_gnss)

            # Add features
            df_features = self._add_kinematic_features(df_epoch)

            # Create windows
            is_train = split != "test"
            x_k, x_c, y, valid_idx = self._create_windows_and_targets(
                df_features, trip_meta, is_train
            )

            all_X_kin.extend(x_k)
            all_X_ctx.extend(x_c)
            all_y.extend(y)
            all_meta_indices.extend(valid_idx)

        # Filter metadata to matched rows
        meta_df_processed = meta_df.loc[all_meta_indices].reset_index(drop=True)

        # Scaling
        if split == "train":
            self._fit_scaler(all_X_kin, all_X_ctx)

        X_kin_arr, X_ctx_arr = self._transform(all_X_kin, all_X_ctx)

        if split != "test":
            y_arr = np.array(all_y, dtype=np.float32)
        else:
            y_arr = np.array([])  # Empty for test

        # 3. Save to Cache
        print(f"Saving {split} data to cache...")
        np.save(cache_kin, X_kin_arr)
        np.save(cache_ctx, X_ctx_arr)
        meta_df_processed.to_parquet(cache_meta)
        if split != "test":
            np.save(cache_y, y_arr)

        return (
            X_kin_arr,
            X_ctx_arr,
            (y_arr if split != "test" else None),
            meta_df_processed,
        )


def load_data(split="train", load_cached_data=True):
    """
    Wrapper function to initialize preprocessor and load data.
    """
    preprocessor = GNSSPreprocessor()
    return preprocessor.process_data(split, load_cached_data)
