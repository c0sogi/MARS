import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import wls_to_meters, setup_logger

# Setup logger
logger = setup_logger("data_loader.log")


def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude (WGS84).
    Vectorized implementation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Altitude (approximate)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


class GNSSDataset(Dataset):
    def __init__(self, X_kin, X_sky, y=None, indices=None):
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_sky = torch.FloatTensor(X_sky)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.indices = indices  # To track original metadata indices if needed

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_kin[idx], self.X_sky[idx], self.y[idx]
        return self.X_kin[idx], self.X_sky[idx]


class DataProcessor:
    def __init__(self, mode="train"):
        self.mode = mode
        self.scaler_kin = StandardScaler()
        self.scaler_sky = StandardScaler()
        self.is_fitted = False

        if mode == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif mode == "validation":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif mode == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def save_scalers(self, path):
        import joblib

        scalers = {"kin": self.scaler_kin, "sky": self.scaler_sky}
        joblib.dump(scalers, path)
        logger.info(f"Scalers saved to {path}")

    def load_scalers(self, path):
        import joblib

        if os.path.exists(path):
            scalers = joblib.load(path)
            self.scaler_kin = scalers["kin"]
            self.scaler_sky = scalers["sky"]
            self.is_fitted = True
            logger.info(f"Scalers loaded from {path}")
        else:
            logger.warning(f"Scaler file not found at {path}")

    def _aggregate_gnss(self, df_gnss):
        """
        Aggregates raw GNSS data by epoch (utcTimeMillis).
        """
        # Calculate derived raw features if needed before aggregation
        # E.g. Elevation/Azimuth are already there.

        # Group by epoch
        # We take the first WLS position as it is common for the epoch
        agg_funcs = {
            "WlsPositionXEcefMeters": "first",
            "WlsPositionYEcefMeters": "first",
            "WlsPositionZEcefMeters": "first",
            "Cn0DbHz": ["mean", "std"],
            "RawPseudorangeUncertaintyMeters": "mean",
            "SvElevationDegrees": ["mean", "std"],
            "SvAzimuthDegrees": "mean",  # Circular mean is better but simple mean for now
        }

        # Check if columns exist
        available_funcs = {}
        for col, func in agg_funcs.items():
            if col in df_gnss.columns:
                available_funcs[col] = func

        # Add satellite count
        if "Svid" in df_gnss.columns:
            # Just counting rows per group
            pass

        grouped = df_gnss.groupby("utcTimeMillis")
        df_agg = grouped.agg(available_funcs)

        # Flatten columns
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]

        # Add satellite count
        df_agg["sat_count"] = grouped.size()

        # Rename for consistency
        rename_map = {
            "WlsPositionXEcefMeters_first": "wls_x",
            "WlsPositionYEcefMeters_first": "wls_y",
            "WlsPositionZEcefMeters_first": "wls_z",
            "Cn0DbHz_mean": "mean_cn0",
            "Cn0DbHz_std": "std_cn0",
            "RawPseudorangeUncertaintyMeters_mean": "mean_unc",
            "SvElevationDegrees_mean": "mean_elev",
            "SvElevationDegrees_std": "std_elev",
            "SvAzimuthDegrees_mean": "mean_azim",
        }
        df_agg.rename(columns=rename_map, inplace=True)

        # Drop epochs with missing WLS baseline
        # This is critical to avoid NaNs in coordinate conversion and target calculation
        df_agg.dropna(subset=["wls_x", "wls_y", "wls_z"], inplace=True)

        # Fill NaNs in other columns (e.g. std columns for single satellite epochs)
        df_agg.fillna(0, inplace=True)

        # Convert WLS ECEF to LLA
        lat, lon, alt = ecef_to_lla(
            df_agg["wls_x"].values, df_agg["wls_y"].values, df_agg["wls_z"].values
        )
        df_agg["wls_lat"] = lat
        df_agg["wls_lon"] = lon
        df_agg["wls_alt"] = alt

        return df_agg.reset_index()

    def _create_windows(self, df_trip, window_size):
        """
        Creates sliding windows from aggregated trip data.
        Returns lists of kinematic arrays, sky arrays, and targets (if available).
        """
        X_kin_list = []
        X_sky_list = []
        y_list = []
        indices_list = []  # utcTimeMillis of center

        # Ensure sorted by time
        df_trip = df_trip.sort_values("utcTimeMillis").reset_index(drop=True)

        timestamps = df_trip["utcTimeMillis"].values
        half_window = window_size // 2

        # Arrays for fast access
        lats = df_trip["wls_lat"].values
        lons = df_trip["wls_lon"].values
        alts = df_trip["wls_alt"].values

        # Sky features
        sky_cols = [
            "mean_cn0",
            "std_cn0",
            "mean_unc",
            "mean_elev",
            "std_elev",
            "mean_azim",
            "sat_count",
        ]
        # Ensure columns exist
        existing_sky_cols = [c for c in sky_cols if c in df_trip.columns]
        sky_data = df_trip[existing_sky_cols].values

        # Pre-calculate deltas (velocity) for the whole sequence
        # Pad first element to keep length same
        delta_lat = np.diff(lats, prepend=lats[0]) * Config.DEG_TO_M_LAT
        # Lon delta depends on lat, approx for velocity
        delta_lon = (
            np.diff(lons, prepend=lons[0])
            * Config.DEG_TO_M_LAT
            * np.cos(np.radians(lats))
        )
        delta_alt = np.diff(alts, prepend=alts[0])

        # Signal metrics for kinematic stream (per epoch)
        # We use mean_cn0 and mean_unc as per-step features
        cn0_seq = df_trip["mean_cn0"].values
        unc_seq = df_trip["mean_unc"].values

        # Targets
        has_target = "dLat_m" in df_trip.columns
        if has_target:
            targets = df_trip[["dLat_m", "dLon_m"]].values

        n_samples = len(df_trip)

        for i in range(n_samples):
            # Define window indices
            start_idx = i - half_window
            end_idx = i + half_window + 1  # Slice is exclusive at end

            # Check bounds
            if start_idx < 0 or end_idx > n_samples:
                continue

            # Check time continuity
            # We expect 1Hz data. Window span should be approx window_size * 1000 ms
            t_start = timestamps[start_idx]
            t_end = timestamps[end_idx - 1]
            expected_span = (window_size - 1) * 1000
            if (
                abs((t_end - t_start) - expected_span) > 2000
            ):  # Allow 2s tolerance for jitter
                continue

            # Center values
            center_lat = lats[i]
            center_lon = lons[i]
            center_alt = alts[i]

            # Construct Kinematic Sequence
            # Relative coordinates
            win_lats = (lats[start_idx:end_idx] - center_lat) * Config.DEG_TO_M_LAT
            win_lons = (
                (lons[start_idx:end_idx] - center_lon)
                * Config.DEG_TO_M_LAT
                * np.cos(np.radians(center_lat))
            )
            win_alts = alts[start_idx:end_idx] - center_alt

            win_dlat = delta_lat[start_idx:end_idx]
            win_dlon = delta_lon[start_idx:end_idx]
            win_dalt = delta_alt[start_idx:end_idx]

            win_cn0 = cn0_seq[start_idx:end_idx]
            win_unc = unc_seq[start_idx:end_idx]

            # Stack features: (Window, Features)
            kin_features = np.stack(
                [
                    win_lats,
                    win_lons,
                    win_alts,
                    win_dlat,
                    win_dlon,
                    win_dalt,
                    win_cn0,
                    win_unc,
                ],
                axis=1,
            )

            # Construct Sky Context (Average over window)
            # Or just take center? The idea says "Aggregated statistics for the window"
            # Let's take the mean of the sky stats over the window
            win_sky = np.mean(sky_data[start_idx:end_idx], axis=0)

            X_kin_list.append(kin_features)
            X_sky_list.append(win_sky)
            indices_list.append(timestamps[i])

            if has_target:
                y_list.append(targets[i])

        return X_kin_list, X_sky_list, y_list, indices_list

    def process_data(self, load_cached_data=True, sample_frac=1.0):
        """
        Main data processing function.
        """
        # Cache filenames
        cache_X_kin = Config.get_npy_cache_path(f"{self.mode}_X_kin")
        cache_X_sky = Config.get_npy_cache_path(f"{self.mode}_X_sky")
        cache_y = Config.get_npy_cache_path(f"{self.mode}_y")
        cache_meta = Config.get_cache_path(f"{self.mode}_meta")
        scaler_path = os.path.join(Config.WORKING_DIR, "scaler.joblib")

        # Try loading cache
        if load_cached_data:
            if os.path.exists(cache_X_kin) and os.path.exists(cache_X_sky):
                logger.info(f"Loading cached {self.mode} data...")
                X_kin = np.load(cache_X_kin)
                X_sky = np.load(cache_X_sky)

                # Load scaler if not fitted (e.g. inference mode)
                if not self.is_fitted and os.path.exists(scaler_path):
                    self.load_scalers(scaler_path)

                y = np.load(cache_y) if os.path.exists(cache_y) else None

                # If training, we might want to fit scaler on loaded data if not loaded
                if self.mode == "train" and not self.is_fitted:
                    # Flatten kin for scaling
                    B, L, D = X_kin.shape
                    self.scaler_kin.fit(X_kin.reshape(-1, D))
                    self.scaler_sky.fit(X_sky)
                    self.is_fitted = True
                    self.save_scalers(scaler_path)

                return X_kin, X_sky, y

        # Compute from scratch
        logger.info(f"Computing {self.mode} data from scratch...")
        df_meta = pd.read_csv(self.metadata_path)

        # Subsample for debugging if requested
        if sample_frac < 1.0:
            trips = df_meta["tripId"].unique()
            n_sample = int(len(trips) * sample_frac)
            sampled_trips = np.random.choice(trips, n_sample, replace=False)
            df_meta = df_meta[df_meta["tripId"].isin(sampled_trips)]
            logger.info(f"Subsampled {n_sample} trips.")

        all_X_kin = []
        all_X_sky = []
        all_y = []

        unique_trips = df_meta["tripId"].unique()

        for trip_id in unique_trips:
            trip_meta = df_meta[df_meta["tripId"] == trip_id]
            if trip_meta.empty:
                continue

            # Load GNSS
            gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
            gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

            if not os.path.exists(gnss_path):
                continue

            df_gnss = pd.read_csv(gnss_path)

            # Aggregate
            df_agg = self._aggregate_gnss(df_gnss)

            # If train/val, merge with Ground Truth to get targets
            if self.mode in ["train", "validation"]:
                # GT data is in trip_meta
                # Need to merge on time
                # trip_meta has UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
                df_merged = pd.merge(
                    df_agg,
                    trip_meta[
                        ["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
                    ],
                    left_on="utcTimeMillis",
                    right_on="UnixTimeMillis",
                    how="inner",
                )

                if df_merged.empty:
                    continue

                # Compute residuals
                d_lat, d_lon = wls_to_meters(
                    df_merged["wls_lat"].values,
                    df_merged["wls_lon"].values,
                    df_merged["LatitudeDegrees"].values,
                    df_merged["LongitudeDegrees"].values,
                )
                df_merged["dLat_m"] = d_lat
                df_merged["dLon_m"] = d_lon

                df_proc = df_merged
            else:
                # Test mode: we need to keep all aggregated epochs,
                # but we are specifically interested in the ones requested in sample_submission
                # However, for windowing, we need the context.
                # So we process the whole trip, and later filter or align.
                # Actually, the window generation logic aligns by index.
                # For test, we should probably generate windows for ALL timestamps,
                # and then select the ones matching the submission file.
                df_proc = df_agg

            # Create windows
            X_kin, X_sky, y, indices = self._create_windows(df_proc, Config.WINDOW_SIZE)

            if not X_kin:
                continue

            # Filter for Test: Only keep windows centered on requested timestamps
            if self.mode == "test":
                req_times = set(trip_meta["UnixTimeMillis"].values)
                # Filter
                valid_indices = []
                for i, t in enumerate(indices):
                    if t in req_times:
                        valid_indices.append(i)

                if not valid_indices:
                    continue

                X_kin = [X_kin[i] for i in valid_indices]
                X_sky = [X_sky[i] for i in valid_indices]
                # y is empty for test

            all_X_kin.extend(X_kin)
            all_X_sky.extend(X_sky)
            all_y.extend(y)

        # Convert to numpy
        X_kin_arr = np.array(all_X_kin)
        X_sky_arr = np.array(all_X_sky)
        y_arr = (
            np.array(all_y) if all_y else np.zeros((len(X_kin_arr), 2))
        )  # Dummy y for test

        # Scaling
        if self.mode == "train":
            # Fit scalers
            B, L, D = X_kin_arr.shape
            self.scaler_kin.fit(X_kin_arr.reshape(-1, D))
            self.scaler_sky.fit(X_sky_arr)
            self.is_fitted = True
            self.save_scalers(scaler_path)
        elif not self.is_fitted:
            # Load if validation/test and not fitted (e.g. separate run)
            if os.path.exists(scaler_path):
                self.load_scalers(scaler_path)
            else:
                logger.warning(
                    "No scaler found for validation/test! Using unscaled data (bad)."
                )

        # Transform
        if self.is_fitted:
            B, L, D = X_kin_arr.shape
            X_kin_arr = self.scaler_kin.transform(X_kin_arr.reshape(-1, D)).reshape(
                B, L, D
            )
            X_sky_arr = self.scaler_sky.transform(X_sky_arr)

        # Save to cache
        np.save(cache_X_kin, X_kin_arr)
        np.save(cache_X_sky, X_sky_arr)
        if self.mode != "test":
            np.save(cache_y, y_arr)

        logger.info(f"Processed {len(X_kin_arr)} samples for {self.mode}.")
        return X_kin_arr, X_sky_arr, y_arr if self.mode != "test" else None


def get_dataloaders(batch_size=Config.BATCH_SIZE, sample_frac=1.0):
    """
    Creates DataLoaders for train and validation.
    """
    # Process Train
    train_processor = DataProcessor(mode="train")
    X_kin_train, X_sky_train, y_train = train_processor.process_data(
        sample_frac=sample_frac
    )

    # Process Val
    val_processor = DataProcessor(mode="validation")
    # Ensure val uses same scaler
    val_processor.scaler_kin = train_processor.scaler_kin
    val_processor.scaler_sky = train_processor.scaler_sky
    val_processor.is_fitted = True

    X_kin_val, X_sky_val, y_val = val_processor.process_data(sample_frac=sample_frac)

    # Create Datasets
    train_dataset = GNSSDataset(X_kin_train, X_sky_train, y_train)
    val_dataset = GNSSDataset(X_kin_val, X_sky_val, y_val)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
