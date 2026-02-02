import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict

from library.config import Config
from library.utils import get_logger, WGS84


class GNSSPreprocessor:
    """
    Handles data loading, feature engineering, windowing, and caching for the GNSS positioning task.
    """

    def __init__(self):
        self.logger = get_logger("data_processing")
        self.kinematic_scaler = StandardScaler()
        self.sky_scaler = StandardScaler()
        self.is_fitted = False

    def _save_scalers(self):
        """Saves scaler parameters to a JSON file to avoid pickle."""
        if not self.is_fitted:
            self.logger.warning("Scalers not fitted, skipping save.")
            return

        scaler_data = {
            "kinematic_mean": self.kinematic_scaler.mean_.tolist(),
            "kinematic_scale": self.kinematic_scaler.scale_.tolist(),
            "sky_mean": self.sky_scaler.mean_.tolist(),
            "sky_scale": self.sky_scaler.scale_.tolist(),
        }
        with open(Config.SCALER_PATH, "w") as f:
            json.dump(scaler_data, f)
        self.logger.info(f"Scalers saved to {Config.SCALER_PATH}")

    def _load_scalers(self):
        """Loads scaler parameters from a JSON file."""
        if not os.path.exists(Config.SCALER_PATH):
            self.logger.warning(f"Scaler file not found at {Config.SCALER_PATH}")
            return False

        with open(Config.SCALER_PATH, "r") as f:
            data = json.load(f)

        self.kinematic_scaler.mean_ = np.array(data["kinematic_mean"])
        self.kinematic_scaler.scale_ = np.array(data["kinematic_scale"])
        self.kinematic_scaler.var_ = (
            self.kinematic_scaler.scale_**2
        )  # Approximate reconstruction
        self.kinematic_scaler.n_samples_seen_ = (
            1000  # Dummy value to satisfy sklearn check
        )

        self.sky_scaler.mean_ = np.array(data["sky_mean"])
        self.sky_scaler.scale_ = np.array(data["sky_scale"])
        self.sky_scaler.var_ = self.sky_scaler.scale_**2
        self.sky_scaler.n_samples_seen_ = 1000

        self.is_fitted = True
        self.logger.info("Scalers loaded successfully.")
        return True

    def _load_raw_data(self, metadata_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Loads raw GNSS data for all trips in the metadata.
        Returns a dictionary mapping tripId to the raw GNSS DataFrame.
        """
        trip_data = {}
        unique_trips = metadata_df["tripId"].unique()

        # Pre-scan paths to avoid repeated lookups
        # We assume metadata has 'gnss_path' column

        self.logger.info(f"Loading raw GNSS data for {len(unique_trips)} trips...")

        for trip_id in tqdm(unique_trips, desc="Loading Raw Data"):
            trip_meta = metadata_df[metadata_df["tripId"] == trip_id].iloc[0]
            gnss_rel_path = trip_meta["gnss_path"]
            gnss_full_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

            if os.path.exists(gnss_full_path):
                # Load only necessary columns to save memory
                try:
                    df = pd.read_csv(gnss_full_path, usecols=Config.GNSS_COLS_TO_LOAD)
                    # Sort by time just in case
                    df = df.sort_values("utcTimeMillis").reset_index(drop=True)
                    trip_data[trip_id] = df
                except Exception as e:
                    self.logger.error(f"Failed to load {gnss_full_path}: {e}")
            else:
                self.logger.warning(f"File not found: {gnss_full_path}")

        return trip_data

    def _process_trip(
        self,
        trip_id: str,
        gnss_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        mode: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Processes a single trip: merges data, engineers features, and creates windows.
        """
        # 1. Convert WLS ECEF to LLA
        wls_lat, wls_lon, wls_alt = WGS84.ecef_to_lla(
            gnss_df["WlsPositionXEcefMeters"].values,
            gnss_df["WlsPositionYEcefMeters"].values,
            gnss_df["WlsPositionZEcefMeters"].values,
        )
        gnss_df["WlsLat"] = wls_lat
        gnss_df["WlsLon"] = wls_lon
        gnss_df["WlsAlt"] = wls_alt

        # 2. Aggregate GNSS data by epoch (utcTimeMillis)
        # We need to handle multiple satellites per epoch.
        # For kinematic features (center of window), we use the WLS position (which is per epoch).
        # For signal metrics, we average.
        # For sky features, we aggregate over the window later.

        # Group by epoch to get a single row per timestamp for the sequence
        # We take the FIRST WLS position (they are identical for the same epoch)
        # We take MEAN of Cn0 and Uncertainty
        # We also need to keep track of satellite stats for the Sky Context

        # Define aggregations
        agg_dict = {
            "WlsLat": "first",
            "WlsLon": "first",
            "WlsAlt": "first",
            "Cn0DbHz": ["mean", "std"],
            "RawPseudorangeUncertaintyMeters": "mean",
            "SvElevationDegrees": ["mean", "std"],
            "SvAzimuthDegrees": ["mean", "std"],
        }

        epoch_df = gnss_df.groupby("utcTimeMillis").agg(agg_dict)
        # Flatten columns
        epoch_df.columns = ["_".join(col).strip() for col in epoch_df.columns.values]
        epoch_df.reset_index(inplace=True)

        # Rename for clarity
        epoch_df.rename(
            columns={
                "WlsLat_first": "lat",
                "WlsLon_first": "lon",
                "WlsAlt_first": "alt",
                "Cn0DbHz_mean": "mean_cn0",
                "Cn0DbHz_std": "std_cn0",
                "RawPseudorangeUncertaintyMeters_mean": "mean_unc",
                "SvElevationDegrees_mean": "mean_el",
                "SvElevationDegrees_std": "std_el",
                "SvAzimuthDegrees_mean": "mean_az",
                "SvAzimuthDegrees_std": "std_az",
            },
            inplace=True,
        )

        # Fill NaNs in std columns (single satellite epochs)
        epoch_df.fillna(0, inplace=True)

        # 3. Merge with Targets (Ground Truth or Sample Submission)
        # We use 'inner' join to ensure we only process timestamps where we need predictions/have labels
        # Note: Targets have UnixTimeMillis, GNSS has utcTimeMillis.
        merged_df = (
            pd.merge(
                targets_df,
                epoch_df,
                left_on="UnixTimeMillis",
                right_on="utcTimeMillis",
                how="inner",
            )
            .sort_values("UnixTimeMillis")
            .reset_index(drop=True)
        )

        if merged_df.empty:
            return None, None, None, None

        # 4. Compute Targets (Residuals) - Only for train/val
        y_data = np.zeros((len(merged_df), 2))
        if mode in ["train", "val"]:
            # Calculate d_north, d_east from WLS to Ground Truth
            # GT - WLS
            d_lat = merged_df["LatitudeDegrees"] - merged_df["lat"]
            d_lon = merged_df["LongitudeDegrees"] - merged_df["lon"]

            d_north, d_east = WGS84.lat_lon_to_meters_flat(
                d_lat.values, d_lon.values, merged_df["lat"].values
            )
            y_data[:, 0] = d_east
            y_data[:, 1] = d_north

        # 5. Create Windows
        # We need to construct windows centered at each row in merged_df.
        # Since merged_df might have gaps compared to the raw stream, we should ideally look back at the full raw stream.
        # However, for simplicity and robustness given the dense sampling (1Hz), we often use the merged stream itself
        # if it's continuous.
        # To be more robust, we will use the index in the merged dataframe, assuming 1Hz continuity.
        # If gaps exist (>1.5s), we might introduce noise, but SKF-Net is robust to this.

        # Pad the dataframe to handle edges
        pad_size = Config.WINDOW_SIZE // 2

        # Features for Kinematic Stream (Sequence)
        # We need: rel_lat, rel_lon, rel_alt, vel_lat, vel_lon, vel_alt, cn0, unc
        # We calculate these dynamically inside the window generation to ensure "relative centering"

        # Convert absolute lat/lon to metric representation relative to the first point of trip for velocity calc
        # Or simply use differences.

        # Let's extract numpy arrays for speed
        lats = merged_df["lat"].values
        lons = merged_df["lon"].values
        alts = merged_df["alt"].values
        cn0s = merged_df["mean_cn0"].values
        uncs = merged_df["mean_unc"].values

        # Pre-calculate velocities (simple diff, padded)
        # vel[i] = pos[i] - pos[i-1]
        # We'll compute velocities relative to meters.
        # Approximate conversion for velocity calculation
        d_lat_global = np.diff(lats, prepend=lats[0])
        d_lon_global = np.diff(lons, prepend=lons[0])
        d_alt_global = np.diff(alts, prepend=alts[0])

        v_north, v_east = WGS84.lat_lon_to_meters_flat(d_lat_global, d_lon_global, lats)

        # Arrays for windowing
        # Shape: (N, Features)
        # Features: [lat, lon, alt, v_east, v_north, v_alt, cn0, unc]
        # Note: lat/lon here are absolute, we will make them relative in the loop
        seq_data_source = np.stack(
            [lats, lons, alts, v_east, v_north, d_alt_global, cn0s, uncs], axis=1
        )

        # Sky features source
        # [mean_el, std_el, mean_az, std_az, mean_cn0, std_cn0, mean_unc]
        sky_data_source = merged_df[
            [
                "mean_el",
                "std_el",
                "mean_az",
                "std_az",
                "mean_cn0",
                "std_cn0",
                "mean_unc",
            ]
        ].values

        num_samples = len(merged_df)

        # Output arrays
        X_seq_trip = np.zeros(
            (num_samples, Config.WINDOW_SIZE, 8)
        )  # 8 kinematic features
        X_sky_trip = np.zeros((num_samples, 7))  # 7 sky features

        # Padding for edges
        # We replicate the first/last elements
        seq_padded = np.pad(
            seq_data_source, ((pad_size, pad_size), (0, 0)), mode="edge"
        )
        sky_padded = np.pad(
            sky_data_source, ((pad_size, pad_size), (0, 0)), mode="edge"
        )

        for i in range(num_samples):
            # Window indices in padded array
            start = i
            end = i + Config.WINDOW_SIZE

            # Extract window
            window = seq_padded[
                start:end
            ].copy()  # Copy to avoid modifying padded array

            # Center of the window is at index pad_size within the window
            center_idx = pad_size
            center_lat = window[center_idx, 0]
            center_lon = window[center_idx, 1]
            center_alt = window[center_idx, 2]

            # 1. Relative Coordinates (Meters)
            # lat diff
            d_lat = window[:, 0] - center_lat
            d_lon = window[:, 1] - center_lon
            d_n, d_e = WGS84.lat_lon_to_meters_flat(d_lat, d_lon, center_lat)

            # alt diff
            d_h = window[:, 2] - center_alt

            # Update window features
            # [rel_lat_m (North), rel_lon_m (East), rel_alt_m, v_east, v_north, v_alt, cn0, unc]
            # Note: We map d_n -> rel_lat_m, d_e -> rel_lon_m based on config names usually,
            # but standard convention is Lat~North, Lon~East.

            # Construct final feature vector for this window
            # Features: 0:d_n, 1:d_e, 2:d_h, 3:v_e, 4:v_n, 5:v_h, 6:cn0, 7:unc
            window[:, 0] = d_n
            window[:, 1] = d_e
            window[:, 2] = d_h
            # Velocity and metrics remain as is

            X_seq_trip[i] = window

            # Sky Context: Aggregate over the window
            # We take the mean of the window's sky stats
            sky_window = sky_padded[start:end]
            X_sky_trip[i] = np.mean(sky_window, axis=0)

        # Metadata for reconstruction
        meta_trip = merged_df[["tripId", "UnixTimeMillis", "lat", "lon", "alt"]].copy()
        meta_trip["drive_id"] = trip_id.split("-")[0]  # approximate

        return X_seq_trip, X_sky_trip, y_data, meta_trip

    def _process_dataset(
        self,
        metadata_path: str,
        mode: str,
        load_cached: bool,
        cache_seq: str,
        cache_sky: str,
        cache_y: str,
        cache_meta: str,
    ):

        # 1. Check Cache
        if load_cached and os.path.exists(cache_seq) and os.path.exists(cache_meta):
            self.logger.info(f"Loading cached {mode} data...")
            try:
                X_seq = np.load(cache_seq)
                X_sky = np.load(cache_sky)
                y = np.load(cache_y) if mode != "test" else None
                meta = pd.read_parquet(cache_meta)

                # Load scalers if training data loaded from cache
                if mode == "train":
                    self._load_scalers()

                return X_seq, X_sky, y, meta
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute
        self.logger.info(f"Processing {mode} data from scratch...")
        df_meta = pd.read_csv(metadata_path)

        # Load raw GNSS data
        gnss_data_map = self._load_raw_data(df_meta)

        X_seq_list = []
        X_sky_list = []
        y_list = []
        meta_list = []

        unique_trips = df_meta["tripId"].unique()

        for trip_id in tqdm(unique_trips, desc=f"Processing {mode} trips"):
            if trip_id not in gnss_data_map:
                continue

            gnss_df = gnss_data_map[trip_id]
            targets_df = df_meta[df_meta["tripId"] == trip_id]

            x_seq, x_sky, y_trip, meta_trip = self._process_trip(
                trip_id, gnss_df, targets_df, mode
            )

            if x_seq is not None:
                X_seq_list.append(x_seq)
                X_sky_list.append(x_sky)
                if mode != "test":
                    y_list.append(y_trip)
                meta_list.append(meta_trip)

        # Concatenate
        X_seq = np.concatenate(X_seq_list, axis=0)
        X_sky = np.concatenate(X_sky_list, axis=0)
        y = np.concatenate(y_list, axis=0) if mode != "test" else np.array([])
        meta = pd.concat(meta_list, ignore_index=True)

        # 3. Scaling
        # Shape of X_seq: (N, W, F). Reshape to (N*W, F) for scaling, then reshape back
        N, W, F = X_seq.shape
        X_seq_flat = X_seq.reshape(-1, F)

        if mode == "train":
            self.logger.info("Fitting scalers on training data...")
            self.kinematic_scaler.fit(X_seq_flat)
            self.sky_scaler.fit(X_sky)
            self._save_scalers()
            self.is_fitted = True
        elif not self.is_fitted:
            # Try to load if not fitted (e.g. running inference only)
            if not self._load_scalers():
                raise RuntimeError(
                    "Scalers not fitted and cache not found. Run training first."
                )

        # Transform
        X_seq_flat = self.kinematic_scaler.transform(X_seq_flat)
        X_seq = X_seq_flat.reshape(N, W, F)
        X_sky = self.sky_scaler.transform(X_sky)

        # 4. Save Cache
        self.logger.info(f"Saving {mode} data to cache...")
        np.save(cache_seq, X_seq)
        np.save(cache_sky, X_sky)
        if mode != "test":
            np.save(cache_y, y)
        meta.to_parquet(cache_meta)

        return X_seq, X_sky, y, meta

    def process_train(self, load_cached_data: bool = True):
        return self._process_dataset(
            Config.TRAIN_METADATA_PATH,
            "train",
            load_cached_data,
            Config.CACHE_TRAIN_X_SEQ,
            Config.CACHE_TRAIN_X_SKY,
            Config.CACHE_TRAIN_Y,
            Config.CACHE_TRAIN_META,
        )

    def process_val(self, load_cached_data: bool = True):
        return self._process_dataset(
            Config.VAL_METADATA_PATH,
            "val",
            load_cached_data,
            Config.CACHE_VAL_X_SEQ,
            Config.CACHE_VAL_X_SKY,
            Config.CACHE_VAL_Y,
            Config.CACHE_VAL_META,
        )

    def process_test(self, load_cached_data: bool = True):
        return self._process_dataset(
            Config.TEST_METADATA_PATH,
            "test",
            load_cached_data,
            Config.CACHE_TEST_X_SEQ,
            Config.CACHE_TEST_X_SKY,
            None,
            Config.CACHE_TEST_META,
        )
