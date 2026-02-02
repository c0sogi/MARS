import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import ecef_to_lla


class GnssDataset(Dataset):
    def __init__(self, metadata_df, split_type="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing drive_id, phone_name, timestamps, and paths.
            split_type (str): 'train', 'val', or 'test'. Used to determine if targets exist.
            load_cached_data (bool): Whether to load processed data from cache.
        """
        self.metadata_df = metadata_df
        self.split_type = split_type
        self.load_cached_data = load_cached_data

        # Containers for data
        self.sat_features_list = []
        self.global_features_list = []
        self.masks_list = []
        self.targets_list = []
        self.wls_lla_list = []
        self.trip_ids_list = []
        self.timestamps_list = []

        # Normalization stats
        self.sat_mean = None
        self.sat_std = None
        self.global_mean = None
        self.global_std = None

        self._load_data()

    def _load_data(self):
        # Group by drive and phone to process each trip
        # For test set, tripId is unique, but we can group by drive_id and phone_name derived in metadata
        grouped = self.metadata_df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group_df in grouped:
            # Sort by timestamp to ensure temporal order
            group_df = group_df.sort_values("UnixTimeMillis")

            # Process or load cache for this specific drive-phone pair
            data = self._process_drive(drive_id, phone_name, group_df)

            if data is None:
                continue

            self.sat_features_list.append(data["sat_features"])
            self.global_features_list.append(data["global_features"])
            self.masks_list.append(data["masks"])
            self.wls_lla_list.append(data["wls_lla"])
            self.timestamps_list.append(data["timestamps"])

            # targets might be None for test
            if data["targets"] is not None:
                self.targets_list.append(data["targets"])
            else:
                # Create dummy targets for test to keep structure consistent
                self.targets_list.append(
                    np.zeros((len(data["timestamps"]), 2), dtype=np.float32)
                )

            # Create trip_ids list matching the length
            trip_id_str = f"{drive_id}-{phone_name}"
            self.trip_ids_list.extend([trip_id_str] * len(data["timestamps"]))

        # Concatenate all lists
        if len(self.sat_features_list) > 0:
            self.sat_features = np.concatenate(self.sat_features_list, axis=0)
            self.global_features = np.concatenate(self.global_features_list, axis=0)
            self.masks = np.concatenate(self.masks_list, axis=0)
            self.targets = np.concatenate(self.targets_list, axis=0)
            self.wls_lla = np.concatenate(self.wls_lla_list, axis=0)
            self.timestamps = np.concatenate(self.timestamps_list, axis=0)
            self.trip_ids = np.array(self.trip_ids_list)
        else:
            # Handle empty dataset case
            self.sat_features = np.empty(
                (0, Config.MAX_SATELLITES, len(self._get_sat_feature_names()))
            )
            self.global_features = np.empty((0, len(Config.GLOBAL_FEATURES)))
            self.masks = np.empty((0, Config.MAX_SATELLITES))
            self.targets = np.empty((0, 2))
            self.wls_lla = np.empty((0, 3))
            self.timestamps = np.empty((0,))
            self.trip_ids = np.empty((0,))

    def _get_cache_path(self, drive_id, phone_name):
        filename = f"{drive_id}_{phone_name}_{self.split_type}.npz"
        return os.path.join(Config.WORKING_DIR, filename)

    def _process_drive(self, drive_id, phone_name, df_meta_drive):
        cache_path = self._get_cache_path(drive_id, phone_name)

        # 1. Try Loading Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                # Check if loaded data matches expected length (simple integrity check)
                if len(loaded["timestamps"]) == len(df_meta_drive):
                    return {
                        "sat_features": loaded["sat_features"],
                        "global_features": loaded["global_features"],
                        "masks": loaded["masks"],
                        "targets": loaded["targets"] if "targets" in loaded else None,
                        "wls_lla": loaded["wls_lla"],
                        "timestamps": loaded["timestamps"],
                    }
            except Exception as e:
                print(f"Failed to load cache for {drive_id}-{phone_name}: {e}")

        # 2. Compute from Scratch
        # Load GNSS Data
        # Assuming the path in metadata is correct relative to INPUT_DIR
        gnss_rel_path = df_meta_drive.iloc[0]["gnss_path"]
        gnss_full_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_full_path):
            print(f"GNSS file not found: {gnss_full_path}")
            return None

        gnss_df = pd.read_csv(gnss_full_path, usecols=Config.GNSS_COLS)

        # Filter GNSS data to relevant timestamps
        # Note: GNSS utcTimeMillis vs Meta UnixTimeMillis. Assuming they match.
        target_timestamps = df_meta_drive["UnixTimeMillis"].values
        gnss_df = gnss_df[gnss_df["utcTimeMillis"].isin(target_timestamps)]

        if gnss_df.empty:
            print(f"No matching GNSS data for {drive_id}-{phone_name}")
            return None

        # Group by timestamp
        grouped_gnss = gnss_df.groupby("utcTimeMillis")

        # Pre-allocate arrays
        num_samples = len(target_timestamps)
        # Sat features: Cn0, El, sinAz, cosAz, PrUnc, Constellation(7) -> 12 features
        # Features: Cn0(1), El(1), SinAz(1), CosAz(1), PrUnc(1), Constellation(7) = 12
        num_sat_feats = 1 + 1 + 1 + 1 + 1 + 7

        sat_features = np.zeros(
            (num_samples, Config.MAX_SATELLITES, num_sat_feats), dtype=np.float32
        )
        masks = np.zeros((num_samples, Config.MAX_SATELLITES), dtype=np.float32)
        global_features = np.zeros(
            (num_samples, len(Config.GLOBAL_FEATURES)), dtype=np.float32
        )
        wls_lla = np.zeros((num_samples, 3), dtype=np.float32)
        targets = (
            np.zeros((num_samples, 2), dtype=np.float32)
            if self.split_type != "test"
            else None
        )

        # Create a map from timestamp to index for O(1) access
        ts_to_idx = {ts: i for i, ts in enumerate(target_timestamps)}

        # Iterate over GNSS groups
        for ts, group in grouped_gnss:
            if ts not in ts_to_idx:
                continue
            idx = ts_to_idx[ts]

            # --- Global Features ---
            # SatCount
            sat_count = len(group)

            # WLS Position (ECEF) -> LLA
            # Take the first row's WLS position (they are identical for the same epoch)
            wls_x = group.iloc[0]["WlsPositionXEcefMeters"]
            wls_y = group.iloc[0]["WlsPositionYEcefMeters"]
            wls_z = group.iloc[0]["WlsPositionZEcefMeters"]

            # Handle NaNs in WLS
            if np.isnan(wls_x):
                # Fallback or keep as NaN (will likely cause issues later, but handle gracefully)
                lat_wls, lon_wls, alt_wls = 0.0, 0.0, 0.0
            else:
                lat_wls, lon_wls, alt_wls = ecef_to_lla(wls_x, wls_y, wls_z)

            wls_lla[idx] = [lat_wls, lon_wls, alt_wls]

            # Global Feats: [SatCount, WlsAlt]
            global_features[idx] = [sat_count, alt_wls]

            # --- Satellite Features ---
            # Truncate if too many satellites
            if sat_count > Config.MAX_SATELLITES:
                # Sort by Cn0DbHz to keep best satellites
                group = group.sort_values("Cn0DbHz", ascending=False).head(
                    Config.MAX_SATELLITES
                )
                sat_count = Config.MAX_SATELLITES

            # Extract columns
            cn0 = group["Cn0DbHz"].values
            el = group["SvElevationDegrees"].values
            az = np.radians(group["SvAzimuthDegrees"].values)
            pr_unc = group["RawPseudorangeUncertaintyMeters"].values
            const_type = group["ConstellationType"].values.astype(int)

            # Feature Construction
            # 1. Cn0
            sat_features[idx, :sat_count, 0] = cn0
            # 2. Elevation
            sat_features[idx, :sat_count, 1] = el
            # 3. Sin Az
            sat_features[idx, :sat_count, 2] = np.sin(az)
            # 4. Cos Az
            sat_features[idx, :sat_count, 3] = np.cos(az)
            # 5. Pr Unc
            sat_features[idx, :sat_count, 4] = pr_unc
            # 6. Constellation One-Hot (Indices 5 to 11)
            # Map constellation types 0-6 to indices.
            const_type = np.clip(const_type, 0, 6)
            # Create one-hot
            one_hot = np.eye(7)[const_type]
            sat_features[idx, :sat_count, 5:] = one_hot

            # Mask (1 for valid, 0 for padding)
            masks[idx, :sat_count] = 1.0

            # --- Targets ---
            if self.split_type != "test":
                # Get GT from metadata
                # Since df_meta_drive is sorted by timestamp and matches target_timestamps order:
                gt_lat = df_meta_drive.iloc[idx]["LatitudeDegrees"]
                gt_lon = df_meta_drive.iloc[idx]["LongitudeDegrees"]

                # Calculate Residuals
                # Target = GT - WLS
                res_lat = gt_lat - lat_wls
                res_lon = gt_lon - lon_wls
                targets[idx] = [res_lat, res_lon]

        # 3. Save to Cache
        save_dict = {
            "sat_features": sat_features,
            "global_features": global_features,
            "masks": masks,
            "wls_lla": wls_lla,
            "timestamps": target_timestamps,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)

        return save_dict

    def _get_sat_feature_names(self):
        # Helper to define feature dimension
        return ["Cn0", "El", "SinAz", "CosAz", "PrUnc"] + [
            f"Const_{i}" for i in range(7)
        ]

    def fit_normalization(self):
        """
        Computes mean and std for features based on current data (Train set).
        """
        # Flatten sat features for stats: (N * MAX_SAT, Feats)
        # Only consider valid satellites (mask == 1)
        valid_mask = self.masks.astype(bool)
        flat_sat = self.sat_features[valid_mask]

        self.sat_mean = np.mean(flat_sat, axis=0)
        self.sat_std = np.std(flat_sat, axis=0)
        # Avoid div by zero
        self.sat_std[self.sat_std < 1e-6] = 1.0

        # Global features
        self.global_mean = np.mean(self.global_features, axis=0)
        self.global_std = np.std(self.global_features, axis=0)
        self.global_std[self.global_std < 1e-6] = 1.0

    def apply_normalization(self, sat_mean, sat_std, global_mean, global_std):
        """
        Applies provided normalization stats to the dataset.
        """
        self.sat_mean = sat_mean
        self.sat_std = sat_std
        self.global_mean = global_mean
        self.global_std = global_std

        # Normalize Satellite Features (broadcast over N and MAX_SAT)
        # Don't normalize One-Hot encoded columns (indices 5 to 11)
        # Normalize first 5 features
        self.sat_features[:, :, :5] = (
            self.sat_features[:, :, :5] - self.sat_mean[:5]
        ) / self.sat_std[:5]

        # Normalize Global Features
        self.global_features = (
            self.global_features - self.global_mean
        ) / self.global_std

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sat_features[idx], dtype=torch.float32),
            torch.tensor(self.global_features[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.wls_lla[idx], dtype=torch.float32),  # For reconstruction
        )


def get_dataloaders():
    """
    Creates train, val, and test dataloaders.
    Handles metadata loading, dataset creation, and normalization.
    """
    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Sample small subset
    if Config.DEBUG:
        train_meta = train_meta.iloc[: Config.DEBUG_SIZE]
        val_meta = val_meta.iloc[: Config.DEBUG_SIZE]
        test_meta = test_meta.iloc[: Config.DEBUG_SIZE]
        print(f"DEBUG MODE: Reduced train size to {len(train_meta)}")

    # Create Datasets
    print("Initializing Train Dataset...")
    train_dataset = GnssDataset(
        train_meta, split_type="train", load_cached_data=Config.CACHE_DATA
    )

    print("Initializing Validation Dataset...")
    val_dataset = GnssDataset(
        val_meta, split_type="val", load_cached_data=Config.CACHE_DATA
    )

    print("Initializing Test Dataset...")
    test_dataset = GnssDataset(
        test_meta, split_type="test", load_cached_data=Config.CACHE_DATA
    )

    # Fit Normalization on Train
    print("Fitting Normalization on Train Data...")
    train_dataset.fit_normalization()

    # Apply to Val and Test
    print("Applying Normalization...")
    val_dataset.apply_normalization(
        train_dataset.sat_mean,
        train_dataset.sat_std,
        train_dataset.global_mean,
        train_dataset.global_std,
    )
    test_dataset.apply_normalization(
        train_dataset.sat_mean,
        train_dataset.sat_std,
        train_dataset.global_mean,
        train_dataset.global_std,
    )

    # Create Loaders
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
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
