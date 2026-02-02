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
        self.features_list = []
        self.targets_list = []
        self.wls_lla_list = []
        self.trip_ids_list = []
        self.timestamps_list = []

        # Normalization stats
        self.feat_mean = None
        self.feat_std = None

        self._load_data()

    def _load_data(self):
        # Group by drive and phone to process each trip
        grouped = self.metadata_df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group_df in grouped:
            # Sort by timestamp to ensure temporal order
            group_df = group_df.sort_values("UnixTimeMillis")

            # Process or load cache for this specific drive-phone pair
            data = self._process_drive(drive_id, phone_name, group_df)

            if data is None:
                continue

            self.features_list.append(data["features"])
            self.wls_lla_list.append(data["wls_lla"])
            self.timestamps_list.append(data["timestamps"])

            # targets might be None for test
            if data["targets"] is not None:
                self.targets_list.append(data["targets"])
            else:
                # Create dummy targets for test
                self.targets_list.append(
                    np.zeros((len(data["timestamps"]), 2), dtype=np.float32)
                )

            # Create trip_ids list matching the length
            trip_id_str = f"{drive_id}-{phone_name}"
            self.trip_ids_list.extend([trip_id_str] * len(data["timestamps"]))

        # Concatenate all lists
        if len(self.features_list) > 0:
            self.features = np.concatenate(self.features_list, axis=0)
            self.targets = np.concatenate(self.targets_list, axis=0)
            self.wls_lla = np.concatenate(self.wls_lla_list, axis=0)
            self.timestamps = np.concatenate(self.timestamps_list, axis=0)
            self.trip_ids = np.array(self.trip_ids_list)
        else:
            # Handle empty dataset case
            self.features = np.empty((0, Config.TCN_INPUT_DIM))
            self.targets = np.empty((0, 2))
            self.wls_lla = np.empty((0, 3))
            self.timestamps = np.empty((0,))
            self.trip_ids = np.empty((0,))

    def _get_cache_path(self, drive_id, phone_name):
        # Changed suffix to avoid conflict with previous cache
        filename = f"{drive_id}_{phone_name}_{self.split_type}_agg.npz"
        return os.path.join(Config.WORKING_DIR, filename)

    def _process_drive(self, drive_id, phone_name, df_meta_drive):
        cache_path = self._get_cache_path(drive_id, phone_name)

        # 1. Try Loading Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                if len(loaded["timestamps"]) == len(df_meta_drive):
                    return {
                        "features": loaded["features"],
                        "targets": loaded["targets"] if "targets" in loaded else None,
                        "wls_lla": loaded["wls_lla"],
                        "timestamps": loaded["timestamps"],
                    }
            except Exception as e:
                print(f"Failed to load cache for {drive_id}-{phone_name}: {e}")

        # 2. Compute from Scratch
        gnss_rel_path = df_meta_drive.iloc[0]["gnss_path"]
        gnss_full_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_full_path):
            print(f"GNSS file not found: {gnss_full_path}")
            return None

        gnss_df = pd.read_csv(gnss_full_path, usecols=Config.GNSS_COLS)
        gnss_df = gnss_df.fillna(0)

        target_timestamps = df_meta_drive["UnixTimeMillis"].values
        gnss_df = gnss_df[gnss_df["utcTimeMillis"].isin(target_timestamps)]

        if gnss_df.empty:
            print(f"No matching GNSS data for {drive_id}-{phone_name}")
            return None

        grouped_gnss = gnss_df.groupby("utcTimeMillis")

        num_samples = len(target_timestamps)
        # Features: Cn0(Mean, Max, Std), El(Mean, Std), PrUnc(Mean), SatCount, WlsAlt -> 8 features
        features = np.zeros((num_samples, 8), dtype=np.float32)
        wls_lla = np.zeros((num_samples, 3), dtype=np.float32)
        targets = (
            np.zeros((num_samples, 2), dtype=np.float32)
            if self.split_type != "test"
            else None
        )

        ts_to_idx = {ts: i for i, ts in enumerate(target_timestamps)}

        for ts, group in grouped_gnss:
            if ts not in ts_to_idx:
                continue
            idx = ts_to_idx[ts]

            # SatCount
            sat_count = len(group)

            # WLS Position (ECEF) -> LLA
            wls_x = group.iloc[0]["WlsPositionXEcefMeters"]
            wls_y = group.iloc[0]["WlsPositionYEcefMeters"]
            wls_z = group.iloc[0]["WlsPositionZEcefMeters"]

            if np.isnan(wls_x):
                lat_wls, lon_wls, alt_wls = 0.0, 0.0, 0.0
            else:
                lat_wls, lon_wls, alt_wls = ecef_to_lla(wls_x, wls_y, wls_z)

            wls_lla[idx] = [lat_wls, lon_wls, alt_wls]

            # Aggregations (Cite solution_lesson_node_00014)
            cn0 = group["Cn0DbHz"].values
            el = group["SvElevationDegrees"].values
            pr_unc = group["RawPseudorangeUncertaintyMeters"].values

            # 0: Cn0 Mean
            features[idx, 0] = np.mean(cn0)
            # 1: Cn0 Max
            features[idx, 1] = np.max(cn0)
            # 2: Cn0 Std
            features[idx, 2] = np.std(cn0) if len(cn0) > 1 else 0.0
            # 3: El Mean
            features[idx, 3] = np.mean(el)
            # 4: El Std
            features[idx, 4] = np.std(el) if len(el) > 1 else 0.0
            # 5: PrUnc Mean
            features[idx, 5] = np.mean(pr_unc)
            # 6: SatCount
            features[idx, 6] = sat_count
            # 7: WlsAlt
            features[idx, 7] = alt_wls

            if self.split_type != "test":
                gt_lat = df_meta_drive.iloc[idx]["LatitudeDegrees"]
                gt_lon = df_meta_drive.iloc[idx]["LongitudeDegrees"]
                targets[idx] = [gt_lat - lat_wls, gt_lon - lon_wls]

        save_dict = {
            "features": features,
            "wls_lla": wls_lla,
            "timestamps": target_timestamps,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)

        if "targets" not in save_dict:
            save_dict["targets"] = None

        return save_dict

    def fit_normalization(self):
        """
        Computes mean and std for features based on current data (Train set).
        """
        if len(self.features) == 0:
            return
        self.feat_mean = np.nanmean(self.features, axis=0)
        self.feat_std = np.nanstd(self.features, axis=0)
        self.feat_std[self.feat_std < 1e-6] = 1.0

    def apply_normalization(self, mean, std):
        """
        Applies provided normalization stats to the dataset.
        """
        self.feat_mean = mean
        self.feat_std = std
        self.features = (self.features - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.wls_lla[idx], dtype=torch.float32),
        )

    def _get_sat_feature_names(self):
        # Unused in new implementation
        pass


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
        train_dataset.feat_mean,
        train_dataset.feat_std,
    )
    test_dataset.apply_normalization(
        train_dataset.feat_mean,
        train_dataset.feat_std,
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
