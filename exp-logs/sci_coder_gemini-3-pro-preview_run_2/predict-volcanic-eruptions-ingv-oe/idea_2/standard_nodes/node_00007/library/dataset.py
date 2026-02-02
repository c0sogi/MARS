import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.feature_engineering import SeismicFeatureEngineer
from library.utils import TargetScaler


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for Seismic Eruption Prediction.
    Provides (Spectrogram, Tabular Features, Target) tuples.
    """

    def __init__(
        self, metadata_path, features_path, mode="train", feature_engineer=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            features_path (str): Path to the cached tabular features Parquet file.
            mode (str): 'train', 'val', or 'test'.
            feature_engineer (SeismicFeatureEngineer): Instance for on-the-fly processing.
        """
        self.mode = mode
        self.feature_engineer = (
            feature_engineer if feature_engineer else SeismicFeatureEngineer()
        )

        # 1. Load Metadata and Features
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")

        self.meta_df = pd.read_csv(metadata_path)
        self.feats_df = pd.read_parquet(features_path)

        # Merge metadata with features on segment_id to have everything in one place
        # The metadata has 'file_path', features has stats.
        self.data = pd.merge(
            self.meta_df,
            self.feats_df,
            on="segment_id",
            how="inner",
            suffixes=("", "_dup"),
        )

        # Drop duplicate columns if any (like time_to_eruption from both sources)
        self.data = self.data.loc[:, ~self.data.columns.str.endswith("_dup")]

        # Identify feature columns (exclude metadata cols)
        exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
        self.feature_cols = [
            c
            for c in self.feats_df.columns
            if c not in exclude_cols and not c.endswith("_dup")
        ]

        # 2. Load Scalers
        # Spectrogram Scaler
        if os.path.exists(Config.SPEC_MEAN_PATH) and os.path.exists(
            Config.SPEC_STD_PATH
        ):
            self.spec_mean = np.load(Config.SPEC_MEAN_PATH).astype(np.float32)
            self.spec_std = np.load(Config.SPEC_STD_PATH).astype(np.float32)
            # Avoid division by zero
            if self.spec_std == 0:
                self.spec_std = 1.0
        else:
            # Fallback if not computed (should not happen via get_dataloaders)
            self.spec_mean = 0.0
            self.spec_std = 1.0

        # Tabular Scaler
        if os.path.exists(Config.STATS_SCALER_MEAN_PATH) and os.path.exists(
            Config.STATS_SCALER_SCALE_PATH
        ):
            self.tab_mean = np.load(Config.STATS_SCALER_MEAN_PATH).astype(np.float32)
            self.tab_scale = np.load(Config.STATS_SCALER_SCALE_PATH).astype(np.float32)
        else:
            self.tab_mean = np.zeros(len(self.feature_cols), dtype=np.float32)
            self.tab_scale = np.ones(len(self.feature_cols), dtype=np.float32)

        # Target Scaler (only needed for train/val)
        self.target_scaler = TargetScaler()
        if self.mode != "test":
            if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
                Config.TARGET_STD_PATH
            ):
                self.target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
            else:
                # If not found, we assume it will be handled or raw values used (though get_dataloaders ensures it exists)
                pass

        # Augmentation Transforms
        if self.mode == "train":
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            )
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=Config.TIME_MASK_PARAM
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---------------------------------------------------------
        # 1. Spectrogram Branch (On-the-fly)
        # ---------------------------------------------------------
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read CSV efficiently
        # We use float32 to save memory/time, engine='c' is default
        try:
            df_sensor = pd.read_csv(file_path, dtype=np.float32)
        except Exception:
            # Fallback for missing files or read errors: return zeros
            df_sensor = pd.DataFrame(
                np.zeros((Config.SIGNAL_LENGTH, Config.NUM_SENSORS), dtype=np.float32)
            )

        # Compute Spectrogram
        # Shape: (10, n_mels, time)
        spec_tensor = self.feature_engineer.compute_spectrogram(df_sensor)

        # Normalize Spectrogram
        spec_tensor = (spec_tensor - self.spec_mean) / self.spec_std

        # Apply Augmentation (only in train mode)
        if self.mode == "train":
            spec_tensor = self.freq_mask(spec_tensor)
            spec_tensor = self.time_mask(spec_tensor)

        # ---------------------------------------------------------
        # 2. Tabular Branch (Pre-computed)
        # ---------------------------------------------------------
        # Extract features as numpy array
        tab_features = row[self.feature_cols].values.astype(np.float32)

        # Normalize Tabular Features
        tab_features = (tab_features - self.tab_mean) / self.tab_scale
        tab_tensor = torch.tensor(tab_features, dtype=torch.float32)

        # ---------------------------------------------------------
        # 3. Target (if applicable)
        # ---------------------------------------------------------
        if self.mode != "test":
            target_val = row["time_to_eruption"]
            # Scale target
            target_scaled = self.target_scaler.transform(target_val)
            target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
            return spec_tensor, tab_tensor, target_tensor
        else:
            # Return segment_id for submission mapping if needed, but standard is usually just inputs
            # The training loop expects inputs. We can return segment_id as metadata if needed,
            # but for standard PyTorch inference loops, we usually just return data.
            # However, to map predictions back to IDs, the caller typically iterates the dataset/loader
            # which matches the order of self.data.
            return spec_tensor, tab_tensor


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and dataloaders.
    Handles all preprocessing, caching, and scaler fitting steps.

    Args:
        load_cached_data (bool): Whether to use existing cache files.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    print("Initializing Data Pipeline...")

    fe = SeismicFeatureEngineer()

    # ---------------------------------------------------------
    # 1. Preprocessing & Caching
    # ---------------------------------------------------------

    # A. Tabular Features
    # Compute for all splits
    train_feats_path = Config.TRAIN_FEATURES_PATH
    val_feats_path = Config.VAL_FEATURES_PATH
    test_feats_path = Config.TEST_FEATURES_PATH

    fe.cache_tabular_features(Config.TRAIN_METADATA, train_feats_path, load_cached_data)
    fe.cache_tabular_features(Config.VAL_METADATA, val_feats_path, load_cached_data)
    fe.cache_tabular_features(Config.TEST_METADATA, test_feats_path, load_cached_data)

    # B. Spectrogram Statistics (Global Normalization)
    # Computed on training data only
    fe.compute_and_cache_spec_stats(Config.TRAIN_METADATA, load_cached_data)

    # C. Tabular Scaler
    # Fitted on training features only
    fe.fit_and_cache_tabular_scaler(train_feats_path, load_cached_data)

    # D. Target Scaler
    # Fitted on training targets only
    target_scaler = TargetScaler()

    # Check if target scaler is cached
    if (
        load_cached_data
        and os.path.exists(Config.TARGET_MEAN_PATH)
        and os.path.exists(Config.TARGET_STD_PATH)
    ):
        print("Loading cached Target Scaler...")
        target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
    else:
        print("Fitting Target Scaler...")
        # Load train metadata to get targets
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        targets = df_train["time_to_eruption"].values
        target_scaler.fit(targets)
        target_scaler.save(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
        print(
            f"Target Scaler saved. Mean: {target_scaler.mean:.4f}, Std: {target_scaler.std:.4f}"
        )

    # ---------------------------------------------------------
    # 2. Create Datasets
    # ---------------------------------------------------------
    train_dataset = SeismicDataset(
        metadata_path=Config.TRAIN_METADATA,
        features_path=train_feats_path,
        mode="train",
        feature_engineer=fe,
    )

    val_dataset = SeismicDataset(
        metadata_path=Config.VAL_METADATA,
        features_path=val_feats_path,
        mode="val",
        feature_engineer=fe,
    )

    test_dataset = SeismicDataset(
        metadata_path=Config.TEST_METADATA,
        features_path=test_feats_path,
        mode="test",
        feature_engineer=fe,
    )

    # ---------------------------------------------------------
    # 3. Create DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
