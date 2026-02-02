import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import TargetScaler


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for Volcano Eruption Prediction.
    Combines on-the-fly spectrogram generation with pre-computed tabular features.
    """

    def __init__(self, metadata_df, tabular_df, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'segment_id', 'file_path', and 'time_to_eruption' (if train/val).
            tabular_df (pd.DataFrame): DataFrame containing pre-computed tabular features, indexed by 'segment_id'.
            mode (str): One of 'train', 'val', 'test'. Controls augmentation and target return.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.tabular_features = tabular_df.copy()

        # Ensure fast lookup for tabular features
        if "segment_id" in self.tabular_features.columns:
            self.tabular_features = self.tabular_features.set_index("segment_id")

        self.mode = mode
        self.feature_engineer = FeatureEngineer()

        # SpecAugment Transforms (Conservative masking as per Idea 8)
        # Time steps approx 235, Freq bins 128.
        # ~15% masking: Time ~35, Freq ~20.
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=35)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = row["segment_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Load Raw Sensor Data
        try:
            # Using float32 to save memory/time, though original is float64
            df_sensor = pd.read_csv(file_path, dtype=np.float32)
        except Exception as e:
            # Fallback for missing files (should be caught by metadata check, but for safety)
            # Create a dummy zero signal
            df_sensor = pd.DataFrame(
                np.zeros((Config.SIGNAL_LENGTH, Config.NUM_SENSORS), dtype=np.float32)
            )

        # 2. Process Signal (Fill NaNs with 0)
        signal = self.feature_engineer.preprocess_signal(df_sensor)

        # 3. Generate Spectrogram
        # Shape: (Channels, n_mels, Time)
        spectrogram = self.feature_engineer.get_spectrogram(signal)

        # 4. Apply Augmentation (Train only)
        if self.mode == "train":
            # Apply masking to each channel independently or same mask?
            # Usually applied per batch, but here per sample.
            # Torchaudio transforms handle (C, F, T) or (F, T).
            spectrogram = self.time_masking(spectrogram)
            spectrogram = self.freq_masking(spectrogram)

        # 5. Retrieve Tabular Features
        # Extract the row corresponding to segment_id
        if segment_id in self.tabular_features.index:
            tab_feats = self.tabular_features.loc[segment_id].values.astype(np.float32)
        else:
            # Fallback if feature missing (unlikely with correct pipeline)
            tab_feats = np.zeros(self.tabular_features.shape[1], dtype=np.float32)

        tab_tensor = torch.tensor(tab_feats)

        # 6. Prepare Output
        sample = {
            "spectrogram": spectrogram,
            "tabular": tab_tensor,
            "segment_id": segment_id,
        }

        # 7. Get Target (Train/Val only)
        if self.mode != "test":
            # Target is already scaled in the metadata passed to this class?
            # No, we pass the raw metadata and a Scaler is used externally or we scale here.
            # Strategy: The metadata df passed to __init__ should have the SCALED target.
            # This logic is handled in get_dataloaders.
            target = row["time_to_eruption"]
            sample["target"] = torch.tensor(target, dtype=torch.float32)

        return sample


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Val, and Test sets.
    Handles feature extraction, caching, and scaling.

    Args:
        load_cached_data (bool): Whether to load features from parquet cache.

    Returns:
        train_loader, val_loader, test_loader, target_scaler
    """
    print("Initializing Data Loaders...")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Subsample
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_meta = train_meta.head(Config.DEBUG_SAMPLE_SIZE)
        val_meta = val_meta.head(Config.DEBUG_SAMPLE_SIZE)
        test_meta = test_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Generate/Load Tabular Features
    fe = FeatureEngineer()

    train_feats = fe.generate_tabular_features(
        train_meta, Config.TRAIN_FEATURES_PATH, load_cached_data
    )
    val_feats = fe.generate_tabular_features(
        val_meta, Config.VAL_FEATURES_PATH, load_cached_data
    )
    test_feats = fe.generate_tabular_features(
        test_meta, Config.TEST_FEATURES_PATH, load_cached_data
    )

    # 3. Scale Tabular Features
    # We implement a manual StandardScaler to avoid pickle and ensure .npy persistence
    stats_mean_path = os.path.join(Config.WORKING_DIR, "stats_scaler_mean.npy")
    stats_scale_path = os.path.join(Config.WORKING_DIR, "stats_scaler_scale.npy")

    # Drop segment_id for scaling calculation
    feature_cols = [c for c in train_feats.columns if c != "segment_id"]

    if (
        load_cached_data
        and os.path.exists(stats_mean_path)
        and os.path.exists(stats_scale_path)
    ):
        print("Loading tabular scaler statistics...")
        mean_vec = np.load(stats_mean_path)
        scale_vec = np.load(stats_scale_path)
    else:
        print("Computing tabular scaler statistics from training data...")
        # Fill NaNs in features with 0 before computing stats
        X_train = train_feats[feature_cols].fillna(0).values
        mean_vec = np.mean(X_train, axis=0)
        scale_vec = np.std(X_train, axis=0)
        scale_vec[scale_vec == 0] = 1.0  # Prevent divide by zero

        # Save
        np.save(stats_mean_path, mean_vec)
        np.save(stats_scale_path, scale_vec)

    # Apply Scaling
    def apply_scaling(df, mean, scale, cols):
        df_scaled = df.copy()
        # Fill NaNs first
        df_scaled[cols] = df_scaled[cols].fillna(0)
        # Transform
        df_scaled[cols] = (df_scaled[cols] - mean) / scale
        return df_scaled

    print("Applying scaling to tabular features...")
    train_feats = apply_scaling(train_feats, mean_vec, scale_vec, feature_cols)
    val_feats = apply_scaling(val_feats, mean_vec, scale_vec, feature_cols)
    test_feats = apply_scaling(test_feats, mean_vec, scale_vec, feature_cols)

    # 4. Scale Targets
    # Using the provided TargetScaler class
    target_scaler = TargetScaler()

    if load_cached_data and os.path.exists(Config.TARGET_MEAN_PATH):
        print("Loading target scaler...")
        target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
    else:
        print("Fitting target scaler...")
        target_scaler.fit(train_meta["time_to_eruption"].values)
        target_scaler.save(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    # Transform targets in metadata
    train_meta["time_to_eruption"] = target_scaler.transform(
        train_meta["time_to_eruption"].values
    )
    val_meta["time_to_eruption"] = target_scaler.transform(
        val_meta["time_to_eruption"].values
    )
    # Test metadata has no target

    # 5. Create Datasets
    train_dataset = VolcanoDataset(train_meta, train_feats, mode="train")
    val_dataset = VolcanoDataset(val_meta, val_feats, mode="val")
    test_dataset = VolcanoDataset(test_meta, test_feats, mode="test")

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    print(
        f"DataLoaders ready. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader, target_scaler
