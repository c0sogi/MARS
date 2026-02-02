import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.utils import (
    scale_target,
    save_scaler,
    load_scaler,
    SCALER_MEAN_PATH,
    SCALER_SCALE_PATH,
)
from library.features import (
    get_spectrogram,
    spec_augment,
    generate_static_features,
)


class VolcanoDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        mode="train",
        augment=False,
        load_cached_stats=True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply SpecAugment.
            load_cached_stats (bool): Whether to load stats from parquet cache.
        """
        self.mode = mode
        self.augment = augment
        self.metadata_path = metadata_path

        # 1. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)

        # 2. Load/Generate Statistical Features
        # Use unique cache names for each split to avoid collisions in the simple caching logic
        cache_name = f"{mode}_features.parquet"
        self.stats_df = generate_static_features(
            self.metadata, cache_name=cache_name, load_cached_data=load_cached_stats
        )

        # Ensure stats_df is aligned with metadata (though generate_static_features usually handles this,
        # we index by segment_id for fast lookup)
        # The generate_static_features returns a DF indexed by segment_id.

        # 3. Feature Normalization (Statistics Branch)
        stats_mean_path = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
        stats_std_path = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")

        # Convert to float32 to save memory and match torch default
        self.stats_values = self.stats_df.astype(np.float32)

        if self.mode == "train":
            # Compute and save stats scaler
            print("Computing feature normalization statistics...")
            feat_mean = self.stats_values.mean(axis=0).values
            feat_std = self.stats_values.std(axis=0).values

            np.save(stats_mean_path, feat_mean)
            np.save(stats_std_path, feat_std)
        else:
            # Load stats scaler
            if not os.path.exists(stats_mean_path) or not os.path.exists(
                stats_std_path
            ):
                raise FileNotFoundError(
                    f"Feature scaler files not found in {WORKING_DIR}. "
                    "Run training set initialization first."
                )
            feat_mean = np.load(stats_mean_path)
            feat_std = np.load(stats_std_path)

        # Apply normalization: (X - mean) / (std + eps)
        # Using pandas broadcasting or numpy broadcasting
        self.stats_values = (self.stats_values - feat_mean) / (feat_std + 1e-8)

        # 4. Target Normalization (Regression Target)
        self.target_mean = 0.0
        self.target_std = 1.0

        if self.mode != "test":
            if self.mode == "train":
                # Compute and save target scaler
                targets = self.metadata["time_to_eruption"].values
                mean = np.mean(targets)
                std = np.std(targets)
                save_scaler(mean, std)
                self.target_mean = mean
                self.target_std = std
                print(f"Target Scaler Saved: Mean={mean:.4f}, Std={std:.4f}")
            else:
                # Load target scaler
                try:
                    self.target_mean, self.target_std = load_scaler()
                except FileNotFoundError:
                    # Fallback if val is initialized before train (unlikely but possible in dev)
                    # For safety, we just warn or use defaults, but strictly we should raise error.
                    # We'll raise error to enforce correct pipeline order.
                    raise FileNotFoundError(
                        "Target scaler not found. Run training set initialization first."
                    )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.metadata.iloc[idx]
        segment_id = int(row["segment_id"])

        # ---------------------------------------------------------
        # 1. Spectrogram Branch
        # ---------------------------------------------------------
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load sensor data
        # We assume file exists as per metadata checks, but good to be safe
        try:
            df_sensor = pd.read_csv(file_path)
        except FileNotFoundError:
            # Fallback for missing files (should not happen with correct metadata)
            # Create dummy data of correct shape (60001, 10)
            df_sensor = pd.DataFrame(
                np.zeros((60001, 10)), columns=[f"sensor_{i}" for i in range(1, 11)]
            )

        # Generate Log-Mel Spectrogram
        # Shape: (10, 128, T)
        spec = get_spectrogram(df_sensor)

        # Augmentation
        if self.augment:
            spec = spec_augment(spec)

        # ---------------------------------------------------------
        # 2. Statistics Branch
        # ---------------------------------------------------------
        # Retrieve pre-normalized stats
        try:
            stats_vec = self.stats_values.loc[segment_id].values
        except KeyError:
            # Fallback if ID missing from stats cache (unlikely)
            stats_vec = np.zeros(self.stats_values.shape[1], dtype=np.float32)

        stats_tensor = torch.tensor(stats_vec, dtype=torch.float32)

        # ---------------------------------------------------------
        # 3. Target
        # ---------------------------------------------------------
        if self.mode == "test":
            target_val = 0.0
        else:
            raw_target = row["time_to_eruption"]
            target_val = scale_target(raw_target, self.target_mean, self.target_std)

        target_tensor = torch.tensor(target_val, dtype=torch.float32)

        return spec, stats_tensor, target_tensor, segment_id
