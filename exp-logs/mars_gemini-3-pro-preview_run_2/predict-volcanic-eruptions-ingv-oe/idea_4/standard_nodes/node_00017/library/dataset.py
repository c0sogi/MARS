import os
import torch
import numpy as np
import pandas as pd
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config
from library.feature_engineering import generate_spectrogram, get_statistical_features


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for Volcano Eruption Prediction.

    Handles:
    1. Loading raw sensor data and converting to Spectrograms.
    2. Loading pre-computed statistical features.
    3. Applying SpecAugment (during training).
    4. Normalizing inputs (Spectrograms and Stats).
    5. Scaling targets.
    """

    def __init__(self, metadata_path, mode="train", target_scaler=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'.
            target_scaler (TargetScaler, optional): Instance of TargetScaler to handle label scaling.
        """
        self.mode = mode
        self.target_scaler = target_scaler

        # 1. Load Metadata
        self.metadata = pd.read_csv(metadata_path)

        # 2. Determine paths for cached features based on mode
        if self.mode == "train":
            feature_cache_path = Config.TRAIN_FEATURES_PATH
        elif self.mode == "val":
            feature_cache_path = Config.VAL_FEATURES_PATH
        elif self.mode == "test":
            feature_cache_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 3. Load Statistical Features (Cached)
        # This returns a DataFrame with segment_id and features
        self.stats_df = get_statistical_features(
            metadata_path=metadata_path,
            save_path=feature_cache_path,
            load_cached_data=True,
        )

        # Index by segment_id for O(1) retrieval
        self.stats_df = self.stats_df.set_index("segment_id")

        # Identify feature columns (exclude segment_id which is now index)
        self.feature_cols = [c for c in self.stats_df.columns]

        # 4. Setup Feature Scaling (StandardScaler for Stats)
        self._setup_feature_scaling()

        # 5. Setup Target Scaling (Fit if train)
        if self.mode == "train" and self.target_scaler is not None:
            targets = self.metadata["time_to_eruption"].values
            self.target_scaler.fit(targets)

        # 6. Setup Augmentation (Train only)
        if self.mode == "train":
            # Conservative masking parameters (Cite Lesson 8)
            # Reduced to prevent destruction of critical signal features
            self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
            self.time_masking = T.TimeMasking(time_mask_param=30)

    def _setup_feature_scaling(self):
        """
        Computes or loads mean/std for statistical features to ensure
        standard normal distribution inputs to the MLP.
        """
        mean_path = Config.STATS_SCALER_MEAN
        scale_path = Config.STATS_SCALER_SCALE

        if self.mode == "train":
            # Compute stats from the loaded dataframe
            features = self.stats_df[self.feature_cols].values
            self.stats_mean = np.mean(features, axis=0)
            self.stats_scale = np.std(features, axis=0)

            # Avoid division by zero
            self.stats_scale[self.stats_scale < 1e-9] = 1.0

            # Save to disk
            np.save(mean_path, self.stats_mean)
            np.save(scale_path, self.stats_scale)
        else:
            # Load from disk
            if os.path.exists(mean_path) and os.path.exists(scale_path):
                self.stats_mean = np.load(mean_path)
                self.stats_scale = np.load(scale_path)
            else:
                # Fallback if validation/test run before train (should not happen in pipeline)
                # Default to identity scaling
                print("Warning: Stats scaler files not found. Using identity scaling.")
                self.stats_mean = np.zeros(len(self.feature_cols))
                self.stats_scale = np.ones(len(self.feature_cols))

        # Convert to torch tensors for faster operation in __getitem__
        self.stats_mean = torch.tensor(self.stats_mean, dtype=torch.float32)
        self.stats_scale = torch.tensor(self.stats_scale, dtype=torch.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = int(row["segment_id"])

        # ---------------------------------------------------------
        # 1. Spectrogram Generation
        # ---------------------------------------------------------
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load raw CSV
        try:
            df_sensor = pd.read_csv(file_path)
        except FileNotFoundError:
            # Fallback for missing files (robustness)
            df_sensor = pd.DataFrame(
                np.zeros((Config.SIGNAL_LENGTH, Config.NUM_SENSORS)),
                columns=[f"sensor_{i}" for i in range(1, Config.NUM_SENSORS + 1)],
            )

        # Generate Spectrogram [Channels, Mel, Time]
        # Returns tensor in dB scale (approx -80 to 0)
        spec = generate_spectrogram(df_sensor)

        # Normalize Spectrogram to [0, 1] range
        # Assuming top_db=80, values are [-80, 0].
        # (x + 80) / 80 maps -80->0 and 0->1.
        spec = (spec + 80.0) / 80.0

        # Apply Augmentation (Train only)
        if self.mode == "train":
            # SpecAugment expects (..., Freq, Time)
            # Our spec is (Channels, Freq, Time), which works fine.
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # ---------------------------------------------------------
        # 2. Statistical Features
        # ---------------------------------------------------------
        # Retrieve from cached dataframe
        if segment_id in self.stats_df.index:
            stats_vec = self.stats_df.loc[segment_id, self.feature_cols].values
            stats_vec = torch.tensor(stats_vec, dtype=torch.float32)
        else:
            # Should not happen if cache is consistent
            stats_vec = torch.zeros(len(self.feature_cols), dtype=torch.float32)

        # Normalize Stats
        stats_vec = (stats_vec - self.stats_mean) / self.stats_scale

        # ---------------------------------------------------------
        # 3. Target Variable
        # ---------------------------------------------------------
        target_val = 0.0
        if self.mode != "test":
            raw_target = row["time_to_eruption"]

            # Scale target
            if self.target_scaler is not None:
                # Transform expects 2D array, returns 2D array
                scaled_target = self.target_scaler.transform(np.array([[raw_target]]))
                target_val = scaled_target[0, 0]
            else:
                target_val = raw_target

        target_tensor = torch.tensor(target_val, dtype=torch.float32)

        # ---------------------------------------------------------
        # 4. Return
        # ---------------------------------------------------------
        return {
            "spectrogram": spec,  # [C, F, T]
            "features": stats_vec,  # [Feature_Dim]
            "target": target_tensor,  # Scalar
            "segment_id": segment_id,  # Int
        }
