import os
import torch
import pandas as pd
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import load_sensor_data, generate_log_mel_spectrogram


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for the Volcano Eruption Prediction task.

    Serves two inputs per sample:
    1. Log-Mel Spectrogram (10 channels) for the EfficientNet backbone.
    2. Statistical Feature Vector for the MLP branch.

    Also handles SpecAugment and Target/Feature scaling.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        augment: bool = False,
        target_scaler=None,
        stats_scaler=None,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'segment_id', 'file_path',
                                        and optionally 'time_to_eruption'.
            stats_df (pd.DataFrame): DataFrame containing pre-computed statistical features.
            augment (bool): Whether to apply SpecAugment (Time/Freq masking).
            target_scaler (TargetScaler, optional): Instance to scale the target variable.
            stats_scaler (StandardScaler, optional): Instance to scale the stats vector.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.augment = augment
        self.target_scaler = target_scaler
        self.stats_scaler = stats_scaler

        # Index stats by segment_id for O(1) lookup
        # We drop non-feature columns if they exist to keep only the feature vector
        if "segment_id" in stats_df.columns:
            self.stats_data = stats_df.set_index("segment_id")
        else:
            self.stats_data = stats_df

        # Drop target from stats if it accidentally leaked in there,
        # though data_processing usually handles this.
        if "time_to_eruption" in self.stats_data.columns:
            self.stats_data = self.stats_data.drop(columns=["time_to_eruption"])

        # Identify feature columns (ensure consistent order)
        self.feature_cols = sorted(self.stats_data.columns.tolist())

        # Initialize SpecAugment Transforms
        # Conservative masking: <15% of dimensions
        # Freq: 128 mels * 0.15 ~= 19
        # Time: ~235 frames * 0.15 ~= 35
        self.freq_masking = T.FrequencyMasking(freq_mask_param=19)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = row["segment_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # ---------------------------------------------------------
        # 1. Load and Process Sensor Data (Spectrogram)
        # ---------------------------------------------------------
        # Load raw data (fills NaNs with 0)
        df_sensor = load_sensor_data(file_path)

        # Generate Log-Mel Spectrogram -> Shape: [Channels, n_mels, Time]
        # Returns torch.Tensor
        spectrogram = generate_log_mel_spectrogram(df_sensor)

        # Apply Augmentation (Training only)
        if self.augment:
            # torchaudio transforms expect (..., freq, time)
            # Our shape is (Channels, Freq, Time), which works fine.
            spectrogram = self.freq_masking(spectrogram)
            spectrogram = self.time_masking(spectrogram)

        # ---------------------------------------------------------
        # 2. Retrieve Statistical Features (MLP Input)
        # ---------------------------------------------------------
        try:
            stats_row = self.stats_data.loc[segment_id, self.feature_cols]
            stats_vector = stats_row.values.astype(np.float32)
        except KeyError:
            # Fallback if stats are missing (should not happen if pipeline is correct)
            # Return zero vector matching feature dimension
            stats_vector = np.zeros(len(self.feature_cols), dtype=np.float32)

        # Apply Scaling to Stats if scaler provided
        if self.stats_scaler is not None:
            # Reshape for scaler (1, n_features) then flatten back
            stats_vector = self.stats_scaler.transform(
                stats_vector.reshape(1, -1)
            ).flatten()

        stats_tensor = torch.tensor(stats_vector, dtype=torch.float32)

        # ---------------------------------------------------------
        # 3. Retrieve and Scale Target
        # ---------------------------------------------------------
        if "time_to_eruption" in row:
            target_val = row["time_to_eruption"]

            # Apply Target Scaling
            if self.target_scaler is not None:
                # Transform expects array-like, returns array
                # We extract the scalar float
                target_val = self.target_scaler.transform([target_val])[0]

            target_tensor = torch.tensor(target_val, dtype=torch.float32)
            return spectrogram, stats_tensor, target_tensor
        else:
            # Inference mode (Test set)
            return spectrogram, stats_tensor
