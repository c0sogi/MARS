import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from library.config import Config


class EEGProcessor:
    """
    Handles the loading, processing, and conversion of raw EEG signals
    into Log-Mel Spectrogram images for the classification model.
    """

    def __init__(self):
        # Initialize Mel Spectrogram Transform
        # We use torchaudio for GPU-accelerated (if available) and efficient processing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLING_RATE,
            n_fft=Config.N_FFT,
            win_length=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            power=2.0,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # Setup Cache Directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def compute_bipolar_montage(self, eeg_df: pd.DataFrame) -> np.ndarray:
        """
        Computes the Longitudinal Bipolar Montage (Double Banana).
        Generates 16 differential signals from the raw electrodes.

        Args:
            eeg_df: Pandas DataFrame containing raw electrode data.

        Returns:
            np.ndarray: Shape (16, Time_Steps) containing the differential signals.
        """
        # Fill missing values with 0 to ensure continuous signal for FFT
        eeg_df = eeg_df.fillna(0)

        signals = []
        # Explicit order: LL, RL, LP, RP
        chain_order = ["LL", "RL", "LP", "RP"]

        for chain in chain_order:
            if chain not in Config.MONTAGE_PAIRS:
                continue

            pairs = Config.MONTAGE_PAIRS[chain]
            for anode, cathode in pairs:
                # Calculate difference if columns exist, else 0
                if anode in eeg_df.columns and cathode in eeg_df.columns:
                    diff = eeg_df[anode].values - eeg_df[cathode].values
                else:
                    diff = np.zeros(len(eeg_df), dtype=np.float32)
                signals.append(diff)

        # Stack into a single array (Channels, Time)
        return np.stack(signals).astype(np.float32)

    def eeg_to_mel_spec(self, signals: np.ndarray) -> torch.Tensor:
        """
        Converts time-domain EEG signals to Log-Mel Spectrograms.

        Args:
            signals: np.ndarray of shape (Channels, Time).

        Returns:
            torch.Tensor: Shape (Channels, n_mels, Time_Frames).
        """
        # Convert to tensor
        tensor_signals = torch.from_numpy(signals)

        # Apply transforms
        # Input: (..., Time) -> Output: (..., n_mels, Time_Frames)
        mels = self.mel_transform(tensor_signals)
        mels_db = self.db_transform(mels)

        return mels_db

    def stack_and_resize(self, specs: torch.Tensor) -> torch.Tensor:
        """
        Stacks the multi-channel spectrograms into a single composite image,
        resizes to the target dimension, and normalizes.

        Args:
            specs: Tensor of shape (16, n_mels, Time_Frames).

        Returns:
            torch.Tensor: Shape (3, Height, Width) normalized to [0, 1].
        """
        # 1. Vertical Stacking
        # We flatten the channel and frequency dimensions to create a "tall" image
        # Shape: (16 * n_mels, Time_Frames)
        c, f, t = specs.shape
        composite = specs.reshape(c * f, t)

        # 2. Resize
        # Add batch and channel dims for interpolation: (1, 1, H_in, W_in)
        composite = composite.unsqueeze(0).unsqueeze(0)

        target_h, target_w = Config.IMG_SIZE

        # Bilinear interpolation to target size
        resized = torch.nn.functional.interpolate(
            composite, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        # Remove batch/channel dims: (H, W)
        resized = resized.squeeze()

        # 3. Min-Max Normalization to [0, 1]
        r_min = resized.min()
        r_max = resized.max()

        if r_max - r_min > 1e-6:
            resized = (resized - r_min) / (r_max - r_min)
        else:
            resized = torch.zeros_like(resized)

        # 4. Replicate to 3 Channels (RGB)
        # EfficientNet expects 3 input channels
        img_3c = resized.unsqueeze(0).repeat(3, 1, 1)

        return img_3c

    def load_and_process(
        self,
        eeg_path: str,
        offset_seconds: float,
        load_cached_data: bool = False,
        cache_id: str = None,
    ) -> torch.Tensor:
        """
        Orchestrates the loading, slicing, processing, and optional caching of an EEG sample.

        Args:
            eeg_path: Relative path to the parquet file (e.g., 'train_eegs/123.parquet').
            offset_seconds: Start time of the annotated window in seconds.
            load_cached_data: If True, attempts to load from disk before processing.
            cache_id: Unique identifier for the cache file (usually eeg_id_sub_id).

        Returns:
            torch.Tensor: The processed image tensor (3, H, W).
        """
        # --- Caching Logic ---
        cache_file = None
        if cache_id:
            cache_file = os.path.join(self.cache_dir, f"{cache_id}.pt")

        if load_cached_data and cache_file and os.path.exists(cache_file):
            try:
                # Load cached tensor
                return torch.load(cache_file, weights_only=True)
            except Exception:
                # If load fails, proceed to recompute
                pass

        # --- Data Loading ---
        full_path = os.path.join(Config.INPUT_DIR, eeg_path)

        try:
            df = pd.read_parquet(full_path)
        except FileNotFoundError:
            # Return zero tensor if file is missing (safety fallback)
            return torch.zeros((3, *Config.IMG_SIZE), dtype=torch.float32)

        # --- Time Slicing ---
        # Calculate sample indices
        start_sample = int(offset_seconds * Config.SAMPLING_RATE)
        end_sample = start_sample + Config.EEG_LENGTH

        # Boundary checks
        max_len = len(df)
        if start_sample < 0:
            start_sample = 0
        if end_sample > max_len:
            end_sample = max_len

        df_slice = df.iloc[start_sample:end_sample]

        # Padding if the slice is shorter than expected (e.g., end of file)
        if len(df_slice) < Config.EEG_LENGTH:
            pad_len = Config.EEG_LENGTH - len(df_slice)
            # Create zero-filled dataframe for padding
            padding_df = pd.DataFrame(0, index=range(pad_len), columns=df.columns)
            df_slice = pd.concat([df_slice, padding_df], axis=0, ignore_index=True)

        # --- Signal Processing ---
        # 1. Compute Montage
        signals = self.compute_bipolar_montage(df_slice)

        # 2. Convert to Spectrogram
        specs = self.eeg_to_mel_spec(signals)

        # 3. Stack, Resize, Normalize
        img_tensor = self.stack_and_resize(specs)

        # --- Save to Cache ---
        if cache_file:
            try:
                torch.save(img_tensor, cache_file)
            except Exception:
                pass  # Ignore save errors to avoid crashing training

        return img_tensor
