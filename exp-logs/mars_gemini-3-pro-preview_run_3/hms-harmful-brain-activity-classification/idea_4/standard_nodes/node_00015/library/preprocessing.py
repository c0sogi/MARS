import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from library.config import Config
from library.utils import get_logger, seed_everything


class EEGPreprocessor:
    def __init__(self):
        """
        Initializes the EEGPreprocessor with signal processing transforms.
        """
        self.logger = get_logger("preprocessor")
        self.device = torch.device(
            "cpu"
        )  # Preprocessing on CPU to save GPU for training

        # Stream A: Resampler (200Hz -> 50Hz)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=Config.ORIGINAL_SAMPLING_RATE, new_freq=Config.RAW_SAMPLING_RATE
        ).to(self.device)

        # Stream B: Mel Spectrogram
        # Input: (Channels, Time) -> Output: (Channels, n_mels, time_steps)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.ORIGINAL_SAMPLING_RATE,
            n_fft=Config.N_FFT,
            win_length=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        ).to(self.device)

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0).to(
            self.device
        )

    def _load_parquet(self, path, columns=None):
        """
        Safely loads a parquet file.
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            return pd.read_parquet(full_path, columns=columns)
        except Exception as e:
            self.logger.error(f"Failed to load {full_path}: {e}")
            return None

    def _get_window(self, df, offset_seconds):
        """
        Extracts the 50-second window (10,000 samples) based on offset.
        """
        # Calculate start index
        start_idx = int(offset_seconds * Config.ORIGINAL_SAMPLING_RATE)
        # Handle negative offsets or offsets beyond file length (robustness)
        start_idx = max(0, start_idx)

        end_idx = start_idx + (Config.DURATION * Config.ORIGINAL_SAMPLING_RATE)

        # Extract data
        data = df.values

        # Pad if necessary
        time_steps, channels = data.shape

        if end_idx > time_steps:
            # Pad with zeros at the end
            padding = np.zeros((end_idx - time_steps, channels), dtype=data.dtype)
            data = np.vstack([data, padding])

        # Slice the window
        window = data[start_idx:end_idx]

        # Ensure exact length (handle edge cases where slice might be short)
        expected_len = Config.DURATION * Config.ORIGINAL_SAMPLING_RATE
        if len(window) < expected_len:
            padding = np.zeros((expected_len - len(window), channels), dtype=data.dtype)
            window = np.vstack([window, padding])

        return window

    def preprocess_raw_stream(self, eeg_tensor):
        """
        Processes the raw stream: Resample 200Hz -> 50Hz, Normalize.
        Input: Tensor (Channels, Time)
        Output: Numpy (Time_New, Channels)
        """
        # Resample: (Channels, 10000) -> (Channels, 2500)
        resampled = self.resampler(eeg_tensor)

        # Transpose back to (Time, Channels) for GRU input
        resampled = resampled.permute(1, 0).numpy()

        # Z-score normalization per channel
        # Avoid division by zero with eps
        mean = np.mean(resampled, axis=0, keepdims=True)
        std = np.std(resampled, axis=0, keepdims=True)
        normalized = (resampled - mean) / (std + 1e-6)

        return normalized.astype(np.float32)

    def preprocess_spec_stream(self, eeg_tensor):
        """
        Processes the spec stream: MelSpec, Log, Resize, Normalize.
        Input: Tensor (Channels, Time)
        Output: Numpy (Channels, Freq, Time_New) -> (19, 64, 256)
        """
        # Compute Mel Spectrogram: (Channels, n_mels, time_frames)
        mel_spec = self.mel_transform(eeg_tensor)

        # Log scale
        log_mel = self.amp_to_db(mel_spec)

        # Resize to fixed dimensions (Config.SPEC_HEIGHT, Config.SPEC_WIDTH)
        # Input to interpolate needs to be (Batch, Channels, H, W) or (Batch, C, L)
        # We treat the 19 EEG channels as a batch of 1-channel images for resizing
        # Shape: (19, 64, T) -> Unsqueeze -> (19, 1, 64, T)
        img = log_mel.unsqueeze(1)

        resized = F.interpolate(
            img,
            size=(Config.SPEC_HEIGHT, Config.SPEC_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

        # Squeeze back: (19, 64, 256)
        resized = resized.squeeze(1).numpy()

        # Z-score normalization per channel (over freq and time dimensions)
        mean = np.mean(resized, axis=(1, 2), keepdims=True)
        std = np.std(resized, axis=(1, 2), keepdims=True)
        normalized = (resized - mean) / (std + 1e-6)

        return normalized.astype(np.float32)

    def process_sample(self, row, is_test=False):
        """
        Loads and processes a single sample.
        """
        # Load EEG parquet
        df_eeg = self._load_parquet(row["eeg_path"], columns=Config.EEG_CHANNELS)
        if df_eeg is None:
            # Return zeros if load fails
            raw_shape = (Config.RAW_SEQUENCE_LENGTH, Config.N_CHANNELS)
            spec_shape = (Config.N_CHANNELS, Config.SPEC_HEIGHT, Config.SPEC_WIDTH)
            targets = np.zeros(Config.N_CLASSES) if not is_test else None
            return (
                np.zeros(raw_shape, dtype=np.float32),
                np.zeros(spec_shape, dtype=np.float32),
                targets,
            )

        # Determine offset
        offset = 0
        if "eeg_label_offset_seconds" in row:
            offset = row["eeg_label_offset_seconds"]

        # Extract window (10000, 19)
        eeg_data = self._get_window(df_eeg, offset)

        # Handle NaNs in raw data
        eeg_data = np.nan_to_num(eeg_data, nan=0.0)

        # Convert to tensor (Channels, Time) for Torchaudio
        eeg_tensor = torch.tensor(eeg_data.T, dtype=torch.float32).to(self.device)

        # Process Streams
        raw_feat = self.preprocess_raw_stream(eeg_tensor)
        spec_feat = self.preprocess_spec_stream(eeg_tensor)

        # Extract Targets
        targets = None
        if not is_test:
            targets = row[Config.TARGET_COLS].values.astype(np.float32)

        return raw_feat, spec_feat, targets

    def process_dataset(
        self,
        df,
        data_cache_path,
        target_cache_path=None,
        mode="train",
        load_cached=True,
    ):
        """
        Main method to process a dataset. Handles caching using memmap to avoid OOM.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            data_cache_path (str): Path to save/load input features (dictionary of arrays).
            target_cache_path (str): Path to save/load targets.
            mode (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (raw_data, spec_data, targets)
        """
        # Define paths
        raw_path = data_cache_path.replace(".npy", "_raw.npy")
        spec_path = data_cache_path.replace(".npy", "_spec.npy")

        # 1. Check Cache
        if load_cached and os.path.exists(raw_path) and os.path.exists(spec_path):
            if mode == "test" or (
                target_cache_path and os.path.exists(target_cache_path)
            ):
                self.logger.info(f"Loading cached data from {data_cache_path}...")
                try:
                    # Use mmap_mode='r' to avoid loading everything into RAM
                    raw_data = np.load(raw_path, mmap_mode="r")
                    spec_data = np.load(spec_path, mmap_mode="r")

                    targets = None
                    if mode != "test" and target_cache_path:
                        targets = np.load(target_cache_path)

                    self.logger.info(
                        f"Loaded successfully (mmap). Raw: {raw_data.shape}, Spec: {spec_data.shape}"
                    )
                    return raw_data, spec_data, targets
                except Exception as e:
                    self.logger.warning(f"Cache load failed: {e}. Reprocessing...")

        # 2. Process Data
        self.logger.info(f"Processing {len(df)} samples for {mode}...")

        # Define shapes
        n_samples = len(df)
        raw_shape = (n_samples, Config.RAW_SEQUENCE_LENGTH, Config.N_CHANNELS)
        spec_shape = (
            n_samples,
            Config.N_CHANNELS,
            Config.SPEC_HEIGHT,
            Config.SPEC_WIDTH,
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)

        # Create memmaps for writing
        # np.lib.format.open_memmap initializes a .npy file on disk
        raw_memmap = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32, shape=raw_shape
        )
        spec_memmap = np.lib.format.open_memmap(
            spec_path, mode="w+", dtype=np.float32, shape=spec_shape
        )

        target_list = []
        is_test = mode == "test"

        # Iterate and fill memmaps
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % 1000 == 0:
                self.logger.info(f"Processed {idx}/{len(df)}")

            r, s, t = self.process_sample(row, is_test=is_test)

            # Assign directly to disk-backed array
            raw_memmap[idx] = r
            spec_memmap[idx] = s

            if not is_test:
                target_list.append(t)

        # Flush changes to disk
        raw_memmap.flush()
        spec_memmap.flush()

        self.logger.info(f"Processing complete. Saved to {raw_path} and {spec_path}")

        # Handle Targets (small enough for RAM)
        targets = None
        if not is_test:
            targets = np.stack(target_list)
            if target_cache_path:
                np.save(target_cache_path, targets)
                self.logger.info(f"Saved targets to {target_cache_path}")

        # Clean up write-mode memmaps and reload in read-mode
        del raw_memmap
        del spec_memmap

        # Reload with mmap_mode='r' for safety and efficiency
        raw_data = np.load(raw_path, mmap_mode="r")
        spec_data = np.load(spec_path, mmap_mode="r")

        return raw_data, spec_data, targets

    def get_dataset(
        self,
        metadata_path,
        cache_data_path,
        cache_target_path=None,
        mode="train",
        load_cached=True,
        debug=False,
    ):
        """
        Wrapper to load metadata and trigger processing.
        """
        df = pd.read_csv(metadata_path)

        if debug:
            df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
            # Adjust cache paths for debug to avoid overwriting full cache
            cache_data_path = cache_data_path.replace(".npy", "_debug.npy")
            if cache_target_path:
                cache_target_path = cache_target_path.replace(".npy", "_debug.npy")

        return self.process_dataset(
            df, cache_data_path, cache_target_path, mode, load_cached
        )
