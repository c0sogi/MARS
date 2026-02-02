import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import Config


class FeatureExtractor:
    """
    Module for offline feature extraction and caching.
    Generates 3-Channel Multi-Resolution Log-Mel Spectrograms.
    """

    @staticmethod
    def _get_cache_path(filepath):
        """
        Generates a deterministic cache file path based on the input filepath.
        Uses MD5 hash of the relative filepath to ensure safe and unique filenames.
        """
        # Encode the relative path to bytes and hash it
        file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"{file_hash}.npy")

    @staticmethod
    def compute_multires_spec(audio_path):
        """
        Computes three separate Mel Spectrograms with different window sizes
        (20ms, 40ms, 60ms) and stacks them into a (3, H, W) tensor.

        Args:
            audio_path (str): Full path to the audio file.

        Returns:
            np.ndarray: A (3, N_MELS, TIME_STEPS) numpy array.
        """
        # 1. Load Audio
        # Use a try-except block to handle potential corrupted files
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Return a zero tensor of correct shape in case of error to prevent crash
            # Time steps = 1 + N_SAMPLES // HOP_LENGTH = 1 + 16000 // 160 = 101
            n_steps = 1 + Config.N_SAMPLES // Config.HOP_LENGTH
            return np.zeros((3, Config.N_MELS, n_steps), dtype=np.float32)

        # 2. Resample if necessary
        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # 3. Mix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Pad or Truncate to fixed duration
        target_len = Config.N_SAMPLES
        current_len = waveform.shape[1]

        if current_len > target_len:
            # Truncate (take first 1 second)
            waveform = waveform[:, :target_len]
        elif current_len < target_len:
            # Pad with zeros
            pad_amount = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # 5. Compute Multi-Resolution Spectrograms
        specs = []
        # Ensure n_fft is large enough for the largest window (0.06s * 16000 = 960 samples)
        n_fft = 1024

        for win_length in Config.WINDOW_SIZES:
            # Define Transform
            mel_transform = T.MelSpectrogram(
                sample_rate=Config.SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                f_min=Config.F_MIN,
                f_max=Config.F_MAX,
                center=True,  # Ensures consistent time dimension
                pad_mode="reflect",
                power=2.0,
            )

            # Compute Spec
            spec = mel_transform(waveform)  # Shape: (1, n_mels, time)

            # Log Scale (AmplitudeToDB)
            db_transform = T.AmplitudeToDB(top_db=80.0)
            log_spec = db_transform(spec)

            specs.append(log_spec)

        # 6. Stack into (3, n_mels, time)
        # Each spec is (1, 64, 101), cat along dim 0 -> (3, 64, 101)
        multires_spec = torch.cat(specs, dim=0)

        return multires_spec.numpy().astype(np.float32)

    @staticmethod
    def cache_features(df, load_cached_data=True):
        """
        Iterates through the metadata dataframe, processes every audio file,
        and saves the resulting tensors to disk as .npy files.

        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' column.
            load_cached_data (bool): If True, skips processing if the cache file exists.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        processed_count = 0
        skipped_count = 0

        for _, row in df.iterrows():
            rel_path = row["filepath"]
            # Full path to source audio
            audio_path = os.path.join(Config.INPUT_ROOT, rel_path)

            # Path to cached file
            cache_path = FeatureExtractor._get_cache_path(rel_path)

            # Logic:
            # 1. IF load_cached_data is True: Try to load (check existence).
            # 2. IF loading fails OR load_cached_data is False: Compute and Save.

            should_compute = True
            if load_cached_data:
                if os.path.exists(cache_path):
                    should_compute = False

            if should_compute:
                # Compute
                data = FeatureExtractor.compute_multires_spec(audio_path)
                # Save
                np.save(cache_path, data)
                processed_count += 1
            else:
                skipped_count += 1

        print(
            f"Feature Extraction: Processed {processed_count} files, Skipped {skipped_count} (Cached)."
        )
