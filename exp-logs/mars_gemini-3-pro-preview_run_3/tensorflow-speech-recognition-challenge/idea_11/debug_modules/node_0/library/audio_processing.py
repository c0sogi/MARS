import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torchaudio
from library.config import audio_cfg, path_cfg


class MultiResLogMelSpectrogram:
    """
    Generates a 3-channel Log-Mel Spectrogram with different window sizes.
    Channel 0: Short window (20ms) - High Temporal Resolution
    Channel 1: Medium window (40ms) - Balanced
    Channel 2: Long window (60ms) - High Frequency Resolution
    """

    def __init__(self, config):
        self.cfg = config
        self.transforms = []

        # Initialize a MelSpectrogram transform for each window length
        for win_len in self.cfg.win_lengths:
            # center=True and pad_mode='reflect' ensure consistent temporal dimensions
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                win_length=win_len,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                f_min=self.cfg.f_min,
                f_max=self.cfg.f_max,
                power=2.0,
                center=True,
                pad_mode="reflect",
                normalized=False,
            )
            self.transforms.append(melspec)

        # Standard Amplitude to DB conversion
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

    def __call__(self, waveform):
        """
        Args:
            waveform (Tensor): Audio waveform of shape (1, n_samples)
        Returns:
            Tensor: Multi-channel spectrogram of shape (3, n_mels, time_steps)
        """
        specs = []
        for t in self.transforms:
            # Apply STFT + Mel Filterbank
            spec = t(waveform)
            # Convert to dB
            spec = self.db_transform(spec)
            specs.append(spec)

        # Stack spectrograms along the channel dimension
        # Each spec is (1, n_mels, time) -> Result is (3, n_mels, time)
        multi_res_spec = torch.cat(specs, dim=0)
        return multi_res_spec


def load_audio(filepath, target_samples=16000):
    """
    Loads an audio file, resamples to 16kHz, converts to mono, and pads/truncates to fixed length.
    """
    if not os.path.exists(filepath):
        # Return silence if file is missing (robustness for bad paths)
        return torch.zeros(1, target_samples)

    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}. Returning silence. Error: {e}")
        return torch.zeros(1, target_samples)

    # Resample if necessary
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono (average channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Truncate to target_samples
    num_samples = waveform.shape[1]
    if num_samples < target_samples:
        padding = target_samples - num_samples
        # Pad at the end
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif num_samples > target_samples:
        # Truncate
        waveform = waveform[:, :target_samples]

    return waveform


def get_cache_path(filepath, cache_dir):
    """Generates a unique cache filename using MD5 hash of the relative filepath."""
    file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{file_hash}.npy")


def generate_cache(metadata_df, input_root, cache_dir, load_cached_data=True):
    """
    Iterates over the metadata DataFrame, processes audio files into Multi-Res Spectrograms,
    and caches them as .npy files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'filepath' column.
        input_root (str): Root directory for input audio files.
        cache_dir (str): Directory to save cached .npy files.
        load_cached_data (bool): If True, skips processing for existing cache files.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'cache_path' column.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize processor
    processor = MultiResLogMelSpectrogram(audio_cfg)

    cache_paths = []
    processed_count = 0
    total_files = len(metadata_df)

    print(f"Preparing cache for {total_files} files in {cache_dir}...")

    for idx, row in metadata_df.iterrows():
        rel_path = row["filepath"]
        full_audio_path = os.path.join(input_root, rel_path)
        cache_path = get_cache_path(rel_path, cache_dir)

        cache_paths.append(cache_path)

        # Check if cache exists
        if load_cached_data and os.path.exists(cache_path):
            continue

        # Process audio
        waveform = load_audio(full_audio_path, audio_cfg.num_samples)

        # Generate Spectrograms (No gradients needed)
        with torch.no_grad():
            spec = processor(waveform)  # Shape: (3, 64, 101)

        # Save to disk
        np.save(cache_path, spec.numpy())
        processed_count += 1

        if processed_count > 0 and processed_count % 5000 == 0:
            print(f"Processed and cached {processed_count} files...")

    if processed_count > 0:
        print(f"Cache generation complete. Processed {processed_count} new files.")
    else:
        print("All files found in cache. Skipping processing.")

    # Add cache paths to dataframe
    metadata_df["cache_path"] = cache_paths
    return metadata_df
