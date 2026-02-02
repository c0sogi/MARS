import os
import hashlib
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from library.config import audio_cfg, path_cfg


def create_mel_basis(sr, n_fft, n_mels, fmin, fmax):
    """
    Creates a Mel filterbank matrix using pure NumPy.
    """
    n_freqs = n_fft // 2 + 1

    # Mel scale conversion functions
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # Create Mel points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Map to FFT bins
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    weights = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        start = bin_points[i]
        center = bin_points[i + 1]
        end = bin_points[i + 2]

        # Upslope
        if center > start:
            weights[i, start:center] = (np.arange(start, center) - start) / (
                center - start
            )
        # Downslope
        if end > center:
            weights[i, center:end] = (end - np.arange(center, end)) / (end - center)

    return weights


def amplitude_to_db(x, top_db=80.0):
    """
    Converts a power spectrogram to decibel scale.
    """
    x_db = 10 * torch.log10(torch.clamp(x, min=1e-10))
    max_val = x_db.max()
    x_db = torch.clamp(x_db, min=max_val - top_db)
    return x_db


class MultiResLogMelSpectrogram(torch.nn.Module):
    """
    Generates a 3-channel Log-Mel Spectrogram with different window sizes
    using native PyTorch operations.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        # Pre-compute Mel Basis (shared across windows if n_fft is same)
        mel_basis_np = create_mel_basis(
            self.cfg.sample_rate,
            self.cfg.n_fft,
            self.cfg.n_mels,
            self.cfg.f_min,
            self.cfg.f_max,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis_np).float())

        # Create windows for each resolution
        for win_len in self.cfg.win_lengths:
            # Hann window
            win = torch.hann_window(win_len)
            self.register_buffer(f"window_{win_len}", win)

    def __call__(self, waveform):
        """
        Args:
            waveform (Tensor): Audio waveform of shape (1, n_samples)
        Returns:
            Tensor: Multi-channel spectrogram of shape (3, n_mels, time_steps)
        """
        specs = []

        for win_len in self.cfg.win_lengths:
            window = getattr(self, f"window_{win_len}")

            # STFT
            stft = torch.stft(
                waveform,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=win_len,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )

            # Power Spec: (1, Freq, Time)
            power_spec = stft.abs().pow(2.0)

            # Mel Spec: (1, n_mels, Time)
            # mel_basis: (n_mels, Freq)
            # Result: (1, n_mels, Time)
            mel_spec = torch.matmul(self.mel_basis, power_spec)

            # DB Conversion
            spec_db = amplitude_to_db(mel_spec, top_db=80.0)
            specs.append(spec_db)

        # Stack spectrograms along the channel dimension
        multi_res_spec = torch.cat(specs, dim=0)
        return multi_res_spec


def load_audio(filepath, target_samples=16000):
    """
    Loads an audio file using soundfile, resamples if needed, converts to mono,
    and pads/truncates.
    """
    if not os.path.exists(filepath):
        return torch.zeros(1, target_samples)

    try:
        # sf.read returns (data, samplerate)
        # data is (samples,) or (samples, channels)
        data, sr = sf.read(filepath)
        waveform = torch.from_numpy(data).float()
    except Exception as e:
        print(f"Warning: Failed to load {filepath}. Returning silence. Error: {e}")
        return torch.zeros(1, target_samples)

    # Handle channels: Ensure (1, Samples)
    if waveform.ndim > 1:
        # (Samples, Channels) -> (Channels, Samples)
        waveform = waveform.t()
        # Mean to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    else:
        # (Samples,) -> (1, Samples)
        waveform = waveform.unsqueeze(0)

    # Resample if necessary
    if sr != 16000:
        # Simple interpolation
        new_len = int(waveform.shape[1] * 16000 / sr)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)

    # Pad or Truncate to target_samples
    num_samples = waveform.shape[1]
    if num_samples < target_samples:
        padding = target_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif num_samples > target_samples:
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
