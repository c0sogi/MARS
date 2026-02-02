import os
import hashlib
import torch
import random
import numpy as np
import soundfile as sf
import scipy.signal
from library import config

# Import set_seed from config to avoid re-implementation
from library.config import set_seed


def load_and_pad_audio(
    filepath,
    target_len=config.AUDIO_LEN,
    target_sr=config.SAMPLE_RATE,
    load_cached_data=True,
):
    """
    Loads an audio file, resamples it if necessary, and pads/truncates it to a fixed length.
    Implements caching to speed up subsequent loads.

    Uses soundfile and scipy instead of torchaudio to avoid binary incompatibility issues.

    Args:
        filepath (str): Relative path to the audio file from INPUT_DIR or absolute path.
        target_len (int): Desired length in samples.
        target_sr (int): Desired sample rate.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        torch.Tensor: The processed audio waveform of shape (1, target_len).
    """
    # Ensure cache directory exists
    cache_dir = os.path.join(config.WORK_DIR, "audio_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key based on filepath and parameters to ensure uniqueness
    # We include target_len and target_sr in the hash so that changing params invalidates cache
    key_str = f"{filepath}_{target_len}_{target_sr}"
    file_hash = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{file_hash}.npy")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_np = np.load(cache_path)
            return torch.from_numpy(data_np)
        except Exception:
            # If load fails (e.g., corrupt file), proceed to compute from scratch
            pass

    # 2. Compute/process the data from scratch.

    # Resolve full path: check if filepath is relative to INPUT_DIR
    full_path = filepath
    if not os.path.exists(full_path):
        potential_path = os.path.join(config.INPUT_DIR, filepath)
        if os.path.exists(potential_path):
            full_path = potential_path
        else:
            raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Load audio using soundfile
    try:
        audio_data, sample_rate = sf.read(full_path)
    except Exception as e:
        raise IOError(f"Failed to read {full_path}: {e}")

    # Convert to float32
    audio_data = audio_data.astype(np.float32)

    # Handle Stereo -> Mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if necessary
    if sample_rate != target_sr:
        num_samples = int(len(audio_data) * target_sr / sample_rate)
        # Use scipy.signal.resample for resampling
        audio_data = scipy.signal.resample(audio_data, num_samples)
        audio_data = audio_data.astype(np.float32)

    # Convert to Tensor (1, Time)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)

    # Pad or Truncate to target_len
    current_len = waveform.shape[1]
    if current_len > target_len:
        # Truncate
        waveform = waveform[:, :target_len]
    elif current_len < target_len:
        # Pad (pad last dimension on the right)
        pad_amount = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    # Save to cache for future runs
    np.save(cache_path, waveform.numpy())

    return waveform


class MelSpectrogram(torch.nn.Module):
    """
    Custom MelSpectrogram implementation to replace torchaudio.transforms.MelSpectrogram.
    """

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min=0.0, f_max=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        if f_max is None:
            f_max = sample_rate / 2.0

        # Create Mel Basis Matrix
        mel_basis = self._create_mel_basis(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Hann Window
        self.register_buffer("window", torch.hann_window(n_fft))

    def _create_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        n_freqs = n_fft // 2 + 1

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)

        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filters = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            start = bin_points[i]
            center = bin_points[i + 1]
            end = bin_points[i + 2]

            if center > start:
                filters[i, start:center] = (np.arange(start, center) - start) / (
                    center - start
                )

            if end > center:
                filters[i, center:end] = (end - np.arange(center, end)) / (end - center)

        return filters

    def forward(self, x):
        # Handle input shape: (Batch, 1, Time) or (1, Time)
        ndim = x.dim()
        if ndim == 3:
            x_stft_in = x.squeeze(1)
        elif ndim == 2:
            x_stft_in = x
        else:
            raise ValueError(f"Input dimension {ndim} not supported")

        stft = torch.stft(
            x_stft_in,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

        spec = torch.abs(stft).pow(2.0)

        # spec: (Batch, Freq, Frames)
        # mel_basis: (n_mels, Freq)
        # Output: (Batch, n_mels, Frames)
        melspec = torch.einsum("mf,bft->bmt", self.mel_basis, spec)

        if ndim == 3:
            melspec = melspec.unsqueeze(1)

        return melspec


class SpecAugment(torch.nn.Module):
    """
    Applies SpecAugment (Frequency and Time Masking) for regularization.
    Cite solution_lesson_node_00003.
    """

    def __init__(self, freq_mask_param=10, time_mask_param=30):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec):
        # spec shape: (1, n_mels, time) or (Batch, 1, n_mels, time)
        # We assume input is (1, n_mels, time) from dataset

        if self.freq_mask_param > 0:
            f = spec.shape[1]
            f_mask_len = random.randint(0, self.freq_mask_param)
            f_start = random.randint(0, f - f_mask_len)
            spec[:, f_start : f_start + f_mask_len, :] = 0

        if self.time_mask_param > 0:
            t = spec.shape[2]
            t_mask_len = random.randint(0, self.time_mask_param)
            t_start = random.randint(0, t - t_mask_len)
            spec[:, :, t_start : t_start + t_mask_len] = 0

        return spec


def get_featurizer():
    """
    Returns the custom MelSpectrogram transform.
    """
    return MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
