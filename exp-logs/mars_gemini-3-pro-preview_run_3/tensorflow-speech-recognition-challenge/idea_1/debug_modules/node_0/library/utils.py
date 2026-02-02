import os
import hashlib
import torch
import torchaudio
import numpy as np
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

    # Load audio
    # torchaudio.load returns (waveform, sample_rate)
    waveform, sample_rate = torchaudio.load(full_path)

    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)

    # Mix to mono if necessary (dataset is mostly mono, but robust code handles stereo)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

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


def get_featurizer():
    """
    Returns a MelSpectrogram transform configured with parameters from config.

    Returns:
        torchaudio.transforms.MelSpectrogram: The featurizer object.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
