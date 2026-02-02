import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import Config

# Ensure deterministic behavior
torch.manual_seed(Config.SEED)


def get_audio_transforms():
    """
    Creates and returns the list of MelSpectrogram transforms and the AmplitudeToDB transform
    based on the configuration.
    """
    transforms = []
    # Create a transform for each window length (resolution)
    for win_len, n_fft in zip(Config.WIN_LENGTHS, Config.N_FFTS):
        melspec = T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        )
        transforms.append(melspec)

    # Standard Log-scaling
    to_db = T.AmplitudeToDB(stype="power", top_db=80)

    return transforms, to_db


def load_and_process_audio(filepath):
    """
    Loads an audio file, resamples it to the target sample rate, converts to mono,
    and pads/crops it to the fixed duration specified in Config.

    Args:
        filepath (str): Relative path to the audio file.

    Returns:
        torch.Tensor: Processed waveform of shape (1, NUM_SAMPLES).
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)

    # Handle missing files gracefully (though dataset validation passed)
    if not os.path.exists(full_path):
        return torch.zeros(1, Config.NUM_SAMPLES)

    try:
        waveform, sample_rate = torchaudio.load(full_path)
    except Exception:
        # Return silence on load failure
        return torch.zeros(1, Config.NUM_SAMPLES)

    # Resample if necessary
    if sample_rate != Config.SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed length (Center alignment)
    current_len = waveform.shape[1]
    target_len = Config.NUM_SAMPLES

    if current_len < target_len:
        # Pad with zeros
        pad_amount = target_len - current_len
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        waveform = torch.nn.functional.pad(
            waveform, (pad_left, pad_right), mode="constant", value=0
        )
    elif current_len > target_len:
        # Center crop
        start = (current_len - target_len) // 2
        waveform = waveform[:, start : start + target_len]

    return waveform


def compute_multires_spectrogram(waveform, transforms=None, to_db=None):
    """
    Generates a 3-channel Multi-Resolution Log-Mel Spectrogram from the waveform.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        transforms (list, optional): List of MelSpectrogram transforms.
        to_db (callable, optional): AmplitudeToDB transform.

    Returns:
        torch.Tensor: 3-channel tensor of shape (3, N_MELS, TIME_STEPS).
    """
    if transforms is None or to_db is None:
        transforms, to_db = get_audio_transforms()

    specs = []
    for t in transforms:
        # Compute Mel Spectrogram: (1, n_mels, time)
        spec = t(waveform)
        # Convert to Log Scale
        spec = to_db(spec)
        specs.append(spec)

    # Stack along the channel dimension -> (3, n_mels, time)
    multi_res_spec = torch.cat(specs, dim=0)

    # Ensure strict time dimension consistency
    # For 16000 samples and 160 hop, we expect ~101 frames.
    # We fix it to 101 to handle minor padding differences.
    target_frames = 101
    current_frames = multi_res_spec.shape[2]

    if current_frames > target_frames:
        multi_res_spec = multi_res_spec[:, :, :target_frames]
    elif current_frames < target_frames:
        pad = target_frames - current_frames
        multi_res_spec = torch.nn.functional.pad(multi_res_spec, (0, pad))

    return multi_res_spec


def cache_dataset(df, cache_dir, load_cached_data=True):
    """
    Iterates through the metadata DataFrame, processes audio files into Multi-Resolution
    Spectrograms, and saves them as .npy files.

    Args:
        df (pd.DataFrame): Metadata DataFrame containing 'filepath'.
        cache_dir (str): Directory to save cached files.
        load_cached_data (bool): If True, skips processing for existing files.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'cache_path' column.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize transforms once to avoid overhead
    transforms, to_db = get_audio_transforms()

    cache_paths = []

    # Iterate through metadata
    for _, row in df.iterrows():
        filepath = row["filepath"]

        # Generate a unique filename hash
        file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        save_path = os.path.join(cache_dir, f"{file_hash}.npy")

        # Check if we can skip processing
        if load_cached_data and os.path.exists(save_path):
            cache_paths.append(save_path)
            continue

        # Compute features
        waveform = load_and_process_audio(filepath)
        spec_tensor = compute_multires_spectrogram(waveform, transforms, to_db)

        # Save to disk
        np.save(save_path, spec_tensor.numpy())
        cache_paths.append(save_path)

    # Return updated DataFrame
    df_result = df.copy()
    df_result["cache_path"] = cache_paths
    return df_result
