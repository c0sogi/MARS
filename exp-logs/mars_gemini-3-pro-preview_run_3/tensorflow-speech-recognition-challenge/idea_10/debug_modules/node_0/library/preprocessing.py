import os
import hashlib
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from library.config import (
    INPUT_ROOT,
    CACHE_DIR,
    METADATA_DIR,
    SAMPLE_RATE,
    NUM_SAMPLES,
    MEL_SPECTROGRAM_CONFIGS,
    F_MIN,
    F_MAX,
    SEED,
)
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


def get_audio_path(rel_path):
    """
    Resolves the relative path from metadata to the absolute path in input directory.
    """
    return os.path.join(INPUT_ROOT, rel_path)


def load_and_pad_waveform(filepath):
    """
    Loads an audio file, converts to mono, resamples if needed,
    and pads/truncates to the fixed NUM_SAMPLES length.
    """
    try:
        # Load audio
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Return silent waveform in case of corruption
        return torch.zeros(1, NUM_SAMPLES)

    # Resample if sampling rate mismatches
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to Mono (average channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Truncate to NUM_SAMPLES
    channels, length = waveform.shape
    if length > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    elif length < NUM_SAMPLES:
        padding = NUM_SAMPLES - length
        # Pad the last dimension (time) on the right
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform


def compute_multires_melspec(waveform):
    """
    Computes a 3-Channel Multi-Resolution Log-Mel Spectrogram.

    Args:
        waveform (torch.Tensor): Input audio waveform of shape (1, Time).

    Returns:
        numpy.ndarray: Stacked spectrogram features of shape (3, n_mels, time).
    """
    specs = []

    # Iterate over the 3 configurations (Short, Medium, Long windows)
    for config in MEL_SPECTROGRAM_CONFIGS:
        # Define Mel Spectrogram transform
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=config["n_fft"],
            win_length=config["win_length"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            f_min=F_MIN,
            f_max=F_MAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        )

        # Define Amplitude to DB transform (Log-Mel)
        db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # Compute features
        # waveform: (1, T) -> spec: (1, n_mels, frames)
        spec = mel_transform(waveform)
        log_spec = db_transform(spec)

        specs.append(log_spec)

    # Stack along the channel dimension (dim 0)
    # Result shape: (3, n_mels, frames)
    multires_spec = torch.cat(specs, dim=0)

    return multires_spec.numpy()


def get_cache_filename(filepath):
    """
    Generates a deterministic hash filename based on the relative filepath.
    """
    # Normalize path to ensure consistency
    norm_path = os.path.normpath(filepath).replace(os.sep, "/")
    file_hash = hashlib.md5(norm_path.encode("utf-8")).hexdigest()
    return f"{file_hash}.npy"


def process_file(filepath, cache_dir, load_cached_data):
    """
    Checks for cached file; if missing or forced refresh, computes and saves it.

    Args:
        filepath (str): Relative path to the audio file.
        cache_dir (str): Directory to store cached files.
        load_cached_data (bool): Whether to attempt loading existing cache.
    """
    cache_filename = get_cache_filename(filepath)
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. IF load_cached_data is True: Try to load (check existence)
    if load_cached_data and os.path.exists(cache_path):
        return

    # 2. IF loading fails OR load_cached_data is False: Compute and Save
    full_path = get_audio_path(filepath)

    if os.path.exists(full_path):
        waveform = load_and_pad_waveform(full_path)
    else:
        # Fallback for missing source files (though validation checked this)
        waveform = torch.zeros(1, NUM_SAMPLES)

    features = compute_multires_melspec(waveform)

    # Save to cache
    np.save(cache_path, features)


def cache_dataset(load_cached_data=True):
    """
    Main entry point for preprocessing. Iterates over all metadata files
    and ensures features are cached.

    Args:
        load_cached_data (bool): If True, skips re-computation for existing files.
    """
    print("Starting offline feature extraction and caching...")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    metadata_files = ["train.csv", "val.csv", "test.csv"]
    total_processed = 0

    for meta_file in metadata_files:
        meta_path = os.path.join(METADATA_DIR, meta_file)
        if not os.path.exists(meta_path):
            print(f"Metadata file {meta_file} not found. Skipping.")
            continue

        df = pd.read_csv(meta_path)
        print(f"Processing {meta_file} with {len(df)} samples.")

        # Iterate through all files in the metadata
        for _, row in df.iterrows():
            process_file(row["filepath"], CACHE_DIR, load_cached_data)
            total_processed += 1

            if total_processed % 5000 == 0:
                print(f"  Processed {total_processed} files...")

    print(f"Preprocessing complete. Total files processed: {total_processed}")
