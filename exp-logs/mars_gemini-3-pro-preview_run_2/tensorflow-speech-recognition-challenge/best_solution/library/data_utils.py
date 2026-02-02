import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_background_noise(load_cached_data=True):
    """
    Loads all background noise files, concatenates them into a single 1D tensor,
    and returns it. This tensor is used for on-the-fly noise injection and
    silence synthesis during training.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        torch.Tensor: A 1D float tensor containing all background noise concatenated.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "background_noise.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            noise_np = np.load(cache_path)
            # Ensure it's 1D
            if noise_np.ndim > 1:
                noise_np = noise_np.flatten()
            return torch.from_numpy(noise_np).float()
        except Exception as e:
            print(f"Failed to load cached background noise: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Loading background noise from source...")

    # We use train.csv to find background files
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    df = pd.read_csv(Config.TRAIN_CSV)
    # Filter for background noise files
    df_bg = df[df["is_background"] == True]

    if len(df_bg) == 0:
        print("Warning: No background noise files found in metadata.")
        return torch.zeros(Config.NUM_SAMPLES)

    noise_clips = []

    for _, row in df_bg.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            data, sr = sf.read(full_path)

            # Ensure mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # We do not truncate background noise here; we want the full length
            # to allow for random slicing during training augmentation.
            noise_clips.append(data.astype(np.float32))

        except Exception as e:
            print(f"Error reading {full_path}: {e}")

    if not noise_clips:
        # Fallback to a silent tensor if loading failed
        return torch.zeros(Config.NUM_SAMPLES)

    # Concatenate all noise clips into one long array
    full_noise = np.concatenate(noise_clips)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, full_noise)

    return torch.from_numpy(full_noise).float()


def load_all_data(split, load_cached_data=True):
    """
    Loads the entire dataset for a specific split into memory.
    Reads metadata, loads audio files, pads/truncates to fixed length,
    encodes labels, and returns tensors.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        tuple: (waveforms, labels)
            waveforms (torch.Tensor): Shape (N, NUM_SAMPLES)
            labels (torch.Tensor): Shape (N,)
    """
    # Define paths based on split
    if split == "train":
        csv_path = Config.TRAIN_CSV
        cache_wav_path = os.path.join(Config.WORKING_DIR, "train_waveforms.npy")
        cache_lbl_path = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    elif split == "val":
        csv_path = Config.VAL_CSV
        cache_wav_path = os.path.join(Config.WORKING_DIR, "val_waveforms.npy")
        cache_lbl_path = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    elif split == "test":
        csv_path = Config.TEST_CSV
        cache_wav_path = os.path.join(Config.WORKING_DIR, "test_waveforms.npy")
        cache_lbl_path = os.path.join(Config.WORKING_DIR, "test_labels.npy")
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_wav_path)
        and os.path.exists(cache_lbl_path)
    ):
        try:
            print(f"Loading cached {split} data...")
            waveforms_np = np.load(cache_wav_path)
            labels_np = np.load(cache_lbl_path)
            return (
                torch.from_numpy(waveforms_np).float(),
                torch.from_numpy(labels_np).long(),
            )
        except Exception as e:
            print(f"Failed to load cached {split} data: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    num_samples = len(df)
    target_len = Config.NUM_SAMPLES

    # Pre-allocate memory for efficiency
    waveforms_np = np.zeros((num_samples, target_len), dtype=np.float32)
    labels_np = np.zeros(num_samples, dtype=np.int64)

    # Iterate and load
    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        label_str = row["label"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        # Load Audio
        try:
            # sf.read returns (samples, channels) or (samples,)
            audio, sr = sf.read(full_path)

            # Ensure mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Pad or Truncate to fixed length
            curr_len = len(audio)
            if curr_len >= target_len:
                # Truncate
                waveforms_np[idx] = audio[:target_len]
            else:
                # Pad with zeros at the end
                waveforms_np[idx, :curr_len] = audio

        except Exception:
            # In case of file read error, leave as zeros (silence)
            pass

        # Encode Label
        # Use 'unknown' ID if label not found (e.g. if test set has placeholders)
        # Config.LABEL2ID contains 'unknown', 'silence', and all commands.
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID.get("unknown", 0))
        labels_np[idx] = label_id

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_wav_path, waveforms_np)
    np.save(cache_lbl_path, labels_np)

    return torch.from_numpy(waveforms_np).float(), torch.from_numpy(labels_np).long()
