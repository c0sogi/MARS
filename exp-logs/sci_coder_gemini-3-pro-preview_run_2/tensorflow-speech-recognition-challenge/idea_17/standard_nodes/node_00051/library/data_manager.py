import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    INPUT_ROOT,
    NUM_SAMPLES,
    LABEL2ID,
    WORKING_DIR,
)


def load_dataset_to_memory(load_cached_data=True, debug_size=None):
    """
    Loads the dataset into memory (CPU tensors).
    If load_cached_data is True and cache exists, loads from .npy files.
    Otherwise, processes audio files from scratch and saves to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_size (int, optional): If set, only load this many samples for debugging.

    Returns:
        dict: Dictionary containing:
            - train_waveforms (torch.Tensor): Shape (N_train, NUM_SAMPLES)
            - train_labels (torch.Tensor): Shape (N_train,)
            - val_waveforms (torch.Tensor): Shape (N_val, NUM_SAMPLES)
            - val_labels (torch.Tensor): Shape (N_val,)
            - test_waveforms (torch.Tensor): Shape (N_test, NUM_SAMPLES)
            - test_labels (torch.Tensor): Shape (N_test,)
            - background_noise (list of np.ndarray): Raw audio arrays of background noise.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_files = {
        "train_wav": os.path.join(WORKING_DIR, "train_waveforms.npy"),
        "train_lbl": os.path.join(WORKING_DIR, "train_labels.npy"),
        "val_wav": os.path.join(WORKING_DIR, "val_waveforms.npy"),
        "val_lbl": os.path.join(WORKING_DIR, "val_labels.npy"),
        "test_wav": os.path.join(WORKING_DIR, "test_waveforms.npy"),
        "test_lbl": os.path.join(WORKING_DIR, "test_labels.npy"),
        "bg_noise": os.path.join(WORKING_DIR, "background_noise.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading dataset from cache...")
        try:
            data = {}
            # Load main datasets
            data["train_waveforms"] = torch.from_numpy(
                np.load(cache_files["train_wav"])
            )
            data["train_labels"] = torch.from_numpy(np.load(cache_files["train_lbl"]))
            data["val_waveforms"] = torch.from_numpy(np.load(cache_files["val_wav"]))
            data["val_labels"] = torch.from_numpy(np.load(cache_files["val_lbl"]))
            data["test_waveforms"] = torch.from_numpy(np.load(cache_files["test_wav"]))
            data["test_labels"] = torch.from_numpy(np.load(cache_files["test_lbl"]))

            # Load background noise (saved as object array of arrays)
            data["background_noise"] = list(
                np.load(cache_files["bg_noise"], allow_pickle=True)
            )

            # Apply debug slicing if requested
            if debug_size is not None:
                print(f"Debug mode: Slicing dataset to {debug_size} samples.")
                data["train_waveforms"] = data["train_waveforms"][:debug_size]
                data["train_labels"] = data["train_labels"][:debug_size]
                data["val_waveforms"] = data["val_waveforms"][:debug_size]
                data["val_labels"] = data["val_labels"][:debug_size]
                data["test_waveforms"] = data["test_waveforms"][:debug_size]
                data["test_labels"] = data["test_labels"][:debug_size]

            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    # Process from scratch
    print("Processing dataset from scratch...")

    # Load Metadata
    df_train_all = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Separate background noise from training commands
    # Background noise files are typically long and handled separately for mixing
    is_bg = df_train_all["is_background"] == True
    df_bg = df_train_all[is_bg].copy()
    df_train = df_train_all[~is_bg].copy()

    # Apply debug slicing to DataFrames before processing to save time
    if debug_size is not None:
        print(f"Debug mode: Processing only {debug_size} samples per split.")
        df_train = df_train.iloc[:debug_size]
        df_val = df_val.iloc[:debug_size]
        df_test = df_test.iloc[:debug_size]
        # We generally keep all background noise as it's small (only ~6 files)

    def process_split(df):
        waveforms = []
        labels = []

        for _, row in df.iterrows():
            file_path = os.path.join(INPUT_ROOT, row["file_path"])
            label_str = row["label"]

            # Load Audio
            try:
                wav, sr = sf.read(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                wav = np.zeros(NUM_SAMPLES, dtype=np.float32)

            # Ensure Mono
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)

            # Pad/Trim to NUM_SAMPLES (16000)
            wav_len = len(wav)
            if wav_len < NUM_SAMPLES:
                pad_width = NUM_SAMPLES - wav_len
                # Pad at the end
                wav = np.pad(wav, (0, pad_width), mode="constant")
            elif wav_len > NUM_SAMPLES:
                # Center crop
                start = (wav_len - NUM_SAMPLES) // 2
                wav = wav[start : start + NUM_SAMPLES]

            waveforms.append(wav.astype(np.float32))
            labels.append(LABEL2ID.get(label_str, LABEL2ID["unknown"]))

        return np.array(waveforms, dtype=np.float32), np.array(labels, dtype=np.int64)

    print(f"Processing {len(df_train)} training samples...")
    train_wav, train_lbl = process_split(df_train)

    print(f"Processing {len(df_val)} validation samples...")
    val_wav, val_lbl = process_split(df_val)

    print(f"Processing {len(df_test)} test samples...")
    test_wav, test_lbl = process_split(df_test)

    print(f"Processing {len(df_bg)} background noise files...")
    bg_wavs = []
    for _, row in df_bg.iterrows():
        file_path = os.path.join(INPUT_ROOT, row["file_path"])
        try:
            wav, sr = sf.read(file_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            bg_wavs.append(wav.astype(np.float32))
        except Exception as e:
            print(f"Error reading background {file_path}: {e}")

    # Save to cache ONLY if not debugging (to avoid corrupting cache with partial data)
    if debug_size is None:
        print(f"Saving cache to {WORKING_DIR}...")
        np.save(cache_files["train_wav"], train_wav)
        np.save(cache_files["train_lbl"], train_lbl)
        np.save(cache_files["val_wav"], val_wav)
        np.save(cache_files["val_lbl"], val_lbl)
        np.save(cache_files["test_wav"], test_wav)
        np.save(cache_files["test_lbl"], test_lbl)
        # Save list of arrays as object array
        np.save(cache_files["bg_noise"], np.array(bg_wavs, dtype=object))

    data = {
        "train_waveforms": torch.from_numpy(train_wav),
        "train_labels": torch.from_numpy(train_lbl),
        "val_waveforms": torch.from_numpy(val_wav),
        "val_labels": torch.from_numpy(val_lbl),
        "test_waveforms": torch.from_numpy(test_wav),
        "test_labels": torch.from_numpy(test_lbl),
        "background_noise": bg_wavs,
    }

    return data
