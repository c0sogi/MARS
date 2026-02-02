import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from library.configuration import Config
from library.utilities import set_seed


class AudioDataset(Dataset):
    """
    PyTorch Dataset for Audio Tagging.
    Holds preprocessed spectrograms in memory and applies augmentation during training.
    """

    def __init__(self, X, y, fnames, is_training=False, config=None):
        self.X = X
        self.y = y
        self.fnames = fnames
        self.is_training = is_training
        self.config = config

        # Initialize Augmentations
        if self.is_training and self.config:
            self.time_masking = torchaudio.transforms.TimeMasking(
                time_mask_param=config.SPECAUG_TIME_MASK
            )
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=config.SPECAUG_FREQ_MASK
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load spectrogram from memory
        # Shape in memory: (N, 1, n_mels, time) or (N, n_mels, time)
        # We ensure it is (1, n_mels, time) for the model
        spec = torch.from_numpy(self.X[idx])

        if spec.ndim == 2:
            spec = spec.unsqueeze(0)

        # Apply SpecAugment during training
        if self.is_training:
            # SpecAugment expects (channel, freq, time) or (freq, time)
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # Get label
        label = torch.from_numpy(self.y[idx]).float()

        return spec, label


def get_class_mapping(sample_submission_path):
    """
    Reads the sample submission file to determine the correct class order.
    """
    df = pd.read_csv(sample_submission_path)
    # The columns after 'fname' are the classes in order
    classes = df.columns[1:].tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def load_audio_and_transform(args):
    """
    Worker function to process a single audio file.
    Args:
        args: Tuple containing (filepath, input_root, config)
    Returns:
        numpy array of shape (1, n_mels, time)
    """
    filepath, input_root, config = args
    full_path = os.path.join(input_root, filepath)

    # Target length in samples
    target_len = config.SAMPLE_RATE * config.DURATION

    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception as e:
        # Fallback for read errors (should be rare given metadata filtering)
        # Return silent spectrogram
        n_frames = int(target_len / config.HOP_LENGTH) + 1
        return -80.0 * np.ones((1, config.N_MELS, n_frames), dtype=np.float32)

    # Resample if necessary
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Truncate to fixed duration
    current_len = waveform.shape[1]
    if current_len < target_len:
        padding = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # Convert to Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        f_min=config.FMIN,
        f_max=config.FMAX,
    )
    spec = mel_transform(waveform)

    # Convert to Log Scale (dB)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec)

    return spec.numpy().astype(np.float32)


def process_dataset(metadata_df, class_to_idx, config, input_root):
    """
    Processes all files in the metadata dataframe in parallel.
    """
    filepaths = metadata_df["filepath"].tolist()
    args_list = [(fp, input_root, config) for fp in filepaths]

    # Use ThreadPoolExecutor for I/O bound task
    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        results = list(executor.map(load_audio_and_transform, args_list))

    # Stack into a single array: (N, 1, 128, T)
    X = np.stack(results)

    # Process Labels
    num_samples = len(metadata_df)
    num_classes = len(class_to_idx)
    y = np.zeros((num_samples, num_classes), dtype=np.float32)

    if "labels" in metadata_df.columns:
        for i, label_str in enumerate(metadata_df["labels"]):
            if pd.isna(label_str) or label_str == "":
                continue
            labels = label_str.split(",")
            for lbl in labels:
                if lbl in class_to_idx:
                    y[i, class_to_idx[lbl]] = 1.0

    fnames = metadata_df["fname"].values

    return X, y, fnames


def get_data(subset, config, load_cached_data=True):
    """
    Retrieves data for a subset (train/val/test).
    Uses caching to avoid re-processing audio.
    """
    cache_dir = config.OUTPUT_ROOT
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_X = os.path.join(cache_dir, f"{subset}_X.npy")
    cache_path_y = os.path.join(cache_dir, f"{subset}_y.npy")
    cache_path_fnames = os.path.join(cache_dir, f"{subset}_fnames.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_path_X)
            and os.path.exists(cache_path_y)
            and os.path.exists(cache_path_fnames)
        ):
            print(f"Loading {subset} data from cache...")
            try:
                X = np.load(cache_path_X)
                y = np.load(cache_path_y)
                fnames = np.load(cache_path_fnames)
                return X, y, fnames
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Generate from scratch
    print(f"Processing {subset} data from scratch...")

    # Determine Metadata Path
    if subset == "train":
        meta_path = config.TRAIN_CSV
    elif subset == "val":
        meta_path = config.VAL_CSV
    elif subset == "test":
        meta_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown subset: {subset}")

    df = pd.read_csv(meta_path)

    # Handle Debug Mode
    if config.DEBUG:
        df = df.head(config.DEBUG_SUBSET_SIZE)

    # Get Class Mapping
    classes, class_to_idx = get_class_mapping(config.SAMPLE_SUBMISSION)

    # Process Audio and Labels
    X, y, fnames = process_dataset(df, class_to_idx, config, config.INPUT_ROOT)

    # 3. Save to cache
    print(f"Saving {subset} data to cache...")
    np.save(cache_path_X, X)
    np.save(cache_path_y, y)
    np.save(cache_path_fnames, fnames)

    return X, y, fnames


def get_dataloaders(config, load_cached_data=True):
    """
    Creates Training and Validation DataLoaders.
    """
    set_seed(config.SEED)

    # Load Data (Cached or Fresh)
    train_X, train_y, train_fnames = get_data("train", config, load_cached_data)
    val_X, val_y, val_fnames = get_data("val", config, load_cached_data)

    # Initialize Datasets
    train_dataset = AudioDataset(
        train_X, train_y, train_fnames, is_training=True, config=config
    )
    val_dataset = AudioDataset(
        val_X, val_y, val_fnames, is_training=False, config=config
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(config, load_cached_data=True):
    """
    Creates Test DataLoader.
    """
    set_seed(config.SEED)

    test_X, test_y, test_fnames = get_data("test", config, load_cached_data)

    test_dataset = AudioDataset(
        test_X, test_y, test_fnames, is_training=False, config=config
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
