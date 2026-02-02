import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class AudioDataset(Dataset):
    """
    PyTorch Dataset for Audio Classification.
    Serves Log-Mel Spectrograms and Multi-Hot Labels.
    """

    def __init__(self, X, y, fnames, transform=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features (N, F, T).
            y (np.ndarray): Targets (N, C).
            fnames (np.ndarray): Filenames.
            transform (callable, optional): Augmentation/Transform.
            is_test (bool): Whether this is the test set.
        """
        self.X = X
        self.y = y
        self.fnames = fnames
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load spectrogram data
        # Shape: (F, T)
        spec = self.X[idx]

        # Convert to Tensor
        spec = torch.from_numpy(spec).float()

        # Add channel dimension for CNN: (1, F, T)
        spec = spec.unsqueeze(0)

        # Apply transforms (e.g., SpecAugment) if provided
        if self.transform:
            spec = self.transform(spec)

        if self.is_test:
            # For test set, return spectrogram and filename
            return spec, self.fnames[idx]
        else:
            # For train/val, return spectrogram, label, and filename
            label = self.y[idx]
            label = torch.from_numpy(label).float()
            return spec, label, self.fnames[idx]


class TrainAugmentation(torch.nn.Module):
    """
    Applies SpecAugment (Time and Frequency Masking) for training regularization.
    """

    def __init__(self):
        super().__init__()
        # Parameters tuned for 128 Mel bins and ~1800 time steps
        self.time_masking = T.TimeMasking(time_mask_param=48)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=24)

    def forward(self, x):
        return self.freq_masking(self.time_masking(x))


def get_label_map():
    """
    Extracts the ordered list of 80 class names from the sample submission file.
    Returns:
        list: List of class names.
        dict: Mapping from class name to index.
    """
    df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    # Columns are: fname, Label1, Label2, ...
    # We skip 'fname' to get the class list
    labels = df.columns[1:].tolist()
    return labels, {label: i for i, label in enumerate(labels)}


def compute_spectrogram(args):
    """
    Worker function to process a single audio file.
    Reads audio, resamples, pads/truncates, and computes Log-Mel Spectrogram.

    Args:
        args (tuple): (filepath, input_dir)

    Returns:
        np.ndarray: Log-Mel Spectrogram of shape (F, T).
    """
    filepath, input_dir = args
    full_path = os.path.join(input_dir, filepath)

    # 1. Load Audio
    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception as e:
        print(f"Error loading {full_path}: {e}. Returning silent tensor.")
        # Return a silent spectrogram of correct shape
        n_steps = int(Config.SAMPLE_RATE * Config.DURATION / Config.HOP_LENGTH) + 1
        return np.zeros((Config.N_MELS, n_steps), dtype=np.float32)

    # 2. Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # 3. Mix to Mono (Average channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. Pad or Truncate to fixed duration
    target_len = Config.SAMPLE_RATE * Config.DURATION
    current_len = waveform.shape[1]

    if current_len < target_len:
        padding = target_len - current_len
        # Pad with zeros at the end
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_len > target_len:
        # Truncate
        waveform = waveform[:, :target_len]

    # 5. Compute Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
        power=2.0,
    )

    mel_spec = mel_transform(waveform)

    # 6. Convert to Log Scale (dB)
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    log_mel_spec = db_transform(mel_spec)

    # Remove channel dim (1, F, T) -> (F, T) and convert to numpy
    return log_mel_spec.squeeze(0).numpy().astype(np.float32)


def process_subset(metadata_path, subset_name, label_map):
    """
    Loads metadata, processes all audio files in parallel, and encodes labels.

    Args:
        metadata_path (str): Path to the metadata CSV.
        subset_name (str): 'train', 'val', or 'test'.
        label_map (dict): Mapping from label string to index.

    Returns:
        tuple: (X, y, fnames) numpy arrays.
    """
    print(f"Processing {subset_name} data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Prepare arguments for parallel processing
    # filepath in metadata is relative to Config.INPUT_DIR
    file_args = [(row["filepath"], Config.INPUT_DIR) for _, row in df.iterrows()]

    # Process Audio in Parallel
    # Using 12 workers to utilize available vCPUs
    with ThreadPoolExecutor(max_workers=12) as executor:
        X_list = list(executor.map(compute_spectrogram, file_args))

    # Stack into a single array (N, F, T)
    X = np.stack(X_list)
    fnames = df["fname"].values

    # Process Labels
    num_classes = len(label_map)

    if "labels" in df.columns:
        # Multi-hot encoding
        y = np.zeros((len(df), num_classes), dtype=np.float32)
        for i, labels_str in enumerate(df["labels"]):
            if pd.isna(labels_str) or labels_str == "":
                continue
            lbls = labels_str.split(",")
            for lbl in lbls:
                if lbl in label_map:
                    y[i, label_map[lbl]] = 1.0
    else:
        # Dummy labels for test set
        y = np.zeros((len(df), num_classes), dtype=np.float32)

    return X, y, fnames


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Constructs DataLoaders for train, val, and test sets.
    Handles caching of preprocessed spectrograms to disk.

    Args:
        load_cached_data (bool): If True, attempts to load .npy files from cache.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    label_list, label_map = get_label_map()

    datasets = {}

    # Iterate over splits
    splits = [
        ("train", Config.TRAIN_META),
        ("val", Config.VAL_META),
        ("test", Config.TEST_META),
    ]

    for split, meta_path in splits:
        X_path = os.path.join(cache_dir, f"{split}_X.npy")
        y_path = os.path.join(cache_dir, f"{split}_y.npy")
        fnames_path = os.path.join(cache_dir, f"{split}_fnames.npy")

        # Check cache
        if (
            load_cached_data
            and os.path.exists(X_path)
            and os.path.exists(y_path)
            and os.path.exists(fnames_path)
        ):
            print(f"Loading cached {split} data from {cache_dir}...")
            X = np.load(X_path)
            y = np.load(y_path)
            fnames = np.load(fnames_path)
        else:
            # Process from scratch
            X, y, fnames = process_subset(meta_path, split, label_map)

            # Save to cache
            print(f"Caching {split} data to {cache_dir}...")
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(fnames_path, fnames)

        # Create Dataset
        # Apply augmentation only to training set
        is_test = split == "test"
        transform = TrainAugmentation() if split == "train" else None

        datasets[split] = AudioDataset(
            X, y, fnames, transform=transform, is_test=is_test
        )

    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
