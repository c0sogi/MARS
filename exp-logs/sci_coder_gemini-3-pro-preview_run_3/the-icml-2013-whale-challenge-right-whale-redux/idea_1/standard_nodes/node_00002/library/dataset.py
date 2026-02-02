import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Wraps pre-processed tensors for maximum efficiency.
    """

    def __init__(self, data, labels=None, clip_names=None, transform=None):
        """
        Args:
            data (Tensor): Shape (N, 1, n_mels, time)
            labels (Tensor, optional): Shape (N,)
            clip_names (list, optional): List of filenames corresponding to data
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.clip_names = clip_names
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            # For inference/test set, return clip name to map predictions
            name = self.clip_names[idx]
            return x, name


def process_audio(file_path, target_len, mel_transform, db_transform):
    """
    Reads and processes a single audio file into a normalized Log-Mel Spectrogram.
    """
    try:
        # Load audio
        # sf.read returns (data, samplerate)
        audio, sr = sf.read(file_path)

        # Handle channels (convert to mono if needed)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Pad or Truncate to fixed length
        if len(audio) < target_len:
            padding = target_len - len(audio)
            audio = np.pad(audio, (0, padding), "constant")
        elif len(audio) > target_len:
            audio = audio[:target_len]

        # Convert to Tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Compute Mel Spectrogram
        spec = mel_transform(audio_tensor)

        # Convert to Log Scale
        spec = db_transform(spec)

        # Instance-level Min-Max Normalization
        min_val = spec.min()
        max_val = spec.max()
        # Avoid division by zero
        spec = (spec - min_val) / (max_val - min_val + 1e-6)

        # Add channel dimension: (1, n_mels, time)
        spec = spec.unsqueeze(0)

        return spec

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero tensor of correct shape as fallback
        # Shape: (1, n_mels, time_steps)
        # Time steps = ceil(4000/128) approx 32
        # We calculate exact shape from a dummy pass if needed, but 32 is expected.
        return torch.zeros((1, Config.N_MELS, 32), dtype=torch.float32)


def load_or_generate_data(df, split_name, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    Uses .npz format to avoid pickle restrictions.
    """
    # Determine cache filename based on split and debug status
    debug_suffix = "_debug" if Config.DEBUG else ""
    cache_filename = f"{split_name}{debug_suffix}.npz"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            # allow_pickle=False ensures we are loading strictly numerical data
            loaded = np.load(cache_path, allow_pickle=False)

            data_np = loaded["data"]
            data_tensor = torch.from_numpy(data_np)

            labels_tensor = None
            if "labels" in loaded:
                labels_np = loaded["labels"]
                labels_tensor = torch.from_numpy(labels_np)

            return data_tensor, labels_tensor

        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch...")

    # Initialize transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    target_len = int(Config.SAMPLE_RATE * Config.DURATION)

    data_list = []
    label_list = []

    # Iterate through metadata
    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Process
        spec = process_audio(full_path, target_len, mel_transform, db_transform)
        data_list.append(spec)

        # Collect label if available
        if "label" in row:
            label_list.append(row["label"])

    # Stack into single tensor
    data_tensor = torch.stack(data_list)  # Shape: (N, 1, F, T)

    labels_tensor = None
    if label_list:
        labels_tensor = torch.tensor(label_list, dtype=torch.float32)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    save_dict = {"data": data_tensor.numpy()}
    if labels_tensor is not None:
        save_dict["labels"] = labels_tensor.numpy()

    np.savez(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return data_tensor, labels_tensor


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Apply Debug Subset if enabled
    if Config.DEBUG:
        print(f"DEBUG MODE: Limiting datasets to {Config.DEBUG_SUBSET_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # 2. Load/Process Data (with Caching)
    # Train
    train_data, train_labels = load_or_generate_data(
        train_df, "train", load_cached_data
    )

    # SpecAugment for Training
    train_transform = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
        torchaudio.transforms.TimeMasking(time_mask_param=10),
    )

    train_dataset = WhaleDataset(
        train_data,
        train_labels,
        clip_names=train_df["clip_name"].tolist(),
        transform=train_transform,
    )

    # Val
    val_data, val_labels = load_or_generate_data(val_df, "val", load_cached_data)
    val_dataset = WhaleDataset(
        val_data, val_labels, clip_names=val_df["clip_name"].tolist()
    )

    # Test
    test_data, _ = load_or_generate_data(test_df, "test", load_cached_data)
    test_dataset = WhaleDataset(
        test_data, labels=None, clip_names=test_df["clip_name"].tolist()
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
