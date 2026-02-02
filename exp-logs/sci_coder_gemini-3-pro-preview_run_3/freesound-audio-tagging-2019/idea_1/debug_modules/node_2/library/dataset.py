import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from library.config import Config


def get_class_names():
    """
    Reads the sample submission file to retrieve the list of classes
    in the correct order.
    """
    ss_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    # The first column is 'fname', the rest are class labels
    return ss_df.columns[1:].tolist()


def get_metadata_df(mode, load_cached_data=True):
    """
    Loads metadata for the specified mode (train/val/test).
    Implements caching using Parquet files.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'target' column for train/val.
    """
    # Ensure cache directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORK_DIR, f"{mode}_metadata.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure target column is read back as numpy arrays/lists if it exists
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # 2. Compute from scratch
    if mode == "train":
        csv_path = Config.TRAIN_CSV
    elif mode == "val":
        csv_path = Config.VAL_CSV
    elif mode == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pd.read_csv(csv_path)

    # Process labels for train and val
    if mode in ["train", "val"]:
        classes = get_class_names()
        mlb = MultiLabelBinarizer(classes=classes)

        # Parse labels string to list
        df["label_list"] = df["labels"].apply(lambda x: x.split(","))

        # Transform to binary matrix
        targets = mlb.fit_transform(df["label_list"])

        # Store as a column of arrays (or lists) to save in parquet
        # We convert to list to ensure compatibility with parquet engines
        df["target"] = list(targets)

        # Drop intermediate column
        df = df.drop(columns=["label_list"])

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


class AudioDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.mode = mode

        # Audio parameters
        self.sr = Config.SR
        self.duration = Config.DURATION
        self.target_length = self.sr * self.duration

        # Spectrogram transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)

        # ImageNet Normalization
        # Input to this will be [3, H, W] in range [0, 1]
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # 1. Load Audio
        # torchaudio.load returns (waveform, sample_rate)
        # waveform shape: [channels, time]
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            # Fallback for read errors: return a silent tensor
            # This should not happen given the curated metadata, but good for safety
            waveform = torch.zeros(1, self.target_length)
            sr = self.sr

        # 2. Resample if necessary
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # 3. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Pad or Crop to fixed length
        current_len = waveform.shape[1]

        if current_len < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            # Crop
            if self.mode == "train":
                # Random crop
                start = torch.randint(0, current_len - self.target_length, (1,)).item()
            else:
                # Center crop
                start = (current_len - self.target_length) // 2

            waveform = waveform[:, start : start + self.target_length]

        # 5. Compute Log-Mel Spectrogram
        # Shape: [1, n_mels, time]
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)

        # 6. Min-Max Normalization to [0, 1]
        # We perform instance normalization.
        # Adding epsilon to avoid division by zero for silent clips.
        min_val = spec.min()
        max_val = spec.max()
        spec = (spec - min_val) / (max_val - min_val + 1e-6)

        # 7. Convert to 3 channels (for ImageNet models)
        # Shape: [3, n_mels, time]
        spec = spec.repeat(3, 1, 1)

        # 8. Normalize with ImageNet stats
        spec = self.normalize(spec)

        # 9. Return based on mode
        if self.mode in ["train", "val"]:
            # Retrieve target vector
            # In the dataframe, it might be stored as a numpy array or list
            target = torch.tensor(row["target"], dtype=torch.float32)
            return spec, target
        else:
            # For test, return image and filename
            return spec, fname


def get_dataloader(mode, load_cached_data=True, debug=False):
    """
    Creates a DataLoader for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsets the data for debugging.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Load metadata
    df = get_metadata_df(mode, load_cached_data)

    # Debug subset
    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Create Dataset
    dataset = AudioDataset(df, mode=mode)

    # DataLoader arguments
    shuffle = mode == "train"
    drop_last = mode == "train"

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )

    return loader
