import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import set_seed


class SpeechCommandDataset(Dataset):
    """
    Dataset class for Speech Commands.
    Handles loading, caching, and basic length normalization (pad/crop).
    """

    def __init__(self, split, load_cached_data=True, debug=False, max_samples=None):
        self.split = split
        self.debug = debug
        self.target_length = int(Config.SAMPLE_RATE * Config.DURATION)

        # Select Metadata File
        if split == "train":
            csv_path = Config.TRAIN_CSV
        elif split == "val":
            csv_path = Config.VAL_CSV
        elif split == "test":
            csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.metadata = pd.read_csv(csv_path)

        # Apply Debug / Max Samples Limit
        if self.debug:
            limit = 1000 if max_samples is None else max_samples
            self.metadata = self.metadata.iloc[:limit]
        elif max_samples is not None:
            self.metadata = self.metadata.iloc[:max_samples]

        # Define Cache Path
        cache_name = f"{split}_data.npy"
        if self.debug:
            cache_name = f"debug_{cache_name}"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # Load Data (RAM Cache)
        self.data = self._load_data(load_cached_data)

        # Prepare Labels
        if "label" in self.metadata.columns:
            self.labels = self.metadata["label"].values
            # Map labels to IDs, defaulting to 'unknown' if not found
            self.label_ids = [
                Config.LABEL2ID.get(l, Config.LABEL2ID["unknown"]) for l in self.labels
            ]
        else:
            # Fallback for test if label col missing
            self.labels = ["unknown"] * len(self.metadata)
            self.label_ids = [Config.LABEL2ID["unknown"]] * len(self.metadata)

    def _load_data(self, load_cached):
        """
        Loads audio data into memory, using disk cache if available.
        """
        # 1. Try loading from cache
        if load_cached and os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                if len(data) == len(self.metadata):
                    return data
            except Exception as e:
                print(f"Warning: Failed to load cache {self.cache_path}: {e}")

        # 2. Process from scratch
        data_list = []

        for idx, row in self.metadata.iterrows():
            filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
            try:
                # Load audio
                audio, sr = sf.read(filepath)
                # Ensure float32
                audio = audio.astype(np.float32)
                data_list.append(audio)
            except Exception as e:
                # Fallback: silent array
                data_list.append(np.zeros(self.target_length, dtype=np.float32))

        data_array = np.array(data_list, dtype=object)

        # 3. Save to cache
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, data_array, allow_pickle=True)
        except Exception as e:
            print(f"Warning: Failed to save cache {self.cache_path}: {e}")

        return data_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        label_id = self.label_ids[idx]

        current_len = len(waveform)

        # Length Normalization
        if current_len == self.target_length:
            out_wave = waveform
        elif current_len > self.target_length:
            # Crop
            if self.split == "train":
                # Random crop for training (essential for long silence files)
                start = np.random.randint(0, current_len - self.target_length)
            else:
                # Center crop for validation/test
                start = (current_len - self.target_length) // 2
            out_wave = waveform[start : start + self.target_length]
        else:
            # Pad with zeros at the end
            pad_amt = self.target_length - current_len
            out_wave = np.pad(waveform, (0, pad_amt), mode="constant")

        # Return as Tensor (Time,)
        return torch.from_numpy(out_wave), torch.tensor(label_id, dtype=torch.long)


def load_background_noises(load_cached_data=True):
    """
    Loads all background noise files into a single concatenated tensor.
    Used for on-the-fly mixing augmentation.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "background_noise.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            noise_concat = np.load(cache_path)
            return torch.from_numpy(noise_concat)
        except:
            pass

    # Load from directory
    if not os.path.exists(Config.NOISE_DIR):
        return None

    noise_files = [
        os.path.join(Config.NOISE_DIR, f)
        for f in os.listdir(Config.NOISE_DIR)
        if f.endswith(".wav")
    ]

    noises = []
    for f in noise_files:
        try:
            y, sr = sf.read(f)
            noises.append(y.astype(np.float32))
        except:
            pass

    if not noises:
        return None

    # Concatenate into one long buffer
    noise_concat = np.concatenate(noises)

    # Save cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, noise_concat)
    except:
        pass

    return torch.from_numpy(noise_concat)


def get_dataloaders(load_cached_data=True, debug=False, max_samples=None):
    """
    Creates DataLoaders for train, val, and test splits.
    Applies WeightedRandomSampler to the training set to handle class imbalance.
    """
    # 1. Train Dataset
    train_ds = SpeechCommandDataset("train", load_cached_data, debug, max_samples)

    # Calculate Class Weights for Sampling
    labels = train_ds.labels
    label_series = pd.Series(labels)
    class_counts = label_series.value_counts()

    # Weight = 1 / count (Inverse frequency)
    weights_map = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [weights_map[l] for l in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler handles shuffling
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Validation Dataset
    val_ds = SpeechCommandDataset("val", load_cached_data, debug, max_samples)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test Dataset
    test_ds = SpeechCommandDataset("test", load_cached_data, debug, max_samples)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
