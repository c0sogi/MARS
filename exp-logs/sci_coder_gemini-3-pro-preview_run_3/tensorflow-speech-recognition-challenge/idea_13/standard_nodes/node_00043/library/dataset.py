import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    INPUT_ROOT,
    N_SAMPLES,
    SAMPLE_RATE,
    WORKING_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    BACKGROUND_NOISE_DIR,
)
from library.utils import LabelEncoder, set_seed


def load_audio_file(filepath, target_length=N_SAMPLES):
    """
    Loads an audio file and pads/crops it to the target length.
    """
    full_path = os.path.join(INPUT_ROOT, filepath)
    try:
        # Load using soundfile for speed
        wav, sr = sf.read(full_path)

        # Ensure float32
        wav = wav.astype(np.float32)

        # Handle multi-channel (take mean)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Pad or Crop
        curr_len = len(wav)
        if curr_len == target_length:
            return wav
        elif curr_len < target_length:
            # Pad at the end
            pad_width = target_length - curr_len
            return np.pad(wav, (0, pad_width), mode="constant")
        else:
            # Center crop
            start = (curr_len - target_length) // 2
            return wav[start : start + target_length]

    except Exception as e:
        # Return silence in case of error
        return np.zeros(target_length, dtype=np.float32)


class BackgroundNoiseDataset(Dataset):
    """
    Dataset for loading background noise clips.
    """

    def __init__(self, cache_dir=os.path.join(WORKING_DIR, "cache")):
        self.files = glob.glob(os.path.join(BACKGROUND_NOISE_DIR, "*.wav"))
        self.data = []
        os.makedirs(cache_dir, exist_ok=True)

        for f in self.files:
            try:
                wav, sr = sf.read(f)
                wav = wav.astype(np.float32)
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                self.data.append(wav)
            except:
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SpeechDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        mode="train",
        load_cached_data=True,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
            cache_dir (str): Directory to store cached data.
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_path)
        self.label_encoder = LabelEncoder()

        # Prepare cache paths
        os.makedirs(cache_dir, exist_ok=True)
        # Hash metadata path or use mode to differentiate caches
        cache_name = os.path.splitext(os.path.basename(metadata_path))[0]
        self.data_cache_path = os.path.join(cache_dir, f"{cache_name}_data.npy")
        self.targets_cache_path = os.path.join(cache_dir, f"{cache_name}_targets.npy")

        self.data = None
        self.targets = None
        self.silence_map = {}  # Stores full waveforms for silence files

        # Identify silence indices
        if "label" in self.df.columns:
            self.silence_indices = set(
                self.df[self.df["label"] == "silence"].index.tolist()
            )
        else:
            self.silence_indices = set()

        # Attempt to load cache
        loaded = False
        if (
            load_cached_data
            and os.path.exists(self.data_cache_path)
            and os.path.exists(self.targets_cache_path)
        ):
            try:
                self.data = np.load(self.data_cache_path)
                self.targets = np.load(self.targets_cache_path)
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self._process_and_cache()

        # Post-loading: Load full silence waveforms into memory for dynamic cropping
        # We do this regardless of cache to ensure we have the raw data for silence
        if mode == "train":
            for idx in self.silence_indices:
                filepath = self.df.iloc[idx]["filepath"]
                full_path = os.path.join(INPUT_ROOT, filepath)
                try:
                    wav, _ = sf.read(full_path)
                    wav = wav.astype(np.float32)
                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)
                    self.silence_map[idx] = wav
                except:
                    # Fallback to zeros of length N_SAMPLES * 2 to allow some cropping
                    self.silence_map[idx] = np.zeros(N_SAMPLES * 2, dtype=np.float32)

    def _process_and_cache(self):
        """
        Loads all audio files, processes them to fixed length, and saves to disk.
        For silence files, we store a placeholder (zeros) in the cache array,
        as they are handled dynamically in __getitem__.
        """
        n_samples = len(self.df)
        data_array = np.zeros((n_samples, N_SAMPLES), dtype=np.float32)
        targets_list = []

        for idx, row in self.df.iterrows():
            label = row["label"]

            # Encode target
            if self.mode == "test":
                target = -1  # Dummy
            else:
                target = self.label_encoder.encode(label)
            targets_list.append(target)

            # Process Audio
            if idx in self.silence_indices:
                # Placeholder for silence, will be loaded dynamically
                pass
            else:
                # Load and fix length
                wav = load_audio_file(row["filepath"])
                data_array[idx] = wav

        self.data = data_array
        self.targets = np.array(targets_list, dtype=np.int64)

        # Save to cache
        np.save(self.data_cache_path, self.data)
        np.save(self.targets_cache_path, self.targets)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Handle Silence (Dynamic Cropping)
        if idx in self.silence_map:
            full_wav = self.silence_map[idx]
            if len(full_wav) > N_SAMPLES:
                start = np.random.randint(0, len(full_wav) - N_SAMPLES)
                wav = full_wav[start : start + N_SAMPLES]
            else:
                # Pad if shorter (unlikely for background noise files)
                pad_width = N_SAMPLES - len(full_wav)
                wav = np.pad(full_wav, (0, pad_width), mode="constant")
        else:
            # Load from fixed cache
            wav = self.data[idx]

        target = self.targets[idx]

        # Convert to torch tensor
        return torch.from_numpy(wav), torch.tensor(target, dtype=torch.long)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    Handles class imbalance in Train using WeightedRandomSampler.
    """
    set_seed(SEED)

    # 1. Create Datasets
    train_dataset = SpeechDataset(
        TRAIN_CSV, mode="train", load_cached_data=load_cached_data
    )
    val_dataset = SpeechDataset(VAL_CSV, mode="val", load_cached_data=load_cached_data)
    test_dataset = SpeechDataset(
        TEST_CSV, mode="test", load_cached_data=load_cached_data
    )

    # 2. Handle Class Imbalance for Training
    # Calculate weights: 1 / frequency
    targets = train_dataset.targets
    class_counts = np.bincount(targets, minlength=len(LabelEncoder().label_to_idx))

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[targets]

    # Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
