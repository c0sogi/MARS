import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import (
    INPUT_DIR,
    BACKGROUND_NOISE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SAMPLE_RATE,
    DURATION,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MELS,
    LABEL2INT,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


class SpeechDataset(Dataset):
    def __init__(self, split, load_cached_data=True, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION
        self.num_samples = int(self.sample_rate * self.duration)

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(TRAIN_METADATA_PATH)
        elif split == "val":
            self.df = pd.read_csv(VAL_METADATA_PATH)
        elif split == "test":
            self.df = pd.read_csv(TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Debugging: Reduce dataset size
        if DEBUG:
            self.df = self.df.iloc[:DEBUG_SAMPLE_SIZE]

        # Cache Configuration
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        suffix = "_debug" if DEBUG else ""
        self.waveforms_path = os.path.join(
            self.cache_dir, f"{split}_waveforms{suffix}.npy"
        )
        self.labels_path = os.path.join(self.cache_dir, f"{split}_labels{suffix}.npy")

        # Load Background Noises (Train only)
        self.background_noises = {}
        if split == "train":
            self._load_background_noises()

        # Load Data (Waveforms and Labels)
        self._load_data(load_cached_data)

        # Feature Extraction Transforms
        # Note: We keep this on CPU as DataLoader workers are CPU-bound
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    def _load_background_noises(self):
        """Loads background noise files into memory for dynamic mixing."""
        if not os.path.exists(BACKGROUND_NOISE_DIR):
            return

        for filename in os.listdir(BACKGROUND_NOISE_DIR):
            if filename.endswith(".wav"):
                path = os.path.join(BACKGROUND_NOISE_DIR, filename)
                try:
                    wav, sr = sf.read(path)
                    # Ensure float32
                    wav = wav.astype(np.float32)
                    self.background_noises[filename] = wav
                except Exception as e:
                    print(f"Warning: Could not load background noise {filename}: {e}")

    def _load_data(self, load_cached_data):
        """Loads waveforms from cache or processes from scratch."""
        # Try loading from cache
        if (
            load_cached_data
            and os.path.exists(self.waveforms_path)
            and os.path.exists(self.labels_path)
        ):
            try:
                # Use mmap_mode='r' to avoid loading everything into RAM at once if not needed,
                # though 46k * 16k floats is ~3GB, which fits in RAM easily.
                self.waveforms = np.load(self.waveforms_path, mmap_mode="r")
                self.labels = np.load(self.labels_path)
                print(f"[{self.split}] Loaded cached data from {self.cache_dir}")
                return
            except Exception as e:
                print(f"[{self.split}] Cache load failed ({e}). Recomputing...")

        print(f"[{self.split}] Processing and caching audio files...")

        waveforms_list = []
        labels_list = []

        for _, row in self.df.iterrows():
            # 1. Process Label
            label_str = row["label"]
            label_int = LABEL2INT.get(label_str, LABEL2INT["unknown"])
            labels_list.append(label_int)

            # 2. Process Waveform
            # If this is a background noise sample in the training set, we store a placeholder.
            # The actual audio will be sampled dynamically in __getitem__.
            if self.split == "train" and row.get("is_background", False):
                waveforms_list.append(np.zeros(self.num_samples, dtype=np.float32))
            else:
                file_path = os.path.join(INPUT_DIR, row["file_path"])
                try:
                    wav, sr = sf.read(file_path)
                    wav = wav.astype(np.float32)

                    # Pad or Truncate to exactly 1 second
                    if len(wav) > self.num_samples:
                        wav = wav[: self.num_samples]
                    else:
                        pad_width = self.num_samples - len(wav)
                        wav = np.pad(wav, (0, pad_width), mode="constant")

                    waveforms_list.append(wav)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    waveforms_list.append(np.zeros(self.num_samples, dtype=np.float32))

        # Convert to numpy arrays
        self.waveforms = np.stack(waveforms_list)
        self.labels = np.array(labels_list, dtype=np.int64)

        # Save to cache
        np.save(self.waveforms_path, self.waveforms)
        np.save(self.labels_path, self.labels)
        print(f"[{self.split}] Data cached to {self.cache_dir}")

    def _get_random_background_segment(self):
        """Returns a random 1-second segment from the loaded background noises."""
        if not self.background_noises:
            return np.zeros(self.num_samples, dtype=np.float32)

        # Pick a random noise file
        name = np.random.choice(list(self.background_noises.keys()))
        noise = self.background_noises[name]

        if len(noise) <= self.num_samples:
            # Pad if shorter than 1s (unlikely for background files)
            segment = np.pad(noise, (0, max(0, self.num_samples - len(noise))))
            return segment[: self.num_samples]

        # Random crop
        start = np.random.randint(0, len(noise) - self.num_samples)
        segment = noise[start : start + self.num_samples]
        return segment

    def _augment_audio(self, waveform):
        """Mixes background noise with random SNR."""
        # 80% chance to apply noise mixing
        if np.random.random() < 0.8:
            noise = self._get_random_background_segment()

            signal_energy = np.sum(waveform**2)
            noise_energy = np.sum(noise**2)

            if noise_energy > 1e-9:
                # Random SNR between 0 and 15 dB
                snr_db = np.random.uniform(0, 15)
                target_noise_energy = signal_energy / (10 ** (snr_db / 10))
                scale = np.sqrt(target_noise_energy / (noise_energy + 1e-9))

                waveform = waveform + noise * scale
                waveform = np.clip(waveform, -1.0, 1.0)

        return waveform

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Logic for 'silence' class in Training:
        # If the label is silence, we generate a fresh random crop from background noise files.
        # This ensures we don't just learn the specific 1s clips defined in metadata.
        if self.split == "train" and label == LABEL2INT["silence"]:
            waveform = self._get_random_background_segment()
        else:
            # Load from cache (copy to avoid modifying mmap/cache)
            waveform = self.waveforms[idx].copy()

            # Apply Noise Injection Augmentation (Train only)
            if self.split == "train":
                waveform = self._augment_audio(waveform)

        # Convert to Tensor
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, Time)

        # 1. Log-Mel Spectrogram
        # Add small epsilon to log to avoid -inf
        mel_spec = self.mel_transform(waveform_tensor)
        log_mel_spec = torch.log(mel_spec + 1e-9)  # (1, n_mels, time)

        # 2. Instance Normalization
        # Standardize the single channel: (x - mean) / std
        # Cite solution_lesson_node_00038: Always apply Instance Norm for vision backbones
        means = log_mel_spec.mean(dim=(1, 2), keepdim=True)
        stds = log_mel_spec.std(dim=(1, 2), keepdim=True)
        image = (log_mel_spec - means) / (stds + 1e-5)

        return image, label

    def __len__(self):
        return len(self.labels)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Uses WeightedRandomSampler for Training to handle class imbalance.
    """
    # 1. Train Dataset & Loader
    train_dataset = SpeechDataset("train", load_cached_data=load_cached_data)

    # Calculate Class Weights for Balancing
    labels = train_dataset.labels
    # Count occurrences of each class index
    class_counts = np.bincount(labels, minlength=len(LABEL2INT))
    # Inverse frequency weights (add 1 to avoid div by zero if debug mode misses a class)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Validation Dataset & Loader
    val_dataset = SpeechDataset("val", load_cached_data=load_cached_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test Dataset & Loader
    test_dataset = SpeechDataset("test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
