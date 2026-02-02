import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior for transforms where possible
torch.manual_seed(Config.SEED)


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.

    Attributes:
        specs (np.ndarray): Pre-computed spectrograms (N, F, T).
        labels (np.ndarray): Labels (N,). None for test set.
        clip_names (list): List of filenames (for submission).
        transform (nn.Module): Augmentation transforms (SpecAugment).
        is_train (bool): Flag to enable/disable augmentation.
    """

    def __init__(self, specs, labels, clip_names, is_train=False):
        self.specs = specs
        self.labels = labels
        self.clip_names = clip_names
        self.is_train = is_train

        # Augmentations
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Load spectrogram: (F, T) -> (1, F, T) for CNN input
        spec = torch.tensor(self.specs[idx], dtype=torch.float32).unsqueeze(0)

        # Apply Augmentation (Training only)
        if self.is_train and Config.USE_SPECAUG:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # Instance Normalization
        # (x - mean) / (std + eps)
        # We normalize per sample to handle varying recording volumes/noise floors
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / (std + 1e-6)
        else:
            spec = spec - mean

        # Handle Labels
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec, label
        else:
            # For test set, return clip name for submission mapping
            return spec, self.clip_names[idx]


def compute_spectrogram(audio_path):
    """
    Reads audio file and computes Log-Mel Spectrogram.
    """
    try:
        # Read audio
        audio, sr = sf.read(audio_path)

        # Ensure correct sampling rate (though analysis says all are 2000Hz)
        if sr != Config.SR:
            # Simple resampling if needed, though unlikely based on analysis
            # For this specific task, we assume inputs are mostly correct or we'd use torchaudio.resample
            pass

        # Handle channels (flatten to mono)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Fix length (Pad or Truncate)
        target_len = int(Config.SR * Config.DURATION)
        current_len = len(audio)

        if current_len < target_len:
            pad_width = target_len - current_len
            audio = np.pad(audio, (0, pad_width), mode="constant")
        elif current_len > target_len:
            audio = audio[:target_len]

        # Convert to Tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Compute Mel Spectrogram
        # Note: We do this on CPU for preprocessing to save GPU memory for training,
        # or we could move to GPU if batch processing. Single item is fast enough on CPU.
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            power=Config.POWER,
            center=True,
        )

        spec = mel_transform(audio_tensor)

        # Convert to Log Scale (dB)
        # standard formula: 10 * log10(spec + eps) or using AmplitudeToDB
        # torchaudio's AmplitudeToDB handles power->db
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        spec_db = db_transform(spec)

        return spec_db.numpy()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return a zero tensor of expected shape in case of failure
        expected_frames = int(target_len / Config.HOP_LENGTH) + 1
        # Note: n_fft/hop_length logic in torchaudio might result in slight frame diff depending on 'center'
        # With center=True (default), frames = audio_len // hop_length + 1
        return np.zeros((Config.N_MELS, 201), dtype=np.float32)


def generate_dataset_arrays(df, input_dir):
    """
    Iterates through dataframe, loads audio, generates spectrograms.
    Returns stacked numpy arrays.
    """
    specs_list = []
    labels_list = []
    clip_names_list = []

    print(f"Processing {len(df)} files...")

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        spec = compute_spectrogram(full_path)
        specs_list.append(spec)
        clip_names_list.append(row["clip_name"])

        if "label" in row:
            labels_list.append(row["label"])

    specs_array = np.stack(specs_list)

    if labels_list:
        labels_array = np.array(labels_list, dtype=np.int64)
    else:
        labels_array = None

    return specs_array, labels_array, clip_names_list


def load_or_create_data(df, cache_path, input_dir, load_cached_data=True):
    """
    Loads data from .npz cache if available, otherwise computes it and saves to cache.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            specs = data["specs"]
            # Handle potential None for labels in test set
            if "labels" in data and data["labels"].shape != ():
                labels = data["labels"]
            else:
                labels = None
            clip_names = data["clip_names"].tolist()
            return specs, labels, clip_names
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    specs, labels, clip_names = generate_dataset_arrays(df, input_dir)

    # Save
    print(f"Saving data to {cache_path}...")
    if labels is not None:
        np.savez(cache_path, specs=specs, labels=labels, clip_names=clip_names)
    else:
        np.savez(cache_path, specs=specs, clip_names=clip_names)

    return specs, labels, clip_names


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Subset data
    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SUBSET_SIZE} samples per split.")
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

        # Modify cache paths for debug to avoid overwriting full cache
        train_cache = Config.TRAIN_CACHE.replace(".npz", "_debug.npz")
        val_cache = Config.VAL_CACHE.replace(".npz", "_debug.npz")
        test_cache = Config.TEST_CACHE.replace(".npz", "_debug.npz")
    else:
        train_cache = Config.TRAIN_CACHE
        val_cache = Config.VAL_CACHE
        test_cache = Config.TEST_CACHE

    # 1. Prepare Training Data
    print("Preparing Training Data...")
    train_specs, train_labels, train_names = load_or_create_data(
        train_df, train_cache, Config.INPUT_DIR, load_cached_data
    )
    train_dataset = WhaleDataset(train_specs, train_labels, train_names, is_train=True)

    # 2. Prepare Validation Data
    print("Preparing Validation Data...")
    val_specs, val_labels, val_names = load_or_create_data(
        val_df, val_cache, Config.INPUT_DIR, load_cached_data
    )
    val_dataset = WhaleDataset(val_specs, val_labels, val_names, is_train=False)

    # 3. Prepare Test Data
    print("Preparing Test Data...")
    test_specs, test_labels, test_names = load_or_create_data(
        test_df, test_cache, Config.INPUT_DIR, load_cached_data
    )
    test_dataset = WhaleDataset(test_specs, test_labels, test_names, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
