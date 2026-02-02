import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from library.config import (
    INPUT_ROOT,
    WORK_DIR,
    METADATA_DIR,
    SR,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    NORMALIZED,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG,
    DEBUG_SAMPLES,
    N_FOLDS,
)


# Ensure reproducibility
def seed_everything(seed=SEED):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


class AudioPreprocessor:
    """
    Handles loading, padding, spectrogram generation, and normalization.
    """

    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            normalized=NORMALIZED,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )
        self.target_length = int(SR * 2.0)  # 4000 samples for 2.0s at 2000Hz

    def process(self, file_path):
        # Construct full path
        full_path = os.path.join(INPUT_ROOT, file_path)

        try:
            wav, sr = sf.read(full_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            wav = np.zeros(self.target_length)

        # Pad or crop to fixed length
        if len(wav) < self.target_length:
            pad_width = self.target_length - len(wav)
            wav = np.pad(wav, (0, pad_width), mode="constant")
        else:
            wav = wav[: self.target_length]

        # Convert to tensor: (1, T)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)

        # Mel Spectrogram: (1, F, T)
        spec = self.mel_transform(wav_tensor)

        # Log Scale
        spec = self.db_transform(spec)

        # Instance Standardization (Zero-Mean, Unit-Variance per clip)
        mean = spec.mean()
        std = spec.std()
        if std > 1e-6:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        return spec.numpy()


def get_data(load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    """
    suffix = "_debug" if DEBUG else ""
    train_data_path = os.path.join(WORK_DIR, f"train_data{suffix}.npy")
    train_labels_path = os.path.join(WORK_DIR, f"train_labels{suffix}.npy")
    test_data_path = os.path.join(WORK_DIR, f"test_data{suffix}.npy")
    test_clips_path = os.path.join(WORK_DIR, f"test_clips{suffix}.npy")

    # Check if files exist
    files_exist = (
        os.path.exists(train_data_path)
        and os.path.exists(train_labels_path)
        and os.path.exists(test_data_path)
        and os.path.exists(test_clips_path)
    )

    if load_cached_data and files_exist:
        print("Loading cached data...")
        train_data = np.load(train_data_path)
        train_labels = np.load(train_labels_path)
        test_data = np.load(test_data_path)
        test_clips = np.load(test_clips_path)
        return train_data, train_labels, test_data, test_clips

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Combine Train and Val for Cross-Validation
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Debug Subsampling
    if DEBUG:
        print(f"Debug mode: Sampling {DEBUG_SAMPLES} rows.")
        full_train_df = full_train_df.sample(
            n=min(DEBUG_SAMPLES, len(full_train_df)), random_state=SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(DEBUG_SAMPLES, len(test_df)), random_state=SEED
        ).reset_index(drop=True)

    preprocessor = AudioPreprocessor()

    # Process Train
    train_data_list = []
    train_labels_list = []

    print("Processing training files...")
    for idx, row in full_train_df.iterrows():
        spec = preprocessor.process(row["file_path"])
        train_data_list.append(spec)
        train_labels_list.append(row["label"])

    train_data = np.stack(train_data_list)
    train_labels = np.array(train_labels_list, dtype=np.float32)

    # Process Test
    test_data_list = []
    test_clips_list = []

    print("Processing test files...")
    for idx, row in test_df.iterrows():
        spec = preprocessor.process(row["file_path"])
        test_data_list.append(spec)
        test_clips_list.append(row["clip"])

    test_data = np.stack(test_data_list)
    test_clips = np.array(test_clips_list)

    # Save to cache
    np.save(train_data_path, train_data)
    np.save(train_labels_path, train_labels)
    np.save(test_data_path, test_data)
    np.save(test_clips_path, test_clips)

    print(f"Data processed and saved to {WORK_DIR}")
    return train_data, train_labels, test_data, test_clips


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, training=False, clips=None):
        self.data = data
        self.labels = labels
        self.clips = clips
        self.training = training

        # SpecAugment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already (1, F, T) float32
        x = torch.from_numpy(self.data[idx])

        if self.training:
            x = self.time_mask(x)
            x = self.freq_mask(x)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)  # (1,)
            return x, y
        else:
            # Test mode, return clip name
            return x, self.clips[idx]


def get_dataloaders(fold=0, load_cached_data=True):
    """
    Returns train, val, and test dataloaders for a specific fold.
    """
    train_data, train_labels, test_data, test_clips = get_data(load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the requested fold
    # skf.split requires X and y, but X can be zeros for the purpose of splitting
    splits = list(skf.split(np.zeros(len(train_labels)), train_labels))
    train_idx, val_idx = splits[fold]

    # Subset data
    X_train = train_data[train_idx]
    y_train = train_labels[train_idx]
    X_val = train_data[val_idx]
    y_val = train_labels[val_idx]

    # Weighted Random Sampler for Train to handle imbalance
    # Calculate weights based on class counts in the training fold
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / (class_counts + 1e-6)  # Add epsilon to avoid div by zero
    sample_weights = class_weights[y_train.astype(int)]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_train), replacement=True
    )

    # Datasets
    train_dataset = WhaleDataset(X_train, y_train, training=True)
    val_dataset = WhaleDataset(X_val, y_val, training=False)
    test_dataset = WhaleDataset(test_data, clips=test_clips, training=False)

    # DataLoaders
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
