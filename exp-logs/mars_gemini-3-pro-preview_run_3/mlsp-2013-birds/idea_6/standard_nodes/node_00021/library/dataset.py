import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import log_message

# Ensure torchaudio uses the correct backend if necessary, though defaults are usually fine
torchaudio.set_audio_backend("soundfile")


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles loading cached spectrograms, applying augmentations, and formatting for ResNet.
    """

    def __init__(self, metadata_df, phase="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'rec_id' and 'labels'.
            phase (str): 'train', 'val', or 'test'. Controls augmentations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform
        self.num_classes = Config.NUM_CLASSES

        # Pre-define augmentations
        # Brightness and Contrast Jitter for photometric augmentation
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

        # Normalization for ImageNet pre-trained models
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        rec_id = row["rec_id"]

        # Load cached spectrogram: shape (1, 224, 224)
        cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cached file not found: {cache_path}. Run cache_data() first."
            )

        spec = np.load(cache_path)  # Shape: (1, 224, 224)
        spec_tensor = torch.from_numpy(spec).float()

        # 1. Augmentation: Time Shifting (Train only)
        # Random horizontal roll
        if self.phase == "train":
            shift = np.random.randint(0, spec_tensor.shape[-1])
            spec_tensor = torch.roll(spec_tensor, shifts=shift, dims=-1)

        # 2. Channel Replication: 1 -> 3
        # Replicate to match ResNet input expectation
        image = spec_tensor.repeat(3, 1, 1)  # Shape: (3, 224, 224)

        # 3. Augmentation: Photometric Jitter (Train only)
        if self.phase == "train":
            image = self.jitter(image)

        # 4. Normalization
        # ImageNet normalization
        image = self.normalize(image)

        # 5. Labels
        label_str = str(row["labels"])
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        if label_str != "?" and label_str != "nan":
            try:
                indices = [int(x) for x in label_str.split()]
                label_vec[indices] = 1.0
            except ValueError:
                pass  # Handle empty or malformed labels gracefully

        return image, label_vec, rec_id


def process_audio(file_path):
    """
    Loads audio, computes Log-Mel Spectrogram, and resizes it.
    Returns a numpy array of shape (1, 224, 224).
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Load audio
    waveform, sr = torchaudio.load(full_path)

    # Resample if necessary
    if sr != Config.SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.SR)
        waveform = resampler(waveform)

    # Ensure constant duration (pad or truncate)
    target_len = Config.SR * Config.DURATION
    current_len = waveform.shape[1]

    if current_len < target_len:
        pad_amt = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # Compute Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SR,
        n_mels=Config.N_MELS,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )
    mel_spec = mel_transform(waveform)

    # Convert to Log Scale (dB)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    log_mel = db_transform(mel_spec)

    # Resize to 224x224
    resize_transform = T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True)
    resized_spec = resize_transform(log_mel)

    # Min-Max Normalize to [0, 1] for stability before saving
    # This helps with subsequent augmentations and model stability
    min_val = resized_spec.min()
    max_val = resized_spec.max()
    if max_val - min_val > 1e-6:
        resized_spec = (resized_spec - min_val) / (max_val - min_val)
    else:
        resized_spec = torch.zeros_like(resized_spec)

    return resized_spec.numpy()


def prepare_cache(load_cached_data=True):
    """
    Iterates through all metadata files, processes audio, and caches spectrograms.
    """
    log_message("Preparing data cache...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load all metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    unique_files = all_data[["rec_id", "file_path"]].drop_duplicates()

    processed_count = 0
    skipped_count = 0

    for _, row in unique_files.iterrows():
        rec_id = row["rec_id"]
        file_path = row["file_path"]
        cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")

        if load_cached_data and os.path.exists(cache_path):
            skipped_count += 1
            continue

        try:
            spec = process_audio(file_path)
            np.save(cache_path, spec)
            processed_count += 1
        except Exception as e:
            log_message(f"Error processing {file_path}: {e}")

    log_message(
        f"Cache preparation complete. Processed: {processed_count}, Skipped (Cached): {skipped_count}"
    )


def get_cv_folds():
    """
    Merges Train and Validation sets and performs K-Fold Cross Validation.
    Returns a DataFrame with a 'fold' column.
    """
    # Load and merge train/val
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Combine
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare X (indices)
    X = full_train_df[["rec_id"]].values

    # K-Fold Cross Validation
    # We use KFold instead of IterativeStratification to ensure every sample is assigned to a fold.
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    full_train_df["fold"] = -1

    fold_idx = 0
    for _, test_index in kf.split(X):
        full_train_df.iloc[test_index, full_train_df.columns.get_loc("fold")] = fold_idx
        fold_idx += 1

    # Verify coverage
    if (full_train_df["fold"] == -1).any():
        log_message("Warning: Some samples were not assigned a fold!")

    return full_train_df


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Main entry point to get DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1) to be used as Validation.
        load_cached_data (bool): Whether to use existing cache.

    Returns:
        train_loader, val_loader
    """
    # 1. Ensure Cache Exists
    prepare_cache(load_cached_data=load_cached_data)

    # 2. Get Folds
    df = get_cv_folds()

    # 3. Split based on fold_idx
    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    # 4. Create Datasets
    train_dataset = BirdDataset(train_df, phase="train")
    val_dataset = BirdDataset(val_df, phase="val")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns DataLoader for the test set.
    """
    prepare_cache(load_cached_data=load_cached_data)

    test_df = pd.read_csv(Config.TEST_METADATA)
    test_dataset = BirdDataset(test_df, phase="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader
