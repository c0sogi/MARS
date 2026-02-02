import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class WhaleDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (Tensor): Preprocessed spectrogram images (N, C, H, W).
            labels (Tensor, optional): Labels (N,).
            transform (callable, optional): Augmentations.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Apply transforms (e.g., SpecAugment)
        if self.transform:
            image = self.transform(image)

        # Instance Normalization: (x - mean) / (std + eps)
        # This helps with varying recording levels and centers the data for the network
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / (std + 1e-6)
        else:
            image = image - mean

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram transform based on Config.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        center=True,
    )


def preprocess_dataset(df, root_dir, cache_name, load_cached_data=True):
    """
    Loads audio files, converts to spectrograms, resizes, and caches the result.

    Returns:
        data (Tensor): (N, 1, 224, 224)
        labels (Tensor): (N,) or None
    """
    cache_dir = Config.WORKING_DIR
    data_path = os.path.join(cache_dir, f"{cache_name}_data.npy")
    labels_path = os.path.join(cache_dir, f"{cache_name}_labels.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(data_path):
        print(f"Loading cached data from {data_path}...")
        try:
            data = np.load(data_path)
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
                return torch.from_numpy(data), torch.from_numpy(labels)
            else:
                return torch.from_numpy(data), None
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {len(df)} files for {cache_name}...")

    mel_transform = get_spectrogram_transform()
    db_transform = torchaudio.transforms.AmplitudeToDB()
    # Resize to 224x224 for ConvNeXt compatibility (standard ImageNet size)
    resize_transform = transforms.Resize((224, 224), antialias=True)

    data_list = []
    labels_list = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        try:
            # Load Audio
            wav, sr = sf.read(file_path)

            # Ensure correct length (pad or truncate)
            target_len = int(Config.SAMPLE_RATE * Config.DURATION)
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            else:
                wav = wav[:target_len]

            # Convert to Tensor (1, Time)
            waveform = torch.from_numpy(wav).float().unsqueeze(0)

            # Generate Spectrogram
            spec = mel_transform(waveform)
            spec = db_transform(spec)

            # Resize
            spec = resize_transform(spec)

            data_list.append(spec)

            if "label" in row:
                labels_list.append(row["label"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Use a zero tensor as fallback to keep alignment
            data_list.append(torch.zeros((1, 224, 224)))
            if "label" in row:
                labels_list.append(row["label"])

    # Stack into a single tensor
    data_tensor = torch.stack(data_list)
    data_np = data_tensor.numpy()

    # Save cache
    np.save(data_path, data_np)

    if labels_list:
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
        labels_np = labels_tensor.numpy()
        np.save(labels_path, labels_np)
        return data_tensor, labels_tensor
    else:
        return data_tensor, None


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        cache_suffix = "_debug"
    else:
        cache_suffix = ""

    # Preprocess Data (with Caching)
    train_data, train_labels = preprocess_dataset(
        train_df, Config.INPUT_ROOT, f"train{cache_suffix}", load_cached_data
    )
    val_data, val_labels = preprocess_dataset(
        val_df, Config.INPUT_ROOT, f"val{cache_suffix}", load_cached_data
    )
    test_data, _ = preprocess_dataset(
        test_df, Config.INPUT_ROOT, f"test{cache_suffix}", load_cached_data
    )

    # Augmentations for Training
    # SpecAugment: Frequency Masking and Time Masking
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35),
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, transform=train_transform)
    val_dataset = WhaleDataset(val_data, val_labels, transform=None)
    test_dataset = WhaleDataset(test_data, None, transform=None)

    # Weighted Random Sampler for Class Imbalance
    # Calculate weights based on class frequencies
    targets = train_labels.long().numpy()
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
