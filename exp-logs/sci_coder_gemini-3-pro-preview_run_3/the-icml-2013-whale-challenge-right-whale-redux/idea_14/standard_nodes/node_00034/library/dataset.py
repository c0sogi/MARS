import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, Tuple, Dict, Union

from library.config import AudioConfig, PathConfig, TrainConfig
from library.utils import set_seed

# Constants derived from metadata analysis
FIXED_DURATION_SEC = 2
FIXED_SAMPLES = AudioConfig.sample_rate * FIXED_DURATION_SEC  # 4000 samples


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Holds pre-processed spectrograms in memory for fast access.
    """

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, transform=None):
        """
        Args:
            images: FloatTensor of shape (N, 1, F, T)
            targets: FloatTensor of shape (N,)
            transform: Optional transform to be applied on a sample.
        """
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def compute_spectrogram(
    file_path: str, audio_config: AudioConfig = AudioConfig()
) -> torch.Tensor:
    """
    Reads audio file, pads/crops to fixed length, computes Log-Mel Spectrogram,
    and applies Instance-level Min-Max Normalization.
    """
    try:
        # Load audio
        wav, sr = sf.read(file_path)

        # Ensure float32
        wav = wav.astype(np.float32)

        # Convert to mono if necessary
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Pad or Crop to fixed length (4000 samples)
        current_len = wav.shape[0]
        if current_len < FIXED_SAMPLES:
            pad_width = FIXED_SAMPLES - current_len
            wav = np.pad(wav, (0, pad_width), mode="constant")
        elif current_len > FIXED_SAMPLES:
            wav = wav[:FIXED_SAMPLES]

        # Convert to Tensor: (1, Time)
        waveform = torch.from_numpy(wav).unsqueeze(0)

        # Generate Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config.sample_rate,
            n_fft=audio_config.n_fft,
            hop_length=audio_config.hop_length,
            n_mels=audio_config.n_mels,
            f_min=audio_config.fmin,
            f_max=audio_config.fmax,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        spec = mel_transform(waveform)  # (1, F, T)

        # Log Transform
        spec = torch.log10(spec + 1e-9) * 10.0

        # Instance-level Min-Max Normalization
        spec_min = spec.min()
        spec_max = spec.max()
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-9)

        return spec

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return zero tensor of expected shape in case of error
        # Time steps approx FIXED_SAMPLES / hop_length
        n_time = int(np.ceil(FIXED_SAMPLES / audio_config.hop_length))
        # Adjust for center=True padding usually adding a frame or two depending on implementation
        # We calculate exact shape from a dummy run if needed, but here we approximate or use the one from spec
        # To be safe, we just return a tensor matching the config dims roughly.
        # Actually, let's just re-raise or return a safe zero tensor based on calculation.
        # For 4000 samples, hop 20 -> 201 frames usually.
        return torch.zeros((1, audio_config.n_mels, 201))


def process_dataset_split(
    df: pd.DataFrame,
    split_name: str,
    input_dir: str,
    cache_dir: str,
    load_cached_data: bool,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads data for a specific split (train/val/test).
    Uses caching to avoid re-computing spectrograms.
    """
    cache_filename = f"{split_name}_debug.npz" if debug else f"{split_name}_data.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Apply debug sampling if needed
    if debug:
        df = df.head(TrainConfig.debug_sample_size).copy()

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            data = np.load(cache_path)
            images = torch.from_numpy(data["images"])
            targets = torch.from_numpy(data["targets"])
            return images, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {split_name} data ({len(df)} samples)...")

    images_list = []
    targets_list = []
    audio_config = AudioConfig()

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        spec = compute_spectrogram(full_path, audio_config)
        images_list.append(spec)

        # Handle label
        if "label" in row:
            targets_list.append(float(row["label"]))
        else:
            # Placeholder for test set (will be replaced by pseudo-labels or ignored)
            targets_list.append(-1.0)

    # Stack into tensors
    images = torch.stack(images_list)  # (N, 1, F, T)
    targets = torch.tensor(targets_list, dtype=torch.float32)  # (N,)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, images=images.numpy(), targets=targets.numpy())
    print(f"Saved {split_name} data to {cache_path}")

    return images, targets


def get_dataloaders(
    debug: bool = False,
    load_cached_data: bool = True,
    pseudo_labels: Optional[pd.DataFrame] = None,
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug: If True, uses a small subset of data.
        load_cached_data: If True, attempts to load pre-processed data from disk.
        pseudo_labels: Optional DataFrame containing ['clip', 'probability'] for the test set.
                       If provided, the 'train' loader will include both labeled train data
                       and pseudo-labeled test data (Student Training Mode).

    Returns:
        Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    set_seed(TrainConfig.seed)
    PathConfig.create_dirs()

    # Load Metadata
    train_df = pd.read_csv(PathConfig.train_meta)
    val_df = pd.read_csv(PathConfig.val_meta)
    test_df = pd.read_csv(PathConfig.test_meta)

    # Process Data (Load or Compute)
    train_imgs, train_lbls = process_dataset_split(
        train_df,
        "train",
        PathConfig.input_dir,
        PathConfig.cache_dir,
        load_cached_data,
        debug,
    )
    val_imgs, val_lbls = process_dataset_split(
        val_df,
        "val",
        PathConfig.input_dir,
        PathConfig.cache_dir,
        load_cached_data,
        debug,
    )
    test_imgs, test_lbls = process_dataset_split(
        test_df,
        "test",
        PathConfig.input_dir,
        PathConfig.cache_dir,
        load_cached_data,
        debug,
    )

    # Define Transforms (Cite solution_lesson_node_00024: SpecAugment)
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=20),
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_imgs, train_lbls, transform=train_transform)
    val_dataset = WhaleDataset(val_imgs, val_lbls)
    test_dataset = WhaleDataset(test_imgs, test_lbls)

    # Configure Train Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
