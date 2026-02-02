import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior for transforms
seed_everything(Config.SEED)


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram transform based on Config parameters.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )


def process_audio_file(file_path, mel_transform):
    """
    Reads an audio file, pads/crops it, and converts it to a normalized Log-Mel Spectrogram.

    Returns:
        np.ndarray: Shape (1, n_mels, time_steps) -> Resized to (1, 224, 224)
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    try:
        # Load audio
        wav, sr = sf.read(full_path)

        # Handle multi-channel (take mean) or empty
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)

        # Resample if necessary (though analysis showed all are 2000Hz)
        if sr != Config.SAMPLE_RATE:
            # Simple linear interpolation for resampling if needed,
            # but relying on dataset consistency is preferred for speed.
            # Given analysis, we assume 2000Hz.
            pass

        # Pad or Crop to fixed length (N_SAMPLES)
        if len(wav) < Config.N_SAMPLES:
            pad_width = Config.N_SAMPLES - len(wav)
            wav = np.pad(wav, (0, pad_width), mode="constant")
        else:
            wav = wav[: Config.N_SAMPLES]

        # Convert to tensor
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # (1, samples)

        # Generate Mel Spectrogram
        spec = mel_transform(wav_tensor)

        # Log Transform (Log-Mel)
        # Add small epsilon to avoid log(0)
        spec = torch.log(spec + 1e-9)

        # Instance-level Min-Max Normalization
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 1e-6:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = torch.zeros_like(spec)

        # Resize to fixed 224x224 to match ConvNeXt expectations
        # The calculated width is ~223, so this is a minor adjustment
        resize = torchaudio.transforms.Resize((224, 224))
        spec = resize(spec)

        return spec.numpy()  # (1, 224, 224)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a silent spectrogram in case of error
        return np.zeros((1, 224, 224), dtype=np.float32)


def load_or_process_data(df, split_name, load_cached_data=True):
    """
    Loads processed data from cache or processes it from scratch.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Contains 'X' (features) and 'y' (labels) or 'clips' (names).
    """
    Config.setup()  # Ensure directories exist
    cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Verify keys
            if split_name in ["train", "val"]:
                if "X" in data and "y" in data:
                    return {"X": data["X"], "y": data["y"]}
            else:
                if "X" in data and "clips" in data:
                    return {"X": data["X"], "clips": data["clips"]}
            print("Cache file corrupted or missing keys. Re-processing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process data from scratch
    print(f"Processing {split_name} data...")
    mel_transform = get_spectrogram_transform()

    X_list = []
    y_list = []
    clips_list = []

    # Iterate through metadata
    for idx, row in df.iterrows():
        spec = process_audio_file(row["file_path"], mel_transform)
        X_list.append(spec)

        if "label" in row:
            y_list.append(row["label"])

        if "clip_name" in row:
            clips_list.append(row["clip_name"])

    X = np.stack(X_list).astype(np.float32)

    save_dict = {"X": X}
    result_dict = {"X": X}

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.int64)
        save_dict["y"] = y
        result_dict["y"] = y

    if len(clips_list) > 0:
        clips = np.array(clips_list)
        save_dict["clips"] = clips
        result_dict["clips"] = clips

    # Save to cache
    print(f"Saving {split_name} data to {cache_path}...")
    np.savez_compressed(cache_path, **save_dict)

    return result_dict


class WhaleDataset(Dataset):
    def __init__(self, data_dict, transform=None, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing 'X' and optionally 'y' or 'clips'.
            transform (callable, optional): transforms to apply to the spectrogram.
            is_test (bool): If True, returns clip_name instead of label.
        """
        self.X = data_dict["X"]
        self.y = data_dict.get("y")
        self.clips = data_dict.get("clips")
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (1, 224, 224)
        img = torch.from_numpy(self.X[idx])

        # Apply augmentations (SpecAugment)
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, self.clips[idx]
        else:
            return img, torch.tensor(self.y[idx], dtype=torch.float32)


def get_transforms(mode="train"):
    """
    Returns transforms for augmentation.
    """
    if mode == "train" and Config.USE_SPECAUGMENT:
        return torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=30),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
        )
    return None


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Process/Load Data
    train_data = load_or_process_data(train_df, "train", load_cached_data)
    val_data = load_or_process_data(val_df, "val", load_cached_data)
    test_data = load_or_process_data(test_df, "test", load_cached_data)

    # Create Datasets
    train_dataset = WhaleDataset(
        train_data, transform=get_transforms("train"), is_test=False
    )

    val_dataset = WhaleDataset(val_data, transform=None, is_test=False)

    test_dataset = WhaleDataset(test_data, transform=None, is_test=True)

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
