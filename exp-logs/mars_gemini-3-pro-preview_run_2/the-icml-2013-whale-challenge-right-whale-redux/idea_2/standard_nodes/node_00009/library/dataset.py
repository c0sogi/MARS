import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import seed_everything

# Set seeds for reproducibility
seed_everything(Config.SEED)


def get_spectrogram_transform(config):
    """
    Creates the MelSpectrogram transform pipeline.
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.FMIN,
        f_max=config.FMAX,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def load_audio_file(filepath, config):
    """
    Loads an audio file, pads/truncates it to the fixed duration, and returns the waveform.
    """
    target_len = int(config.SAMPLE_RATE * config.DURATION)

    try:
        # Load audio
        wav, sr = sf.read(filepath)

        # Handle multi-channel (though analysis says all are mono)
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)

        # Resample if necessary (analysis says all are 2000Hz)
        if sr != config.SAMPLE_RATE:
            # Simple linear interpolation for resampling if needed,
            # but relying on analysis that SR is consistent.
            # For robustness, we could use scipy or torchaudio resample,
            # but keeping it simple as per analysis.
            pass

        # Pad or Truncate
        current_len = len(wav)
        if current_len < target_len:
            pad_width = target_len - current_len
            # Pad with zeros at the end
            wav = np.pad(wav, (0, pad_width), mode="constant")
        elif current_len > target_len:
            wav = wav[:target_len]

        return torch.from_numpy(wav).float()

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return torch.zeros(target_len).float()


def process_and_cache_data(df, cache_name, config, load_cached_data=True):
    """
    Loads audio files listed in the dataframe, converts them to spectrograms,
    and caches the result to disk as a .npy file.

    Returns:
        data_tensor: Tensor of shape (N, 1, F, T)
        labels_tensor: Tensor of shape (N,)
        clips_list: List of filenames
    """
    # Construct cache filename
    debug_suffix = "_debug" if config.DEBUG else ""
    cache_path_data = os.path.join(
        config.WORKING_DIR, f"{cache_name}{debug_suffix}_data.npy"
    )
    cache_path_labels = os.path.join(
        config.WORKING_DIR, f"{cache_name}{debug_suffix}_labels.npy"
    )
    cache_path_clips = os.path.join(
        config.WORKING_DIR, f"{cache_name}{debug_suffix}_clips.npy"
    )

    # Check if cache exists and we want to load it
    if (
        load_cached_data
        and os.path.exists(cache_path_data)
        and os.path.exists(cache_path_labels)
        and os.path.exists(cache_path_clips)
    ):
        print(f"Loading cached data from {cache_path_data}...")
        data = np.load(cache_path_data)
        labels = np.load(cache_path_labels)
        clips = np.load(cache_path_clips)

        # Convert back to tensors
        data_tensor = torch.from_numpy(data)
        labels_tensor = torch.from_numpy(labels).float()
        return data_tensor, labels_tensor, clips.tolist()

    print(f"Processing {len(df)} samples for {cache_name}...")

    # Initialize transform
    transform = get_spectrogram_transform(config)

    data_list = []
    labels_list = []
    clips_list = []

    for idx, row in df.iterrows():
        # Construct full path
        file_path = os.path.join(config.INPUT_ROOT, row["file_path"])

        # Load and process audio
        waveform = load_audio_file(file_path, config)

        # Compute spectrogram
        # waveform shape: (Time,) -> need (1, Time) for transform?
        # torchaudio transform expects (..., Time)
        spec = transform(waveform)  # Shape: (n_mels, time)

        data_list.append(spec.numpy())

        # Handle label
        if "label" in row:
            labels_list.append(row["label"])
        else:
            labels_list.append(0)  # Dummy label for test

        # Handle clip name
        if "clip" in row:
            clips_list.append(row["clip"])
        else:
            # Fallback to filename from path
            clips_list.append(os.path.basename(row["file_path"]))

    # Stack data
    # Result shape: (N, n_mels, time)
    data_array = np.stack(data_list)
    # Add channel dimension: (N, 1, n_mels, time)
    data_array = data_array[:, np.newaxis, :, :]

    labels_array = np.array(labels_list)
    clips_array = np.array(clips_list)

    # Save to cache
    np.save(cache_path_data, data_array)
    np.save(cache_path_labels, labels_array)
    np.save(cache_path_clips, clips_array)
    print(f"Cached data saved to {config.WORKING_DIR}")

    return (
        torch.from_numpy(data_array),
        torch.from_numpy(labels_array).float(),
        clips_list,
    )


class WhaleDataset(Dataset):
    def __init__(self, data, labels, clips, is_train=False, config=None):
        """
        Args:
            data (Tensor): Spectrogram data (N, 1, F, T)
            labels (Tensor): Labels (N,)
            clips (list): List of clip filenames
            is_train (bool): Whether to apply augmentation
            config (Config): Configuration object
        """
        self.data = data
        self.labels = labels
        self.clips = clips
        self.is_train = is_train
        self.config = config

        # SpecAugment transforms
        if self.is_train and config:
            self.time_masking = torchaudio.transforms.TimeMasking(
                time_mask_param=config.SPECAUG_TIME_MASK_PARAM
            )
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=config.SPECAUG_FREQ_MASK_PARAM
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data
        spec = self.data[idx]  # (1, F, T)
        label = self.labels[idx]
        clip = self.clips[idx]

        # Apply Augmentation if training
        if self.is_train:
            # SpecAugment expects (channel, freq, time) or (freq, time)
            # We have (1, F, T)
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        return spec, label, clip


def get_dataloaders(config, load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.
    Handles caching and weighted sampling.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Handle Debug Mode
    if config.DEBUG:
        print(f"DEBUG MODE: Limiting datasets to {config.MAX_DEBUG_SAMPLES} samples.")
        train_df = train_df.head(config.MAX_DEBUG_SAMPLES)
        val_df = val_df.head(config.MAX_DEBUG_SAMPLES)
        test_df = test_df.head(config.MAX_DEBUG_SAMPLES)

    # 3. Process and Cache Data
    # Train
    train_data, train_labels, train_clips = process_and_cache_data(
        train_df, "train", config, load_cached_data
    )
    # Val
    val_data, val_labels, val_clips = process_and_cache_data(
        val_df, "val", config, load_cached_data
    )
    # Test
    test_data, test_labels, test_clips = process_and_cache_data(
        test_df, "test", config, load_cached_data
    )

    # 4. Create Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, train_clips, is_train=True, config=config
    )
    val_dataset = WhaleDataset(
        val_data, val_labels, val_clips, is_train=False, config=config
    )
    test_dataset = WhaleDataset(
        test_data, test_labels, test_clips, is_train=False, config=config
    )

    # 5. Create WeightedRandomSampler for Training
    # Calculate weights
    # Convert labels to numpy for counting
    np_labels = train_labels.numpy()
    class_counts = np.bincount(np_labels.astype(int))

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Weight = 1 / count
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[np_labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup/Batch Norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
