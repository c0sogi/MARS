import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import library.config as config
from library.utils import set_seed

# Ensure reproducibility
set_seed(config.SEED)


def get_transforms(train=False):
    """
    Returns the composition of transforms for the audio data.
    """
    transforms = []

    # 1. Mel Spectrogram
    # Config: SAMPLE_RATE=2000, N_FFT=512, HOP_LENGTH=10, N_MELS=128
    # Hop length of 10 samples @ 2000Hz = 5ms.
    transforms.append(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            power=2.0,
        )
    )

    # 2. Amplitude to DB (Log-Mel)
    transforms.append(torchaudio.transforms.AmplitudeToDB(top_db=80.0))

    # 3. SpecAugment (Only for training)
    if train:
        # Time Masking: Max 200ms.
        # Frame duration = 10 samples / 2000 Hz = 0.005s = 5ms.
        # Max mask frames = 200ms / 5ms = 40 frames.
        transforms.append(torchaudio.transforms.TimeMasking(time_mask_param=40))

        # Frequency Masking: Conservative value, e.g., 15 bins out of 128
        transforms.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=15))

    return torch.nn.Sequential(*transforms)


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, train=False):
        """
        Args:
            data (np.ndarray): Array of raw waveforms (N, samples).
            labels (np.ndarray, optional): Array of labels (N,).
            train (bool): Whether to apply training augmentations.
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.train = train
        self.transforms = get_transforms(train=train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get raw waveform: (4000,)
        waveform = self.data[idx]

        # Add channel dimension: (1, 4000)
        waveform = waveform.unsqueeze(0)

        # Apply transforms -> (1, 128, TimeFrames)
        spec = self.transforms(waveform)

        if self.labels is not None:
            label = self.labels[idx]
            return spec, label
        else:
            # Return dummy label for test set
            return spec, torch.tensor(-1.0)


def process_audio_files(metadata_df):
    """
    Loads audio files, pads/crops to fixed length, and returns numpy arrays.
    """
    # Fixed length: 2.0s * 2000Hz = 4000 samples
    target_length = int(2.0 * config.SAMPLE_RATE)

    waveforms = []
    labels = []

    has_labels = "label" in metadata_df.columns

    for _, row in metadata_df.iterrows():
        full_path = os.path.join(config.INPUT_ROOT, row["filepath"])

        try:
            # Load audio
            wav, sr = torchaudio.load(full_path)

            # Resample if necessary (though analysis says all are 2000Hz)
            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                wav = resampler(wav)

            # Convert to mono if necessary (analysis says all are mono)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Flatten to (Time,)
            wav = wav.squeeze(0)

            # Pad or Crop
            if wav.shape[0] < target_length:
                padding = target_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            elif wav.shape[0] > target_length:
                wav = wav[:target_length]

            waveforms.append(wav.numpy())

            if has_labels:
                labels.append(row["label"])

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            # Append zeros in case of error to maintain alignment
            waveforms.append(np.zeros(target_length, dtype=np.float32))
            if has_labels:
                labels.append(0)  # Assume noise if error

    waveforms_np = np.array(waveforms, dtype=np.float32)

    if has_labels:
        labels_np = np.array(labels, dtype=np.float32)
        return waveforms_np, labels_np
    else:
        return waveforms_np, None


def load_dataset_cached(
    metadata_path, cache_prefix, load_cached_data=True, debug=False, debug_size=100
):
    """
    Loads dataset from metadata, using caching mechanism.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    suffix = "_debug" if debug else ""
    data_path = os.path.join(cache_dir, f"{cache_prefix}{suffix}_data.npy")
    labels_path = os.path.join(cache_dir, f"{cache_prefix}{suffix}_labels.npy")

    # Load metadata
    df = pd.read_csv(metadata_path)
    if debug:
        df = df.iloc[:debug_size]

    # Try loading from cache
    if load_cached_data and os.path.exists(data_path):
        print(f"Loading {cache_prefix}{suffix} data from cache...")
        data = np.load(data_path)
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            labels = None
        return data, labels

    # Process from scratch
    print(f"Processing {cache_prefix}{suffix} data from scratch...")
    data, labels = process_audio_files(df)

    # Save to cache
    print(f"Saving {cache_prefix}{suffix} data to cache...")
    np.save(data_path, data)
    if labels is not None:
        np.save(labels_path, labels)

    return data, labels


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    debug_size = config.DEBUG_SIZE if debug else 0

    # Load Data
    train_data, train_labels = load_dataset_cached(
        config.TRAIN_METADATA_PATH, "train", load_cached_data, debug, debug_size
    )
    val_data, val_labels = load_dataset_cached(
        config.VAL_METADATA_PATH, "val", load_cached_data, debug, debug_size
    )
    test_data, _ = load_dataset_cached(
        config.TEST_METADATA_PATH, "test", load_cached_data, debug, debug_size
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, train=True)
    val_dataset = WhaleDataset(val_data, val_labels, train=False)
    test_dataset = WhaleDataset(test_data, None, train=False)

    # Create DataLoaders
    # Drop last for train to ensure stable batch statistics for Mixup
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
