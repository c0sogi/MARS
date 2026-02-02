import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Stores preprocessed Log-Mel Spectrograms and labels.
    """

    def __init__(self, features, targets, is_test=False):
        """
        Args:
            features (Tensor): Tensor of shape (N, F, T) containing log-mel spectrograms.
            targets (Tensor or list): Tensor of labels (0/1) for train/val, or list of clip IDs for test.
            is_test (bool): Flag indicating if this is the test set.
        """
        self.features = features
        self.targets = targets
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features are (F, T), need to add channel dim -> (1, F, T)
        x = self.features[idx].unsqueeze(0)

        if self.is_test:
            # Return clip ID for submission generation
            y = self.targets[idx]
            return x, y
        else:
            # Return float label for BCEWithLogitsLoss
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y


def compute_log_mel_spec(filepath):
    """
    Loads audio, ensures fixed length, and computes Log-Mel Spectrogram.
    """
    try:
        waveform, sample_rate = torchaudio.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Return a silent tensor of correct shape as fallback
        return torch.zeros(
            (Config.N_MELS, (Config.TARGET_LENGTH // Config.HOP_LENGTH) + 1)
        )

    # Resample if necessary
    if sample_rate != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Fix length to TARGET_LENGTH (pad or truncate)
    current_len = waveform.shape[1]
    if current_len < Config.TARGET_LENGTH:
        padding = Config.TARGET_LENGTH - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_len > Config.TARGET_LENGTH:
        waveform = waveform[:, : Config.TARGET_LENGTH]

    # Compute Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_mels=Config.N_MELS,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
        center=True,
    )

    mel_spec = mel_transform(waveform)

    # Apply Log transformation (Log-Mel)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    # Remove channel dimension: (1, F, T) -> (F, T)
    return log_mel_spec.squeeze(0)


def load_and_process_split(
    csv_path,
    cache_data_path,
    cache_label_path,
    load_cached_data=True,
    is_test=False,
    limit=None,
):
    """
    Handles caching logic: loads from .npy if available, else processes from scratch.
    """
    # Only use cache if explicitly requested AND we are processing the full dataset (limit is None)
    use_cache = load_cached_data and (limit is None)

    if (
        use_cache
        and os.path.exists(cache_data_path)
        and os.path.exists(cache_label_path)
    ):
        print(f"Loading cached data from {cache_data_path}...")
        data = np.load(cache_data_path)
        labels = np.load(cache_label_path, allow_pickle=True)
        return torch.from_numpy(data), labels

    print(f"Processing data from {csv_path} (Limit: {limit})...")
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.iloc[:limit]

    data_list = []
    label_list = []

    for _, row in df.iterrows():
        # Construct full path
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # Process audio
        spec = compute_log_mel_spec(filepath)
        data_list.append(spec.numpy())

        if is_test:
            label_list.append(row["clip"])
        else:
            label_list.append(row["label"])

    # Convert to numpy arrays
    data_array = np.array(data_list, dtype=np.float32)
    label_array = np.array(label_list)  # int for train/val, object for test

    # Save to cache only if we processed the full dataset
    if limit is None:
        print(f"Saving processed data to {cache_data_path}...")
        os.makedirs(os.path.dirname(cache_data_path), exist_ok=True)
        np.save(cache_data_path, data_array)
        np.save(cache_label_path, label_array)

    return torch.from_numpy(data_array), label_array


def get_datasets(load_cached_data=True, limit=None):
    """
    Factory function to create Train, Val, and Test datasets.
    Handles loading, global normalization, and Dataset wrapping.
    """
    # 1. Load Training Data
    print("--- Loading Training Data ---")
    train_x, train_y = load_and_process_split(
        Config.TRAIN_CSV,
        Config.CACHE_TRAIN_DATA,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=load_cached_data,
        is_test=False,
        limit=limit,
    )

    # 2. Load Validation Data
    print("--- Loading Validation Data ---")
    val_x, val_y = load_and_process_split(
        Config.VAL_CSV,
        Config.CACHE_VAL_DATA,
        Config.CACHE_VAL_LABELS,
        load_cached_data=load_cached_data,
        is_test=False,
        limit=limit,
    )

    # 3. Load Test Data
    print("--- Loading Test Data ---")
    test_x, test_ids = load_and_process_split(
        Config.TEST_CSV,
        Config.CACHE_TEST_DATA,
        Config.CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
        is_test=True,
        limit=limit,
    )

    # 4. Compute Normalization Statistics (from Training Set ONLY)
    print("Computing normalization statistics from training set...")
    mean = train_x.mean()
    std = train_x.std()
    print(f"Global Mean: {mean}, Global Std: {std}")

    # 5. Apply Normalization
    # (x - mean) / std
    train_x = (train_x - mean) / (std + 1e-9)
    val_x = (val_x - mean) / (std + 1e-9)
    test_x = (test_x - mean) / (std + 1e-9)

    # 6. Create Dataset Objects
    train_dataset = WhaleDataset(train_x, train_y, is_test=False)
    val_dataset = WhaleDataset(val_x, val_y, is_test=False)
    test_dataset = WhaleDataset(test_x, test_ids, is_test=True)

    return train_dataset, val_dataset, test_dataset
