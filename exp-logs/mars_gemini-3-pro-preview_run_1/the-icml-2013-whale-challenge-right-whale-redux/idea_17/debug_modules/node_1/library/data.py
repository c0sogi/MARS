import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Define transformations globally to ensure consistency
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=Config.SAMPLE_RATE,
    n_fft=Config.N_FFT,
    win_length=Config.WIN_LENGTH,
    hop_length=Config.HOP_LENGTH,
    f_min=Config.F_MIN,
    f_max=Config.F_MAX,
    n_mels=Config.N_MELS,
    power=Config.POWER,
    center=True,
    pad_mode="reflect",
    norm="slaney",
)

amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)


def process_audio_file(filepath):
    """
    Loads an audio file, pads/crops it to the fixed duration,
    and computes the Log-Mel Spectrogram.
    """
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        # Fallback for corrupted files (though analysis showed none)
        # Create a silent waveform
        waveform = torch.zeros(1, Config.NUM_SAMPLES)
        sr = Config.SAMPLE_RATE

    # Resample if necessary (though analysis showed all are 2000Hz)
    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=Config.SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed length
    current_len = waveform.shape[1]
    target_len = Config.NUM_SAMPLES

    if current_len < target_len:
        pad_amount = target_len - current_len
        # Pad at the end
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_len > target_len:
        # Crop from the center
        start = (current_len - target_len) // 2
        waveform = waveform[:, start : start + target_len]

    # Compute Spectrogram
    # Input: (1, Time) -> Output: (1, Freq, Time)
    spec = mel_spectrogram_transform(waveform)
    log_spec = amplitude_to_db_transform(spec)

    return log_spec


def load_dataset_from_cache_or_compute(df, cache_name, load_cached_data=True):
    """
    Handles caching logic: Load from .npy if available, else compute and save.
    """
    cache_path_data = Config.get_cache_path(f"{cache_name}_data.npy")
    cache_path_ids = Config.get_cache_path(f"{cache_name}_ids.npy")

    # If labels exist in dataframe, we cache them too
    has_labels = "label" in df.columns
    cache_path_labels = Config.get_cache_path(f"{cache_name}_labels.npy")

    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path_data):
        print(f"Loading {cache_name} data from cache...")
        data = np.load(cache_path_data)
        ids = np.load(cache_path_ids)

        labels = None
        if has_labels and os.path.exists(cache_path_labels):
            labels = np.load(cache_path_labels)

        return data, labels, ids

    # 2. Compute from scratch
    print(f"Processing {cache_name} data from scratch...")
    data_list = []
    label_list = []
    id_list = []

    for idx, row in df.iterrows():
        # Metadata filepath is relative to input root
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])

        spec = process_audio_file(full_path)
        # Convert to numpy for storage
        data_list.append(spec.numpy())
        id_list.append(row["clip"])

        if has_labels:
            label_list.append(row["label"])

    # Stack into arrays
    # Shape: (N, 1, F, T)
    data_array = np.stack(data_list).astype(np.float32)
    ids_array = np.array(id_list)

    if has_labels:
        labels_array = np.array(label_list).astype(np.float32)
    else:
        labels_array = None

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path_data, data_array)
    np.save(cache_path_ids, ids_array)
    if has_labels:
        np.save(cache_path_labels, labels_array)

    print(f"Saved {cache_name} data to cache.")

    return data_array, labels_array, ids_array


class WhaleDataset(Dataset):
    """
    Dataset class serving pre-processed spectrograms.
    Applies SpecAugment during training.
    """

    def __init__(self, data, labels=None, ids=None, mode="train"):
        self.data = data  # numpy array (N, 1, F, T)
        self.labels = labels  # numpy array (N,)
        self.ids = ids
        self.mode = mode

        # SpecAugment Transforms
        # Time mask param 20 corresponds to 200ms given hop length 20 and SR 2000
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to tensor
        spec = torch.from_numpy(self.data[idx])

        # Apply Augmentation if training
        if self.mode == "train":
            # SpecAugment expects (..., Freq, Time)
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec, label
        else:
            # For test set, return clip ID as well for submission
            clip_id = self.ids[idx]
            return spec, clip_id


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to a batch of data.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main function to prepare data and return dataloaders.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to attempt loading from cache.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print(f"Debug Mode: using {Config.DEBUG_SUBSET_SIZE} samples.")
        df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]
        # Use separate cache names for debug to avoid overwriting full cache
        cache_suffix = "_debug"
    else:
        cache_suffix = ""

    # 2. Prepare Data (Load/Compute/Cache)
    train_data, train_labels, train_ids = load_dataset_from_cache_or_compute(
        df_train, f"train{cache_suffix}", load_cached_data
    )
    val_data, val_labels, val_ids = load_dataset_from_cache_or_compute(
        df_val, f"val{cache_suffix}", load_cached_data
    )
    test_data, _, test_ids = load_dataset_from_cache_or_compute(
        df_test, f"test{cache_suffix}", load_cached_data
    )

    # 3. Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, train_ids, mode="train")
    val_dataset = WhaleDataset(val_data, val_labels, val_ids, mode="val")
    test_dataset = WhaleDataset(test_data, None, test_ids, mode="test")

    # 4. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for batchnorm stability
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

    print(
        f"Data Loaders Ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
