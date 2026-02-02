import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Wraps pre-processed in-memory numpy arrays.
    Applies SpecAugment during training.
    """

    def __init__(self, data, labels=None, training=False):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.training = training

        # SpecAugment Transforms
        # Frequency Masking: Mask up to 15% of mel bins (approx 20 bins)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        # Time Masking: Strictly limited by Config to preserve short calls
        self.time_mask = T.TimeMasking(time_mask_param=Config.MAX_TIME_MASK_FRAMES)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is (3, 128, T)
        x = self.data[idx]

        if self.training:
            # Apply SpecAugment to the spectrograms
            # We apply the same mask logic to all channels or independent?
            # Usually independent is fine, but for RGB-like structure,
            # applying consistent masking might be better.
            # However, torchaudio applies to the whole tensor if passed (C, F, T).
            # It masks a range of time steps across all channels.
            x = self.freq_mask(x)
            x = self.time_mask(x)

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            return x, torch.zeros(1)  # Dummy label for test


class MixupCollate:
    """
    Collate function to apply Mixup augmentation to a batch.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: list of (input, target) tuples
        Returns:
            mixed_inputs, targets_a, targets_b, lam
        """
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])

        batch_size = inputs.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        targets_a = targets
        targets_b = targets[index]

        return mixed_inputs, targets_a, targets_b, lam


def compute_spectrograms(waveform, sample_rate):
    """
    Generates 3-channel Multi-Resolution Log-Mel Spectrogram.
    """
    # Ensure waveform is exactly the target duration
    target_len = int(Config.DURATION * Config.SAMPLE_RATE)
    current_len = waveform.shape[1]

    if current_len < target_len:
        pad_amt = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    specs = []
    # Generate 3 channels with different window sizes
    for win_length in Config.WINDOW_SIZES:
        # n_fft must be >= win_length. We use the next power of 2 or just a sufficiently large number
        # to accommodate the largest window (1000). 2048 is safe.
        n_fft = 2048

        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        # (1, n_mels, time)
        spec = transform(waveform)
        specs.append(spec)

    # Stack along channel dimension -> (3, n_mels, time)
    multi_res_spec = torch.cat(specs, dim=0)

    # Log transform
    multi_res_spec = torch.log(multi_res_spec + 1e-9)

    # Instance Normalization
    mean = multi_res_spec.mean(dim=(1, 2), keepdim=True)
    std = multi_res_spec.std(dim=(1, 2), keepdim=True)
    multi_res_spec = (multi_res_spec - mean) / (std + 1e-5)

    return multi_res_spec


def process_dataset(df_path, dataset_name, load_cached_data=True):
    """
    Loads audio paths from metadata, processes them into tensors, and caches them.
    """
    cache_data_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_data.npy")
    cache_label_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_data_path):
        print(f"Loading cached {dataset_name} data from {Config.WORKING_DIR}...")
        data = np.load(cache_data_path)
        if os.path.exists(cache_label_path):
            labels = np.load(cache_label_path)
        else:
            labels = None
        return data, labels

    # 2. Process from scratch
    print(f"Processing {dataset_name} data from scratch...")
    df = pd.read_csv(df_path)

    # Pre-allocate lists
    data_list = []
    label_list = []

    # Iterate
    for idx, row in df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])

        try:
            waveform, sr = torchaudio.load(filepath)

            # Resample if necessary (though data analysis showed 2000Hz consistent)
            if sr != Config.SAMPLE_RATE:
                resampler = T.Resample(sr, Config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            spec_tensor = compute_spectrograms(waveform, Config.SAMPLE_RATE)
            data_list.append(spec_tensor.numpy())

            if "label" in row:
                label_list.append(row["label"])

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # Append zero tensor to maintain alignment or skip?
            # Skipping breaks alignment with dataframe indices if we needed them,
            # but for training we just need data/label pairs.
            continue

    # Stack
    data_array = np.stack(data_list)  # (N, 3, 128, T)

    if label_list:
        label_array = np.array(label_list, dtype=np.float32)
    else:
        label_array = None

    # 3. Save to cache
    print(f"Saving {dataset_name} data to cache...")
    np.save(cache_data_path, data_array)
    if label_array is not None:
        np.save(cache_label_path, label_array)

    return data_array, label_array


def get_dataloaders(load_cached_data=True, debug_subset=False):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        debug_subset (bool): If True, limits data to a small subset for debugging.
    """

    # 1. Load Data
    train_data, train_labels = process_dataset(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_data, val_labels = process_dataset(Config.VAL_CSV, "val", load_cached_data)
    test_data, _ = process_dataset(Config.TEST_CSV, "test", load_cached_data)

    # Debug mode
    if debug_subset:
        print("Debug mode: Truncating datasets...")
        train_data, train_labels = train_data[:100], train_labels[:100]
        val_data, val_labels = val_data[:100], val_labels[:100]
        test_data = test_data[:100]

    # 2. Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, training=True)
    val_dataset = WhaleDataset(val_data, val_labels, training=False)
    test_dataset = WhaleDataset(test_data, None, training=False)

    # 3. Create DataLoaders
    # Train loader uses MixupCollate
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=MixupCollate(alpha=Config.MIXUP_ALPHA),
        pin_memory=True,
    )

    # Val and Test loaders use default collate
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
