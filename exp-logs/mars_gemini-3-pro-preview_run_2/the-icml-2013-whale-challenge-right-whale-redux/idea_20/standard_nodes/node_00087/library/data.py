import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


# --- Transforms ---
class SpecAugment:
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
    """

    def __init__(self, freq_mask_param, time_mask_param):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, spec):
        # spec shape: (..., freq, time)
        return self.time_mask(self.freq_mask(spec))


# --- Dataset ---
class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Handles waveform loading (cached or from disk), spectrogram generation, and normalization.
    """

    def __init__(self, df, data_array, config, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.data_array = data_array  # Numpy array of waveforms (N, samples)
        self.config = config
        self.transform = transform
        self.is_test = is_test

        # Audio Processing definitions
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
            normalized=False,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=config.TOP_DB)

        # Target length in samples (2.0s * 2000Hz = 4000)
        self.target_length = int(2.0 * config.SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Waveform
        if self.data_array is not None:
            # Retrieve from cache using the pre-assigned cache index
            cache_idx = row["cache_idx"]
            waveform = self.data_array[cache_idx]
            waveform = torch.from_numpy(waveform).float()
        else:
            # Fallback: Load from disk
            file_path = os.path.join(self.config.INPUT_ROOT, row["file_path"])
            waveform, sr = sf.read(file_path)
            waveform = torch.from_numpy(waveform).float()
            # Note: We assume SR is correct based on analysis, but could resample if needed.

        # 2. Fix Length (Pad or Crop)
        current_len = waveform.shape[0]
        if current_len < self.target_length:
            pad_amt = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        elif current_len > self.target_length:
            waveform = waveform[: self.target_length]

        # 3. Generate Spectrogram
        # Input: (1, time) -> Output: (1, n_mels, time)
        spec = self.mel_spec(waveform.unsqueeze(0))
        spec = self.amp_to_db(spec)

        # 4. Instance Standardization
        # Zero-Mean, Unit-Variance per clip
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 5. Augmentation (Train only)
        if self.transform:
            spec = self.transform(spec)

        # 6. Label
        if self.is_test:
            # Dummy label for test set
            label = torch.tensor(0.0, dtype=torch.float32)
        else:
            label = torch.tensor(row["label"], dtype=torch.float32)

        return spec, label


# --- Caching Utility ---
def load_and_cache_waveforms(df, cache_name, config, load_cached_data=True):
    """
    Loads audio files listed in the dataframe and caches them as a numpy array.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"{cache_name}.npy")
    target_length = int(2.0 * config.SAMPLE_RATE)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            if len(data) == len(df):
                return data
            else:
                print("Cache size mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing and caching {len(df)} audio files to {cache_path}...")
    data_list = []

    for _, row in df.iterrows():
        file_path = os.path.join(config.INPUT_ROOT, row["file_path"])
        try:
            wav, _ = sf.read(file_path)
            # Fix length before stacking to ensure numpy array is uniform
            if len(wav) < target_length:
                wav = np.pad(wav, (0, target_length - len(wav)))
            elif len(wav) > target_length:
                wav = wav[:target_length]
            data_list.append(wav)
        except Exception as e:
            # Fallback for corrupt files (should not happen based on analysis)
            data_list.append(np.zeros(target_length))

    data = np.array(data_list, dtype=np.float32)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)

    return data


# --- Main DataLoader Function ---
def get_dataloaders(
    config, fold=0, mode="train", pseudo_df=None, load_cached_data=True
):
    """
    Creates DataLoaders for Training (with K-Fold), Pseudo-Labeling, or Testing.

    Args:
        config: Config object.
        fold: Current fold index (0-4).
        mode: 'train', 'pseudo', or 'test'.
        pseudo_df: DataFrame containing pseudo-labels (used in Round 2).
        load_cached_data: Whether to use cached .npy files.
    """
    seed_everything(config.SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_CSV)
    val_meta = pd.read_csv(config.VAL_CSV)
    test_meta = pd.read_csv(config.TEST_CSV)

    # Combine provided train and val splits to form the full labeled dataset for CV
    full_train_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Cache Management
    # We assign a 'cache_idx' to map dataframe rows to the numpy array
    full_train_df["cache_idx"] = range(len(full_train_df))
    test_meta["cache_idx"] = range(len(test_meta))

    # Load/Create Cache arrays
    train_data = load_and_cache_waveforms(
        full_train_df, "full_train_waveforms", config, load_cached_data
    )
    test_data = load_and_cache_waveforms(
        test_meta, "test_waveforms", config, load_cached_data
    )

    # 3. Mode Handling
    if mode == "test":
        # Test Loader
        ds = WhaleDataset(test_meta, test_data, config, transform=None, is_test=True)
        loader = DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        return loader

    else:
        # Train/Pseudo Mode (Cross-Validation)
        skf = StratifiedKFold(
            n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
        )

        # Get indices for the requested fold
        # Note: skf.split requires y, so we pass labels
        fold_generator = skf.split(full_train_df, full_train_df["label"])
        train_idx, val_idx = next(x for i, x in enumerate(fold_generator) if i == fold)

        train_sub_df = full_train_df.iloc[train_idx].copy()
        val_sub_df = full_train_df.iloc[val_idx].copy()

        # Data source for training starts with just the labeled train data
        train_data_source = train_data

        # Handle Pseudo-Labeling (Round 2)
        if pseudo_df is not None and not pseudo_df.empty:
            # pseudo_df should have 'clip' and 'label' columns, and 'file_path'
            # We need to map these to the test_data cache

            # Filter test_meta to find the rows corresponding to pseudo_df
            # We assume pseudo_df has 'clip' column matching test_meta
            pseudo_subset = test_meta[test_meta["clip"].isin(pseudo_df["clip"])].copy()

            # Map labels from pseudo_df
            label_map = dict(zip(pseudo_df["clip"], pseudo_df["label"]))
            pseudo_subset["label"] = pseudo_subset["clip"].map(label_map)

            # Adjust cache_idx for the combined array
            # We will concatenate train_data and test_data for the dataset
            # So test indices need to be offset by len(train_data)
            offset = len(train_data)
            pseudo_subset["cache_idx"] = pseudo_subset["cache_idx"] + offset

            # Merge DataFrames
            train_sub_df = pd.concat([train_sub_df, pseudo_subset], ignore_index=True)

            # Merge Data Arrays
            # Note: This creates a new large array in memory.
            # Given sizes (~300MB + ~400MB), this is safe (220GB RAM available).
            train_data_source = np.concatenate([train_data, test_data], axis=0)

        # Define Transforms
        train_transform = SpecAugment(config.FREQ_MASK_PARAM, config.TIME_MASK_PARAM)

        # Create Datasets
        train_ds = WhaleDataset(
            train_sub_df,
            train_data_source,
            config,
            transform=train_transform,
            is_test=False,
        )

        val_ds = WhaleDataset(
            val_sub_df,
            train_data,  # Validation always comes from original train set
            config,
            transform=None,
            is_test=False,
        )

        # Weighted Random Sampler for Training
        # Calculate weights based on class prevalence in the current training split
        labels = train_sub_df["label"].values
        class_counts = np.bincount(labels.astype(int))
        # Handle potential division by zero if a class is missing (unlikely in stratified)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels.astype(int)]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_sub_df), replacement=True
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader
