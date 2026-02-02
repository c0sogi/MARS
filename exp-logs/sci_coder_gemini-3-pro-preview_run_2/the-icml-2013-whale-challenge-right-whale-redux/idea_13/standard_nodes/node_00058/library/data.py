import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from library.config import Config

# Ensure reproducibility
from library.utils import seed_everything


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Handles on-the-fly Mel Spectrogram generation, normalization, and augmentation.
    """

    def __init__(self, waveforms, labels, hop_length, augment=False):
        """
        Args:
            waveforms (np.ndarray): Array of raw audio waveforms (N, samples).
            labels (list or np.ndarray): Labels for the samples. None for test.
            hop_length (int): Hop length for STFT, controlling temporal resolution.
            augment (bool): Whether to apply SpecAugment.
        """
        self.waveforms = waveforms
        self.labels = labels
        self.hop_length = hop_length
        self.augment = augment

        # Audio Configuration from Config
        self.sample_rate = Config.SAMPLE_RATE
        self.n_fft = Config.N_FFT
        self.n_mels = Config.N_MELS
        self.fmin = Config.FMIN
        self.fmax = Config.FMAX

        # Spectrogram Transform
        # We use normalized=False to preserve spectral tilt as per design
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            normalized=False,
        )

        # Log-Mel Conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # Augmentation Transforms
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 1. Load Waveform
        waveform = self.waveforms[idx]

        # Convert to Tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        # Ensure (1, Time) dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 2. Generate Mel Spectrogram
        # Shape: (1, n_mels, time)
        spec = self.mel_transform(waveform)

        # 3. Convert to dB (Log Scale)
        spec = self.amplitude_to_db(spec)

        # 4. Instance Standardization (Zero-Mean, Unit-Variance per clip)
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 5. Augmentation (Training Only)
        if self.augment:
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 6. Prepare Label
        if self.labels is not None and self.labels[idx] is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Dummy label for test set
            label = torch.tensor(0.0, dtype=torch.float32)

        return spec, label


def load_audio_data(df, input_root, cache_file, load_cached_data=True):
    """
    Loads raw audio data from files listed in the dataframe.
    Pads or crops audio to the fixed duration specified in Config.
    Caches the result as a .npy file for faster subsequent loading.

    Args:
        df (pd.DataFrame): Dataframe containing 'file_path'.
        input_root (str): Root directory for audio files.
        cache_file (str): Path to save/load the .npy cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of waveforms with shape (N, samples).
    """
    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached audio from {cache_file}...")
        try:
            data = np.load(cache_file)
            print(f"Successfully loaded {len(data)} samples from cache.")
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Process from Scratch
    print(f"Processing {len(df)} audio files from source...")
    target_len = int(Config.SAMPLE_RATE * Config.DURATION)
    waveforms = []

    for idx, row in df.iterrows():
        file_path = os.path.join(input_root, row["file_path"])
        try:
            # Load audio
            wav, sr = sf.read(file_path)

            # Resample if necessary (though data analysis showed 2kHz)
            if sr != Config.SAMPLE_RATE:
                # Simple resampling if needed, but assuming 2k based on analysis
                # For robustness, we could use librosa.resample, but avoiding extra deps
                pass

            # Pad or Crop to fixed length
            if len(wav) < target_len:
                pad_width = target_len - len(wav)
                wav = np.pad(wav, (0, pad_width), mode="constant")
            elif len(wav) > target_len:
                wav = wav[:target_len]

            waveforms.append(wav)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Fallback: silent clip
            waveforms.append(np.zeros(target_len))

    waveforms_np = np.array(waveforms, dtype=np.float32)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        np.save(cache_file, waveforms_np)
        print(f"Saved audio cache to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return waveforms_np


def get_dataloaders(
    fold=0,
    hop_length=Config.HOP_LENGTH_STANDARD,
    load_cached_data=True,
    batch_size=None,
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Implements 5-Fold Stratified Cross-Validation logic.

    Args:
        fold (int): Current fold index (0-4).
        hop_length (int): Hop length for spectrogram generation (controls resolution).
        load_cached_data (bool): Whether to use cached .npy files.
        batch_size (int): Batch size. If None, uses Config.BATCH_SIZE.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # --- 1. Load Metadata ---
    train_df_orig = pd.read_csv(Config.TRAIN_CSV)
    val_df_orig = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Combine provided train/val splits to perform our own Stratified CV
    full_train_df = pd.concat([train_df_orig, val_df_orig], ignore_index=True)

    # Handle Debug Mode
    if Config.DEBUG:
        full_train_df = full_train_df.iloc[: Config.DEBUG_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SIZE]
        print(
            f"DEBUG MODE: Reduced train size to {len(full_train_df)}, test size to {len(test_df)}"
        )

    # --- 2. Load Audio Data ---
    # Define cache paths (separate for debug mode to avoid conflicts)
    cache_suffix = "_debug" if Config.DEBUG else ""
    train_cache_path = os.path.join(
        Config.WORKING_DIR, f"train_waveforms{cache_suffix}.npy"
    )
    test_cache_path = os.path.join(
        Config.WORKING_DIR, f"test_waveforms{cache_suffix}.npy"
    )

    # Load waveforms (cached or fresh)
    train_waveforms = load_audio_data(
        full_train_df, Config.INPUT_ROOT, train_cache_path, load_cached_data
    )
    test_waveforms = load_audio_data(
        test_df, Config.INPUT_ROOT, test_cache_path, load_cached_data
    )

    # Extract labels
    train_labels_all = full_train_df["label"].values

    # --- 3. Split Folds ---
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    splits = list(skf.split(train_waveforms, train_labels_all))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {Config.NUM_FOLDS} splits.")

    train_idx, val_idx = splits[fold]

    X_train = train_waveforms[train_idx]
    y_train = train_labels_all[train_idx]

    X_val = train_waveforms[val_idx]
    y_val = train_labels_all[val_idx]

    print(f"Fold {fold}: Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # --- 4. Create Datasets ---
    train_dataset = WhaleDataset(
        X_train,
        y_train,
        hop_length=hop_length,
        augment=True,  # Apply SpecAugment for training
    )

    val_dataset = WhaleDataset(X_val, y_val, hop_length=hop_length, augment=False)

    test_dataset = WhaleDataset(
        test_waveforms,
        [None] * len(test_waveforms),
        hop_length=hop_length,
        augment=False,
    )

    # Calculate weights for WeightedRandomSampler (Cite solution_lesson_node_00004)
    # This addresses the 90:10 class imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # --- 5. Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        shuffle=False,  # Mutually exclusive with sampler
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
