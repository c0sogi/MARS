import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


def get_transforms(phase: str):
    """
    Returns the data augmentation transforms for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        nn.Module or None: The transform pipeline.
    """
    if phase == "train":
        # SpecAugment for training: Time and Frequency Masking
        return torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=10),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
        )
    # No augmentation for validation or test
    return None


class WhaleDataset(Dataset):
    """
    Dataset class for Right Whale Detection.
    Generates Mel-Spectrograms on-the-fly from raw waveforms to allow for
    flexible resolution and augmentation.
    """

    def __init__(self, waveforms, labels=None, transform=None, training=False):
        """
        Args:
            waveforms (np.ndarray): Array of shape (N, samples) containing raw audio.
            labels (np.ndarray, optional): Array of shape (N,) containing targets.
            transform (nn.Module, optional): Augmentation transforms.
            training (bool): Flag indicating if this is a training dataset.
        """
        self.waveforms = waveforms
        self.labels = labels
        self.transform = transform
        self.training = training

        # Initialize MelSpectrogram transform
        # n_fft=1024, hop_length=64 gives high resolution (approx 128x63)
        # normalized=False preserves spectral tilt (Pink noise)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=Config.MEL_NORMALIZED,
        )

        # Standard Log-Mel conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 1. Load waveform
        wav = self.waveforms[idx]  # Shape: (samples,)

        # 2. Convert to Tensor and add channel dimension
        # Shape: (1, samples)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)

        # 3. Generate Mel-Spectrogram
        # Shape: (1, n_mels, time) -> Approx (1, 128, 63)
        spec = self.mel_spectrogram(wav_tensor)

        # 4. Convert to Log Scale (dB)
        spec = self.amplitude_to_db(spec)

        # 5. Instance Standardization (Zero-Mean, Unit-Variance)
        # Critical for convergence in audio tasks
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 6. Apply Augmentation (SpecAugment)
        if self.training and self.transform is not None:
            spec = self.transform(spec)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec, label
        else:
            return spec


def _load_and_cache_data(
    df,
    input_root,
    data_cache_path,
    labels_cache_path=None,
    clips_cache_path=None,
    load_cached=True,
):
    """
    Helper function to load audio files, pad/crop them, and cache as numpy arrays.
    """
    # Attempt to load from cache
    if load_cached:
        if os.path.exists(data_cache_path):
            print(f"Loading cached data from {data_cache_path}")
            data = np.load(data_cache_path)

            labels = None
            if labels_cache_path and os.path.exists(labels_cache_path):
                labels = np.load(labels_cache_path)

            clips = None
            if clips_cache_path and os.path.exists(clips_cache_path):
                clips = np.load(clips_cache_path)

            # Basic integrity check
            if len(data) == len(df):
                return data, labels, clips
            else:
                print("Cached data size mismatch. Recomputing...")

    print(f"Processing {len(df)} audio files...")

    # Fixed length for all audio clips: 2.0 seconds * 2000 Hz = 4000 samples
    fixed_length = int(Config.SAMPLE_RATE * 2.0)
    num_samples = len(df)

    # Pre-allocate memory
    data = np.zeros((num_samples, fixed_length), dtype=np.float32)
    labels_list = []
    clips_list = []

    for i, row in df.iterrows():
        # Construct full path
        file_path = os.path.join(input_root, row["file_path"])

        try:
            # Read audio
            wav, sr = sf.read(file_path)

            # Ensure fixed length (pad or crop)
            if len(wav) > fixed_length:
                wav = wav[:fixed_length]
            elif len(wav) < fixed_length:
                pad_width = fixed_length - len(wav)
                wav = np.pad(wav, (0, pad_width), mode="constant")

            data[i] = wav

            if "label" in row:
                labels_list.append(row["label"])

            if "clip" in row:
                clips_list.append(row["clip"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Keep zero-initialized array for this index (silence)

    # Save to cache
    os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)
    np.save(data_cache_path, data)

    labels_arr = None
    if labels_list:
        labels_arr = np.array(labels_list, dtype=np.int64)
        if labels_cache_path:
            np.save(labels_cache_path, labels_arr)

    clips_arr = None
    if clips_list:
        clips_arr = np.array(clips_list)
        if clips_cache_path:
            np.save(clips_cache_path, clips_arr)

    return data, labels_arr, clips_arr


def get_data_loaders(fold=0, load_cached_data=True):
    """
    Generates DataLoaders for the specified fold using Stratified K-Fold.
    Merges train and validation metadata to perform the split.

    Args:
        fold (int): The fold index (0-4).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, valid_loader, test_loader, test_clips)
    """
    seed_everything(Config.SEED)

    # --- 1. Load Metadata ---
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # --- 2. Prepare Labeled Data (Train + Val) ---
    # Load separate caches (to respect Config paths) but merge in memory
    train_data, train_labels, _ = _load_and_cache_data(
        train_df,
        Config.INPUT_ROOT,
        Config.CACHE_TRAIN_DATA,
        Config.CACHE_TRAIN_LABELS,
        load_cached=load_cached_data,
    )

    val_data, val_labels, _ = _load_and_cache_data(
        val_df,
        Config.INPUT_ROOT,
        Config.CACHE_VAL_DATA,
        Config.CACHE_VAL_LABELS,
        load_cached=load_cached_data,
    )

    # Concatenate to form full dataset for Cross-Validation
    full_data = np.concatenate([train_data, val_data], axis=0)
    full_labels = np.concatenate([train_labels, val_labels], axis=0)

    # --- 3. Stratified K-Fold Split ---
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(full_data, full_labels))

    if fold >= Config.NUM_FOLDS:
        raise ValueError(f"Fold {fold} out of range (0-{Config.NUM_FOLDS-1})")

    train_idx, valid_idx = splits[fold]

    X_train, y_train = full_data[train_idx], full_labels[train_idx]
    X_valid, y_valid = full_data[valid_idx], full_labels[valid_idx]

    # --- 4. Create Datasets ---
    train_dataset = WhaleDataset(
        X_train, y_train, transform=get_transforms("train"), training=True
    )

    valid_dataset = WhaleDataset(
        X_valid, y_valid, transform=get_transforms("valid"), training=False
    )

    # --- 5. Weighted Random Sampler ---
    # Handle class imbalance by oversampling the minority class
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    # Assign weight to each sample based on its label
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # --- 6. Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 7. Test Data ---
    test_data, _, test_clips = _load_and_cache_data(
        test_df,
        Config.INPUT_ROOT,
        Config.CACHE_TEST_DATA,
        None,
        Config.CACHE_TEST_CLIPS,
        load_cached=load_cached_data,
    )

    test_dataset = WhaleDataset(
        test_data, labels=None, transform=get_transforms("test"), training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, test_clips
