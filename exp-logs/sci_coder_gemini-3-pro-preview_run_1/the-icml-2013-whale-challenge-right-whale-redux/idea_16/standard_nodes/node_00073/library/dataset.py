import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Set seeds for reproducibility in data processing
torch.manual_seed(Config.SEEDS[0])
np.random.seed(Config.SEEDS[0])


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Handles loading pre-processed spectrograms and applies SpecAugment during training.
    """

    def __init__(self, features, labels, clips, training=False):
        """
        Args:
            features (np.ndarray): Array of spectrograms (N, 1, F, T).
            labels (np.ndarray): Array of labels (N,).
            clips (np.ndarray): Array of clip filenames (N,).
            training (bool): If True, applies SpecAugment.
        """
        self.features = torch.from_numpy(features).float()

        if labels is not None:
            self.labels = torch.from_numpy(labels).float()
        else:
            # Dummy labels for test set if None provided
            self.labels = torch.zeros(len(features), dtype=torch.float32)

        self.clips = clips
        self.training = training

        # Augmentations (SpecAugment)
        # Time Masking: Masking a block of time steps
        self.time_masking = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        # Frequency Masking: Masking a block of frequency channels
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load spectrogram: (1, F, T)
        spec = self.features[idx]
        label = self.labels[idx]
        clip = self.clips[idx]

        if self.training:
            # Apply Amplitude Jitter (Robustness to volume changes)
            # Cite solution_lesson_node_00006: Enhancing generalization with noise.
            # Since spec is in dB, we add a random scalar.
            jitter = torch.empty(1).uniform_(
                -Config.AMP_JITTER_DB, Config.AMP_JITTER_DB
            )
            spec = spec + jitter

            # Apply SpecAugment
            # Note: Input to transforms must be (..., freq, time)
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        return spec, label, clip


def get_transforms():
    """
    Creates the preprocessing pipeline (Waveform -> Log-Mel Spectrogram).
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        center=True,
    )

    amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def process_subset(df, subset_name, load_cached_data, debug):
    """
    Loads audio files, converts to spectrograms, and caches the result as .npy files.

    Logic:
    1. If load_cached_data is True and files exist, load from disk.
    2. Else, process audio files from scratch.
    3. Save to disk for future use (unless in debug mode).
    """
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Define cache file paths
    data_path = os.path.join(Config.OUTPUT_DIR, f"{subset_name}_data.npy")
    labels_path = os.path.join(Config.OUTPUT_DIR, f"{subset_name}_labels.npy")
    clips_path = os.path.join(Config.OUTPUT_DIR, f"{subset_name}_clips.npy")

    # Logic: 1. Try to load cached data
    if load_cached_data and not debug:
        if (
            os.path.exists(data_path)
            and os.path.exists(labels_path)
            and os.path.exists(clips_path)
        ):
            print(f"Loading {subset_name} data from cache...")
            try:
                data = np.load(data_path)
                labels = np.load(labels_path)
                clips = np.load(clips_path)
                return data, labels, clips
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing...")

    # Logic: 2. Process from scratch
    print(f"Processing {subset_name} data ({len(df)} samples)...")

    transform_pipeline = get_transforms()
    target_samples = int(Config.SAMPLE_RATE * Config.DURATION)

    features_list = []
    labels_list = []
    clips_list = []

    for _, row in df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        clip = row["clip"]
        label = row["label"] if "label" in row else -1

        try:
            # Load audio
            waveform, sr = torchaudio.load(filepath)

            # Resample if needed
            if sr != Config.SAMPLE_RATE:
                resampler = T.Resample(sr, Config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Fix length (Pad or Truncate)
            current_samples = waveform.shape[1]
            if current_samples < target_samples:
                pad_amt = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
            elif current_samples > target_samples:
                waveform = waveform[:, :target_samples]

            # Compute Spectrogram
            # Input: (1, Time) -> Output: (1, F, T)
            spec = transform_pipeline(waveform)

            features_list.append(spec.numpy())
            labels_list.append(label)
            clips_list.append(clip)

        except Exception as e:
            # Skip invalid files, but print warning
            print(f"Warning: Error processing {filepath}: {e}")
            continue

    if not features_list:
        raise RuntimeError(f"No valid data processed for {subset_name}")

    # Stack into arrays
    data_arr = np.stack(features_list)  # Shape: (N, 1, F, T)
    labels_arr = np.array(labels_list, dtype=np.float32)
    clips_arr = np.array(clips_list)

    # Logic: 3. Save to cache (if not debug)
    if not debug:
        print(f"Saving {subset_name} data to cache at {Config.OUTPUT_DIR}...")
        np.save(data_path, data_arr)
        np.save(labels_path, labels_arr)
        np.save(clips_path, clips_arr)

    return data_arr, labels_arr, clips_arr


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        debug (bool): If True, runs on a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("DEBUG MODE: Reducing dataset size.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # Process Data
    # Note: For test set, labels will be -1 but we pass None to Dataset
    train_X, train_y, train_clips = process_subset(
        df_train, "train", load_cached_data, debug
    )
    val_X, val_y, val_clips = process_subset(df_val, "val", load_cached_data, debug)
    test_X, test_y, test_clips = process_subset(
        df_test, "test", load_cached_data, debug
    )

    # Initialize Datasets
    train_dataset = WhaleDataset(train_X, train_y, train_clips, training=True)
    val_dataset = WhaleDataset(val_X, val_y, val_clips, training=False)
    # Pass None for labels to test dataset to indicate inference mode
    test_dataset = WhaleDataset(test_X, None, test_clips, training=False)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Recommended for Mixup
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
