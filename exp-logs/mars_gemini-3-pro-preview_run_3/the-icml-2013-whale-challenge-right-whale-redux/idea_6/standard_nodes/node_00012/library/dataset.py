import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def load_audio_and_convert_to_spec(
    file_path, target_sr=2000, duration=2.0, n_mels=384, n_fft=1024, hop_length=20
):
    """
    Loads an audio file, pads/truncates it to a fixed length, and converts it
    to a normalized Log-Mel Spectrogram.
    """
    try:
        # Load audio
        waveform, sr = torchaudio.load(file_path)

        # Resample if necessary (though dataset analysis says all are 2000Hz)
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Ensure single channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Fix length (Padding / Truncation)
        target_length = int(target_sr * duration)
        current_length = waveform.shape[1]

        if current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > target_length:
            waveform = waveform[:, :target_length]

        # Compute Mel Spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            power=2.0,
        )

        mel_spec = mel_transform(waveform)

        # Convert to DB (Log scale)
        db_transform = T.AmplitudeToDB(top_db=80)
        log_mel_spec = db_transform(mel_spec)

        # Instance-level Min-Max Normalization
        # Shape is (1, n_mels, time)
        spec_min = log_mel_spec.min()
        spec_max = log_mel_spec.max()

        # Avoid division by zero
        if spec_max - spec_min > 1e-6:
            log_mel_spec = (log_mel_spec - spec_min) / (spec_max - spec_min)
        else:
            log_mel_spec = torch.zeros_like(log_mel_spec)

        return log_mel_spec.squeeze(0).numpy()  # Return as (n_mels, time) numpy array

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero tensor of expected shape as fallback
        # Time dimension = target_length // hop_length + 1
        n_time = int(target_sr * duration) // hop_length + 1
        return np.zeros((n_mels, n_time), dtype=np.float32)


def process_and_cache_data(metadata_df, cache_path, input_dir):
    """
    Iterates through metadata, processes audio files, and saves them to a .npz archive.
    """
    print(f"Processing data for cache: {cache_path}...")
    specs = []
    labels = []
    names = []

    # Pre-instantiate transforms if possible, but torchaudio transforms are cheap to init
    # We do it inside the helper for simplicity as it's a one-time cost per file

    total = len(metadata_df)
    for idx, row in metadata_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        spec = load_audio_and_convert_to_spec(
            full_path,
            target_sr=Config.SAMPLE_RATE,
            duration=Config.DURATION,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
        )

        specs.append(spec)
        names.append(row["clip_name"])

        if "label" in row:
            labels.append(row["label"])

    # Stack into numpy arrays
    specs_arr = np.stack(specs).astype(np.float32)  # (N, n_mels, time)
    names_arr = np.array(names)

    save_dict = {"specs": specs_arr, "names": names_arr}

    if labels:
        labels_arr = np.array(labels).astype(np.float32)
        save_dict["labels"] = labels_arr

    # Save compressed
    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved cache to {cache_path}. Shape: {specs_arr.shape}")

    return save_dict


class WhaleDataset(Dataset):
    def __init__(self, data_dict, mode="train", transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing 'specs', 'names', and optionally 'labels'.
            mode (str): 'train', 'val', or 'test'.
            transform (bool): Whether to apply augmentation (SpecAugment).
        """
        self.specs = data_dict["specs"]
        self.names = data_dict["names"]
        self.labels = data_dict.get("labels", None)
        self.mode = mode
        self.transform = transform

        # SpecAugment Transforms
        if self.mode == "train":
            self.time_masking = T.TimeMasking(time_mask_param=Config.SPEC_AUG_TIME_MASK)
            self.freq_masking = T.FrequencyMasking(
                freq_mask_param=Config.SPEC_AUG_FREQ_MASK
            )

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Load spec: (n_mels, time)
        spec = self.specs[idx]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec)  # (H, W)

        # Add channel dim for transforms: (1, H, W)
        spec_tensor = spec_tensor.unsqueeze(0)

        # Apply SpecAugment if training
        if self.mode == "train" and self.transform:
            # SpecAugment expects (channel, freq, time)
            spec_tensor = self.freq_masking(spec_tensor)
            spec_tensor = self.time_masking(spec_tensor)

        # Expand to 3 channels for EfficientNet (C, H, W)
        image = spec_tensor.expand(3, -1, -1)

        if self.mode == "test":
            return image, self.names[idx]
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Main function to prepare datasets and dataloaders.
    Handles caching logic.
    """
    seed_everything(Config.SEED)

    # Define cache filenames
    suffix = "_debug" if debug else ""
    train_cache = os.path.join(Config.WORKING_DIR, f"train{suffix}.npz")
    val_cache = os.path.join(Config.WORKING_DIR, f"val{suffix}.npz")
    test_cache = os.path.join(Config.WORKING_DIR, f"test{suffix}.npz")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # --- TRAIN DATA ---
    if load_cached_data and os.path.exists(train_cache):
        print(f"Loading cached training data from {train_cache}...")
        train_data = np.load(train_cache)
    else:
        train_data = process_and_cache_data(train_df, train_cache, Config.INPUT_DIR)

    # --- VAL DATA ---
    if load_cached_data and os.path.exists(val_cache):
        print(f"Loading cached validation data from {val_cache}...")
        val_data = np.load(val_cache)
    else:
        val_data = process_and_cache_data(val_df, val_cache, Config.INPUT_DIR)

    # --- TEST DATA ---
    if load_cached_data and os.path.exists(test_cache):
        print(f"Loading cached test data from {test_cache}...")
        test_data = np.load(test_cache)
    else:
        test_data = process_and_cache_data(test_df, test_cache, Config.INPUT_DIR)

    # Create Datasets
    train_dataset = WhaleDataset(train_data, mode="train", transform=True)
    val_dataset = WhaleDataset(val_data, mode="val", transform=False)
    test_dataset = WhaleDataset(test_data, mode="test", transform=False)

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
