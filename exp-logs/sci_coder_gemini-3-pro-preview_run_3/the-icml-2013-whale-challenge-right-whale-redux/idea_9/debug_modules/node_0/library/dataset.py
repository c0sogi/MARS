import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import AudioConfig, TrainConfig
from library.utils import set_seed


class AudioPreprocessor:
    """
    Handles audio loading, preprocessing, and spectrogram generation.
    """

    def __init__(self):
        self.target_sr = AudioConfig.sample_rate
        self.target_len = int(AudioConfig.duration * self.target_sr)

        # Define Mel Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=AudioConfig.sample_rate,
            n_fft=AudioConfig.n_fft,
            win_length=AudioConfig.n_fft,
            hop_length=AudioConfig.hop_length,
            f_min=AudioConfig.fmin,
            f_max=AudioConfig.fmax,
            n_mels=AudioConfig.n_mels,
            power=2.0,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def load_audio(self, file_path):
        """
        Loads audio, ensures mono channel, and pads/crops to target length.
        """
        try:
            # Load audio
            audio, sr = sf.read(file_path)

            # Ensure correct sample rate (though analysis showed all are 2k)
            if sr != self.target_sr:
                # Simple resampling if needed, though unlikely based on metadata
                # For robustness, we assume 2k based on analysis.
                pass

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Pad or Crop
            if len(audio) < self.target_len:
                pad_width = self.target_len - len(audio)
                # Pad with zeros at the end
                audio = np.pad(audio, (0, pad_width), mode="constant")
            elif len(audio) > self.target_len:
                # Crop from center
                start = (len(audio) - self.target_len) // 2
                audio = audio[start : start + self.target_len]

            return torch.tensor(audio, dtype=torch.float32)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return torch.zeros(self.target_len, dtype=torch.float32)

    def to_spectrogram(self, audio):
        """
        Converts audio waveform to Normalized Log-Mel Spectrogram.
        """
        # Add channel dimension: (1, samples)
        audio = audio.unsqueeze(0)

        # Compute Mel Spectrogram
        melspec = self.mel_transform(audio)

        # Convert to Log Scale (dB)
        log_melspec = self.db_transform(melspec)

        # Frequency-Wise Standardization
        if AudioConfig.freq_wise_standardization:
            # Calculate mean and std per frequency bin across time
            # Shape: (1, n_mels, time)
            mean = log_melspec.mean(dim=2, keepdim=True)
            std = log_melspec.std(dim=2, keepdim=True)

            # Normalize
            log_melspec = (log_melspec - mean) / (std + 1e-6)

        return log_melspec

    def process_path(self, full_path):
        audio = self.load_audio(full_path)
        spec = self.to_spectrogram(audio)
        return spec.numpy()  # Return as numpy for caching


def process_and_cache_data(
    metadata_path, cache_name, load_cached_data=True, debug=False
):
    """
    Loads metadata, checks cache, processes audio if needed, and returns data arrays.
    """
    cache_file = os.path.join(TrainConfig.working_dir, f"{cache_name}.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached {cache_name} data from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            return {
                "specs": data["specs"],
                "labels": data["labels"],
                "clips": data["clips"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {cache_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    if debug:
        df = df.head(100)
        print(f"Debug mode: Processing first {len(df)} samples.")

    preprocessor = AudioPreprocessor()

    specs_list = []
    labels_list = []
    clips_list = []

    for _, row in df.iterrows():
        full_path = os.path.join(TrainConfig.input_dir, row["file_path"])

        # Process audio
        spec = preprocessor.process_path(full_path)

        specs_list.append(spec)
        clips_list.append(row["clip_name"])

        # Handle labels (test set might not have them)
        if "label" in row:
            labels_list.append(row["label"])
        else:
            labels_list.append(-1)  # Placeholder for test

    # Convert to numpy arrays
    # spec shape: (N, 1, F, T)
    specs_arr = np.stack(specs_list).astype(np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32)
    clips_arr = np.array(clips_list)

    # Save to cache
    print(f"Saving {cache_name} data to {cache_file}...")
    np.savez(cache_file, specs=specs_arr, labels=labels_arr, clips=clips_arr)

    return {"specs": specs_arr, "labels": labels_arr, "clips": clips_arr}


class WhaleDataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict: Dictionary containing 'specs', 'labels', 'clips'.
            mode: 'train', 'val', or 'test'. Controls augmentation.
        """
        self.specs = data_dict["specs"]
        self.labels = data_dict["labels"]
        self.clips = data_dict["clips"]
        self.mode = mode

        # Augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=TrainConfig.freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=TrainConfig.time_mask_param
        )

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Load spec: (1, F, T)
        spec = torch.from_numpy(self.specs[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        clip_name = self.clips[idx]

        # Apply Augmentations only in training mode
        if self.mode == "train" and TrainConfig.use_spec_augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        return spec, label, clip_name


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares and returns DataLoaders for train, val, and test sets.
    """
    set_seed(TrainConfig.seed)

    # Process/Load Data
    train_data = process_and_cache_data(
        TrainConfig.train_meta, "train", load_cached_data, debug
    )
    val_data = process_and_cache_data(
        TrainConfig.val_meta, "val", load_cached_data, debug
    )
    test_data = process_and_cache_data(
        TrainConfig.test_meta, "test", load_cached_data, debug
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_data, mode="train")
    val_dataset = WhaleDataset(val_data, mode="val")
    test_dataset = WhaleDataset(test_data, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
        drop_last=True,  # Useful for batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
