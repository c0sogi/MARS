import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import audio_config, path_config, label_config, train_config
from library.utils import set_seed, LabelManager
from library.transforms import get_spectrogram_transform


def load_background_noise_data():
    """
    Loads all background noise files into memory for dynamic silence generation.
    Returns:
        List[torch.Tensor]: List of audio waveforms.
    """
    if not os.path.exists(path_config.background_noise_dir):
        return []

    noise_files = [
        f for f in os.listdir(path_config.background_noise_dir) if f.endswith(".wav")
    ]
    noise_data = []
    for f in noise_files:
        path = os.path.join(path_config.background_noise_dir, f)
        try:
            # Load waveform
            waveform, sr = torchaudio.load(path)

            # Resample if necessary
            if sr != audio_config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, audio_config.sample_rate)
                waveform = resampler(waveform)

            # Ensure it's mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            noise_data.append(waveform)
        except Exception as e:
            print(f"Warning: Failed to load noise file {f}: {e}")

    return noise_data


class SpeechCommandsDataset(Dataset):
    def __init__(
        self, df, label_manager, transform=None, noise_data=None, is_training=False
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe with 'filepath' and 'fine_label'.
            label_manager (LabelManager): Instance to handle label-to-index conversion.
            transform (nn.Module): Spectrogram transform.
            noise_data (List[Tensor]): List of background noise waveforms.
            is_training (bool): Flag for augmentation/randomness.
        """
        self.df = df.reset_index(drop=True)
        self.label_manager = label_manager
        self.transform = transform
        self.noise_data = noise_data
        self.is_training = is_training
        self.target_length = audio_config.num_samples

    def __len__(self):
        return len(self.df)

    def _get_audio(self, filepath, fine_label):
        """
        Loads audio from disk or generates silence.
        """
        # Logic for Silence Generation
        # If it's a placeholder OR a background noise file, we generate/crop from noise data
        is_noise_file = (
            "_background_noise_" in filepath or filepath == "noise_placeholder"
        )

        if fine_label == label_config.silence_label and is_noise_file:
            if not self.noise_data:
                # Fallback: return zeros
                return torch.zeros(1, self.target_length)

            # Select random noise file
            noise_wave = random.choice(self.noise_data)
            noise_len = noise_wave.shape[1]

            if noise_len <= self.target_length:
                # Pad if too short
                pad_amt = self.target_length - noise_len
                return F.pad(noise_wave, (0, pad_amt))

            # Random crop
            # Ensure we don't go out of bounds
            max_start = noise_len - self.target_length
            start = random.randint(0, max_start)
            return noise_wave[:, start : start + self.target_length]

        # Regular File Loading
        full_path = os.path.join(path_config.input_dir, filepath)

        if not os.path.exists(full_path):
            # Fallback for missing files
            return torch.zeros(1, self.target_length)

        waveform, sr = torchaudio.load(full_path)

        # Resample
        if sr != audio_config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, audio_config.sample_rate)
            waveform = resampler(waveform)

        # Mix to Mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Fix Length (Pad or Crop)
        current_len = waveform.shape[1]

        if current_len < self.target_length:
            # Pad
            pad_amt = self.target_length - current_len
            waveform = F.pad(waveform, (0, pad_amt))
        elif current_len > self.target_length:
            # Crop
            if self.is_training:
                # Random crop for training
                max_start = current_len - self.target_length
                start = random.randint(0, max_start)
            else:
                # Center crop for validation/test
                start = (current_len - self.target_length) // 2

            waveform = waveform[:, start : start + self.target_length]

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        fine_label = row["fine_label"]

        # 1. Get Audio Waveform
        waveform = self._get_audio(filepath, fine_label)

        # 2. Apply Transforms (Spectrogram)
        if self.transform:
            spec = self.transform(waveform)
        else:
            spec = waveform

        # 3. Get Label Index
        # For test set, fine_label might be a dummy, but we still return an index
        try:
            label_idx = self.label_manager.convert_label_to_idx(fine_label)
        except ValueError:
            # Fallback if label not found (should not happen in train/val)
            label_idx = 0

        return spec, label_idx


def _extract_fine_label(row):
    """
    Helper to extract the fine-grained folder name from the filepath.
    """
    # If explicitly labeled silence in metadata (e.g. noise files)
    if row["label"] == "silence":
        return "silence"

    # Parse filepath: train/audio/<label>/<file.wav>
    # Normalize separators
    path = row["filepath"].replace("\\", "/")
    parts = path.split("/")

    if len(parts) >= 2:
        folder = parts[-2]
        if folder == "_background_noise_":
            return "silence"
        return folder

    return "unknown"


def get_train_val_datasets(label_manager, load_cached_data=True):
    """
    Generates training and validation datasets with balancing and caching.

    Args:
        label_manager (LabelManager): For label encoding.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        Tuple[SpeechCommandsDataset, SpeechCommandsDataset]: Train and Val datasets.
    """
    # Ensure cache directory
    cache_dir = path_config.working_dir
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, "train_balanced.parquet")

    # --- Prepare Validation Data ---
    # Validation is simple: load metadata, add fine_label, no balancing
    df_val = pd.read_csv(path_config.val_metadata)
    df_val["fine_label"] = df_val.apply(_extract_fine_label, axis=1)

    # --- Prepare Training Data ---
    if load_cached_data and os.path.exists(train_cache_path):
        df_train_balanced = pd.read_parquet(train_cache_path)
    else:
        df_train = pd.read_csv(path_config.train_metadata)
        df_train["fine_label"] = df_train.apply(_extract_fine_label, axis=1)

        # Filter out original silence files (we will synthesize them)
        df_train = df_train[df_train["fine_label"] != "silence"]

        # Split into Targets and Aux
        targets_mask = df_train["fine_label"].isin(label_config.target_labels)
        df_targets = df_train[targets_mask]
        df_aux = df_train[~targets_mask]

        # Upsample Targets
        balanced_parts = []
        for label in label_config.target_labels:
            subset = df_targets[df_targets["fine_label"] == label]
            if len(subset) == 0:
                continue
            # Upsample to target count
            resampled = subset.sample(
                n=label_config.target_upsample_count,
                replace=True,
                random_state=train_config.seed,
            )
            balanced_parts.append(resampled)

        if balanced_parts:
            df_targets_balanced = pd.concat(balanced_parts)
        else:
            df_targets_balanced = pd.DataFrame(columns=df_train.columns)

        # Synthesize Silence Rows
        # We create placeholder rows that __getitem__ will recognize
        silence_data = {
            "filepath": ["noise_placeholder"] * label_config.target_upsample_count,
            "label": ["silence"] * label_config.target_upsample_count,
            "subject_id": ["noise"] * label_config.target_upsample_count,
            "fine_label": ["silence"] * label_config.target_upsample_count,
        }
        df_silence = pd.DataFrame(silence_data)

        # Combine All
        df_train_balanced = pd.concat(
            [df_targets_balanced, df_aux, df_silence], ignore_index=True
        )

        # Shuffle
        df_train_balanced = df_train_balanced.sample(
            frac=1, random_state=train_config.seed
        ).reset_index(drop=True)

        # Save Cache
        df_train_balanced.to_parquet(train_cache_path)

    # --- Load Resources ---
    noise_data = load_background_noise_data()
    transform = get_spectrogram_transform()

    # --- Create Datasets ---
    train_dataset = SpeechCommandsDataset(
        df_train_balanced,
        label_manager,
        transform=transform,
        noise_data=noise_data,
        is_training=True,
    )

    val_dataset = SpeechCommandsDataset(
        df_val,
        label_manager,
        transform=transform,
        noise_data=noise_data,
        is_training=False,
    )

    return train_dataset, val_dataset


def get_test_dataset(label_manager):
    """
    Generates the test dataset.
    """
    df_test = pd.read_csv(path_config.test_metadata)

    # Assign a dummy fine_label that exists in the label manager.
    # This is required because Dataset converts label to idx.
    # We use the first available class.
    dummy_label = label_manager.classes[0]
    df_test["fine_label"] = dummy_label

    transform = get_spectrogram_transform()

    return SpeechCommandsDataset(
        df_test, label_manager, transform=transform, noise_data=None, is_training=False
    )
