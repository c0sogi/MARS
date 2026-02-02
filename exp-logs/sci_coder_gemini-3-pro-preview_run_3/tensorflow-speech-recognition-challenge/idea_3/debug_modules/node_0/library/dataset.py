import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import PATHS, AUDIO_CONFIG, TRAIN_CONFIG, LABEL_TO_IDX


class LogMelSpectrogram(nn.Module):
    """
    Converts waveform to Log-Mel Spectrogram with specific parameters
    optimized for ResNet34 input and speech command duration.
    """

    def __init__(self):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_CONFIG["sample_rate"],
            n_fft=AUDIO_CONFIG["n_fft"],
            win_length=AUDIO_CONFIG["n_fft"],
            hop_length=AUDIO_CONFIG["hop_length"],
            n_mels=AUDIO_CONFIG["n_mels"],
            f_min=AUDIO_CONFIG["f_min"],
            f_max=AUDIO_CONFIG["f_max"],
            normalized=True,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform):
        # waveform: (1, samples)
        spec = self.mel_spectrogram(waveform)
        log_spec = self.amplitude_to_db(spec)
        return log_spec


class SpecAugment(nn.Module):
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
    Constraints:
    - Mask value is min(spectrogram)
    - Time mask width < 20% of duration
    """

    def __init__(self, freq_mask_param=10, time_mask_param=None):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        # If time_mask_param is not set, it will be calculated dynamically based on 20% rule
        self.time_mask_param = time_mask_param

    def forward(self, spec):
        # spec shape: (1, n_mels, time_steps)
        # We work on the last two dimensions

        # Calculate min value for masking
        mask_value = spec.min().item()

        _, n_mels, time_steps = spec.shape

        # 1. Frequency Masking
        f = np.random.randint(0, self.freq_mask_param + 1)
        f0 = np.random.randint(0, n_mels - f + 1)
        spec[:, f0 : f0 + f, :] = mask_value

        # 2. Time Masking
        # Constraint: Mask width < 20% of total time steps
        max_time_mask = int(0.2 * time_steps)
        if self.time_mask_param is None:
            t_param = max_time_mask
        else:
            t_param = min(self.time_mask_param, max_time_mask)

        if t_param > 0:
            t = np.random.randint(0, t_param + 1)
            t0 = np.random.randint(0, time_steps - t + 1)
            spec[:, :, t0 : t0 + t] = mask_value

        return spec


def get_balanced_dataframes(load_cached_data=True):
    """
    Loads metadata and creates a balanced training dataframe.
    Undersamples 'unknown' and oversamples 'silence'.
    Caches the result to avoid re-processing.
    """
    cache_path = os.path.join(PATHS["cache_dir"], "balanced_train.parquet")
    os.makedirs(PATHS["cache_dir"], exist_ok=True)

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_train = pd.read_parquet(cache_path)
            df_val = pd.read_csv(PATHS["val_csv"])
            df_test = pd.read_csv(PATHS["test_csv"])
            return df_train, df_val, df_test
        except Exception:
            pass  # Fallback to creation

    # 2. Create from scratch
    df_train_raw = pd.read_csv(PATHS["train_csv"])
    df_val = pd.read_csv(PATHS["val_csv"])
    df_test = pd.read_csv(PATHS["test_csv"])

    # Target labels (excluding unknown and silence)
    target_labels = [l for l in LABEL_TO_IDX.keys() if l not in ["unknown", "silence"]]

    # Calculate median count of target labels
    counts = df_train_raw["label"].value_counts()
    target_counts = [counts.get(l, 0) for l in target_labels]
    median_count = int(np.median(target_counts))

    balanced_rows = []

    # Process each label
    for label in LABEL_TO_IDX.keys():
        df_label = df_train_raw[df_train_raw["label"] == label]
        n_samples = len(df_label)

        if n_samples == 0:
            continue

        if label == "unknown":
            # Undersample
            if n_samples > median_count:
                df_label = df_label.sample(
                    n=median_count, random_state=TRAIN_CONFIG["seed"]
                )
            balanced_rows.append(df_label)

        elif label == "silence":
            # Oversample (Replicate rows)
            # We replicate the few silence files so the Dataset class can randomly crop them
            if n_samples > 0:
                n_repeats = int(np.ceil(median_count / n_samples))
                df_repeated = pd.concat([df_label] * n_repeats, ignore_index=True)
                # Trim to exact median count
                df_repeated = df_repeated.iloc[:median_count]
                balanced_rows.append(df_repeated)

        else:
            # Keep target labels as is (or could cap them if needed, but usually we keep)
            # Strategy says: "Undersampling the unknown class... oversampling silence"
            # Implicitly implies keeping target classes as is.
            balanced_rows.append(df_label)

    df_train_balanced = (
        pd.concat(balanced_rows, ignore_index=True)
        .sample(frac=1, random_state=TRAIN_CONFIG["seed"])
        .reset_index(drop=True)
    )

    # Cache the result
    df_train_balanced.to_parquet(cache_path)

    return df_train_balanced, df_val, df_test


class SpeechCommandsDataset(Dataset):
    def __init__(self, df, audio_dir, is_training=False, transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            audio_dir (str): Base directory for audio files.
            is_training (bool): Whether to apply augmentation and random cropping for silence.
            transform (nn.Module): Optional transform to apply (usually None, as we do it internally).
        """
        self.df = df
        self.audio_dir = audio_dir
        self.is_training = is_training

        # Audio loading config
        self.target_sr = AUDIO_CONFIG["sample_rate"]
        self.target_len = AUDIO_CONFIG["num_samples"]

        # Processing pipeline
        self.spectrogram_transform = LogMelSpectrogram()
        self.augment = SpecAugment() if is_training else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_dir, row["filepath"])
        label_str = row["label"]
        label_idx = LABEL_TO_IDX.get(
            label_str, LABEL_TO_IDX["unknown"]
        )  # Default to unknown if issue

        # 1. Load Audio
        # Handle Silence (Background Noise) specially
        if label_str == "silence":
            try:
                # Load full file
                wav, sr = sf.read(filepath, dtype="float32")

                # Resample if needed (unlikely for provided data, but safety)
                if sr != self.target_sr:
                    # Simple resampling not implemented to avoid extra deps,
                    # assuming data is 16k based on analysis.
                    # If needed, we'd use torchaudio.functional.resample
                    pass

                # Crop
                if len(wav) > self.target_len:
                    if self.is_training:
                        # Random crop
                        start = np.random.randint(0, len(wav) - self.target_len)
                    else:
                        # Center crop for validation
                        start = (len(wav) - self.target_len) // 2
                    wav = wav[start : start + self.target_len]
                else:
                    # Pad if too short (unlikely for background noise)
                    pad_len = self.target_len - len(wav)
                    wav = np.pad(wav, (0, pad_len), "constant")

            except Exception as e:
                # Fallback for corrupt files: return silence
                wav = np.zeros(self.target_len, dtype="float32")

        else:
            # Standard files
            try:
                wav, sr = sf.read(filepath, dtype="float32")

                # Pad or Truncate
                if len(wav) > self.target_len:
                    wav = wav[: self.target_len]
                elif len(wav) < self.target_len:
                    pad_len = self.target_len - len(wav)
                    wav = np.pad(wav, (0, pad_len), "constant")
            except Exception:
                wav = np.zeros(self.target_len, dtype="float32")

        # Convert to Tensor (1, Time)
        waveform = torch.from_numpy(wav).float().unsqueeze(0)

        # 2. Generate Spectrogram
        # Shape: (1, n_mels, time)
        spec = self.spectrogram_transform(waveform)

        # 3. Augment (if training)
        if self.augment is not None:
            spec = self.augment(spec)

        return spec, label_idx
