import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import (
    MelSpectrogram,
    AmplitudeToDB,
    FrequencyMasking,
    TimeMasking,
)

from library.config import Config
from library.utils import LabelMapper, set_seed


class SpeechCommandDataset(Dataset):
    """
    PyTorch Dataset for Speech Command Recognition.
    Handles on-the-fly waveform loading, silence synthesis, noise injection,
    and Log-Mel Spectrogram generation.
    """

    def __init__(self, data_records, noise_files=None, mode="train"):
        """
        Args:
            data_records (list of dict): List of {'filepath': str, 'label': str}.
            noise_files (list of str): List of paths to background noise wav files.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data_records
        self.noise_files = noise_files if noise_files else []
        self.mode = mode
        self.mapper = LabelMapper()

        # Audio settings
        self.sample_rate = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.num_samples = int(self.sample_rate * self.duration)

        # Spectrogram transforms
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=80)

        # Augmentations
        self.freq_mask = FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)
        self.time_mask = TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)

        # Preload noise files for efficiency if in training mode
        self.loaded_noise = []
        if self.mode == "train" and self.noise_files:
            for nf in self.noise_files:
                try:
                    wav, sr = torchaudio.load(nf)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        wav = resampler(wav)
                    # Convert to mono
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)
                    self.loaded_noise.append(wav)
                except Exception as e:
                    print(f"Warning: Failed to load noise file {nf}: {e}")

    def __len__(self):
        return len(self.data)

    def _load_audio(self, filepath):
        """Loads and pads/truncates audio to fixed length."""
        try:
            waveform, sr = torchaudio.load(filepath)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or Truncate
            c, t = waveform.shape
            if t > self.num_samples:
                # Random crop for training, center crop for val/test
                if self.mode == "train":
                    start = random.randint(0, t - self.num_samples)
                    waveform = waveform[:, start : start + self.num_samples]
                else:
                    start = (t - self.num_samples) // 2
                    waveform = waveform[:, start : start + self.num_samples]
            elif t < self.num_samples:
                padding = self.num_samples - t
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform

        except Exception as e:
            # Return silent tensor on failure
            return torch.zeros((1, self.num_samples))

    def _synthesize_silence(self):
        """Generates a silence sample by cropping background noise."""
        if not self.loaded_noise:
            return torch.zeros((1, self.num_samples))

        noise_wav = random.choice(self.loaded_noise)
        c, t = noise_wav.shape

        if t <= self.num_samples:
            # Loop if too short
            repeats = (self.num_samples // t) + 1
            noise_wav = noise_wav.repeat(1, repeats)
            t = noise_wav.shape[1]

        start = random.randint(0, t - self.num_samples)
        return noise_wav[:, start : start + self.num_samples]

    def _add_noise(self, waveform):
        """Injects background noise into the waveform."""
        if not self.loaded_noise or random.random() > Config.NOISE_PROB:
            return waveform

        noise = self._synthesize_silence()

        # Calculate Signal and Noise Power
        sig_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        if sig_power == 0 or noise_power == 0:
            return waveform

        # Target SNR
        snr_db = random.uniform(Config.NOISE_SNR_MIN, Config.NOISE_SNR_MAX)
        target_ratio = 10 ** (snr_db / 10)

        scale = torch.sqrt(sig_power / (target_ratio * noise_power))
        return waveform + scale * noise

    def __getitem__(self, idx):
        record = self.data[idx]
        label_str = record["label"]

        # 1. Get Waveform
        if label_str == Config.SILENCE_LABEL:
            # Synthesize silence (ignore filepath if it's a placeholder)
            waveform = self._synthesize_silence()
        else:
            # Load actual file
            filepath = os.path.join(Config.INPUT_ROOT, record["filepath"])
            waveform = self._load_audio(filepath)

            # Apply Noise Injection (Train only, not on silence class itself usually)
            if self.mode == "train":
                waveform = self._add_noise(waveform)

        # 2. Convert to Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 3. Apply SpecAugment (Train only)
        if self.mode == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # 4. Label Handling
        if self.mode == "test":
            # Dummy label for test
            label_idx = -1
        else:
            try:
                label_idx = self.mapper.to_index(label_str)
            except ValueError:
                # Fallback for unexpected labels
                label_idx = self.mapper.to_index(Config.UNKNOWN_LABEL)

        return spec, label_idx


def get_dataloaders(load_cached_data=False):
    """
    Generates DataLoaders for Train, Validation, and Test.
    Implements variance-aware balancing and caching of the balanced dataset list.
    """
    set_seed(Config.SEED)

    # Paths
    cache_path = os.path.join(Config.WORKING_DIR, "train_balanced.parquet")

    # -------------------------------------------------------------------------
    # 1. Prepare Training Data
    # -------------------------------------------------------------------------
    train_records = []
    noise_files = []

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached balanced training data from {cache_path}")
        df_train_balanced = pd.read_parquet(cache_path)
        train_records = df_train_balanced.to_dict("records")

        # We still need to identify noise files for synthesis
        df_raw = pd.read_csv(Config.TRAIN_CSV)
        noise_df = df_raw[df_raw["label"] == "silence"]
        noise_files = [
            os.path.join(Config.INPUT_ROOT, f) for f in noise_df["filepath"].tolist()
        ]

    else:
        print("Generating balanced training data...")
        df_train = pd.read_csv(Config.TRAIN_CSV)

        # Identify background noise files
        # In metadata generation, background noise was labeled 'silence'
        noise_df = df_train[df_train["label"] == "silence"]
        noise_files = [
            os.path.join(Config.INPUT_ROOT, f) for f in noise_df["filepath"].tolist()
        ]

        # Filter out original silence entries from training list (we will synthesize them)
        df_clean = df_train[df_train["label"] != "silence"]

        # Separate Targets and Aux
        targets = df_clean[df_clean["label"].isin(Config.TARGET_LABELS)]
        aux = df_clean[~df_clean["label"].isin(Config.TARGET_LABELS)]

        final_records = []

        # 1. Add Aux (Natural Counts)
        final_records.extend(aux.to_dict("records"))

        # 2. Add Targets (Upsampled to ~2000)
        TARGET_COUNT = 2000
        for label in Config.TARGET_LABELS:
            subset = targets[targets["label"] == label]
            if len(subset) == 0:
                continue

            # Upsample if needed, or downsample if too many (unlikely here)
            if len(subset) < TARGET_COUNT:
                upsampled = subset.sample(
                    n=TARGET_COUNT, replace=True, random_state=Config.SEED
                )
                final_records.extend(upsampled.to_dict("records"))
            else:
                # Keep natural if > 2000 (rarely happens in this dataset for targets)
                final_records.extend(subset.to_dict("records"))

        # 3. Add Synthesized Silence (Virtual Records)
        # We add placeholder records. The Dataset class handles generation.
        for _ in range(TARGET_COUNT):
            final_records.append(
                {
                    "filepath": "virtual_silence",
                    "label": Config.SILENCE_LABEL,
                    "subject_id": "synth",
                }
            )

        # Shuffle
        random.shuffle(final_records)
        train_records = final_records

        # Cache
        pd.DataFrame(train_records).to_parquet(cache_path)
        print(f"Saved balanced training data to {cache_path}")

    # -------------------------------------------------------------------------
    # 2. Prepare Validation Data
    # -------------------------------------------------------------------------
    df_val = pd.read_csv(Config.VAL_CSV)
    val_records = df_val.to_dict("records")

    # Inject some silence into validation for proper metric evaluation
    # (The original val set might have 0 or 1 silence file)
    VAL_SILENCE_COUNT = 200
    for _ in range(VAL_SILENCE_COUNT):
        val_records.append(
            {
                "filepath": "virtual_silence",
                "label": Config.SILENCE_LABEL,
                "subject_id": "synth",
            }
        )

    # -------------------------------------------------------------------------
    # 3. Prepare Test Data
    # -------------------------------------------------------------------------
    df_test = pd.read_csv(Config.TEST_CSV)
    test_records = df_test.to_dict("records")

    # -------------------------------------------------------------------------
    # 4. Create Datasets and Loaders
    # -------------------------------------------------------------------------
    train_dataset = SpeechCommandDataset(
        train_records, noise_files=noise_files, mode="train"
    )
    val_dataset = SpeechCommandDataset(val_records, noise_files=noise_files, mode="val")
    test_dataset = SpeechCommandDataset(test_records, noise_files=None, mode="test")

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

    print(f"DataLoaders Created:")
    print(f"  Train Batches: {len(train_loader)} (Samples: {len(train_dataset)})")
    print(f"  Val Batches:   {len(val_loader)} (Samples: {len(val_dataset)})")
    print(f"  Test Batches:  {len(test_loader)} (Samples: {len(test_dataset)})")

    return train_loader, val_loader, test_loader
