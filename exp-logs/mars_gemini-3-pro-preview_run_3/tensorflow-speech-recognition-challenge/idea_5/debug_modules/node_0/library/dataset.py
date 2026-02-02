import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class SpeechCommandsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath', 'label', etc.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and silence handling.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.target_length = Config.N_SAMPLES

        # Define Multi-Resolution MelSpectrogram transforms
        # We use a fixed n_fft that covers the largest window size (960) -> 2048
        self.mel_transforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=Config.SAMPLE_RATE,
                    n_fft=2048,
                    win_length=win_size,
                    hop_length=Config.HOP_LENGTH,
                    n_mels=Config.N_MELS,
                    f_min=Config.F_MIN,
                    f_max=Config.F_MAX,
                    power=2.0,
                )
                for win_size in Config.WINDOW_SIZES
            ]
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # Augmentation
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )

        # Cache background noise files for 'silence' class to avoid repeated IO
        self.silence_cache = {}
        if "label" in self.df.columns:
            silence_files = self.df[self.df["label"] == "silence"]["filepath"].unique()
            for rel_path in silence_files:
                full_path = os.path.join(Config.INPUT_DIR, rel_path)
                if os.path.exists(full_path):
                    try:
                        wav, sr = torchaudio.load(full_path)
                        # Resample if necessary (though analysis says 16k)
                        if sr != Config.SAMPLE_RATE:
                            resampler = torchaudio.transforms.Resample(
                                sr, Config.SAMPLE_RATE
                            )
                            wav = resampler(wav)
                        # Ensure mono
                        if wav.shape[0] > 1:
                            wav = torch.mean(wav, dim=0, keepdim=True)
                        self.silence_cache[rel_path] = wav
                    except Exception as e:
                        print(f"Warning: Failed to load silence file {rel_path}: {e}")

    def __len__(self):
        return len(self.df)

    def _get_audio(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"] if "label" in row else "unknown"

        # Handle Silence (Background Noise)
        if label == "silence" and filepath in self.silence_cache:
            full_wav = self.silence_cache[filepath]
            wav_len = full_wav.shape[-1]

            if self.mode == "train":
                # Random crop for training
                if wav_len > self.target_length:
                    offset = torch.randint(
                        0, wav_len - self.target_length + 1, (1,)
                    ).item()
                    wav = full_wav[:, offset : offset + self.target_length]
                else:
                    wav = full_wav
            else:
                # Deterministic crop for val/test (take the beginning or center)
                # Taking the beginning is safer for consistency
                if wav_len > self.target_length:
                    wav = full_wav[:, : self.target_length]
                else:
                    wav = full_wav
        else:
            # Standard Audio Loading
            full_path = os.path.join(Config.INPUT_DIR, filepath)
            if not os.path.exists(full_path):
                # Fallback for missing files (should not happen based on validation)
                return torch.zeros(1, self.target_length)

            wav, sr = torchaudio.load(full_path)

            if sr != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
                wav = resampler(wav)

            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

        # Pad or Crop to fixed length
        c, n = wav.shape
        if n < self.target_length:
            # Pad
            padding = self.target_length - n
            # Center padding usually works well, but right padding is standard for RNNs
            wav = torch.nn.functional.pad(wav, (0, padding))
        elif n > self.target_length:
            # Crop (Center crop for non-silence standard files)
            start = (n - self.target_length) // 2
            wav = wav[:, start : start + self.target_length]

        return wav

    def __getitem__(self, idx):
        # 1. Load Audio
        waveform = self._get_audio(idx)

        # 2. Compute Multi-Resolution Spectrograms
        # waveform shape: (1, 16000)
        specs = []
        for transform in self.mel_transforms:
            # Spec shape: (1, n_mels, time)
            spec = transform(waveform)
            specs.append(spec)

        # Stack along channel dimension: (3, n_mels, time)
        multi_res_spec = torch.cat(specs, dim=0)

        # Convert to Log-Mel (dB)
        multi_res_spec = self.amplitude_to_db(multi_res_spec)

        # 3. Augmentation (Train only)
        if self.mode == "train":
            # Apply masking. Note: Torchaudio masks apply to (..., freq, time).
            # We apply to the whole 3-channel block. The mask will be applied
            # independently per channel if passed as (3, F, T) or same if we force it.
            # Standard torchaudio behavior on (C, F, T) is independent masks or error depending on version.
            # To ensure structural consistency (masking same time step across channels),
            # we can apply mask to one and broadcast, but independent masking is a stronger regularizer.
            # We will apply directly.
            try:
                multi_res_spec = self.freq_masking(multi_res_spec)
                multi_res_spec = self.time_masking(multi_res_spec)
            except:
                pass  # Fallback if dimension mismatch occurs in specific versions

        # 4. Label
        label_str = self.df.iloc[idx]["label"]
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])

        return multi_res_spec, label_id

    def get_filename(self, idx):
        return self.df.iloc[idx]["filepath"]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,  # Argument kept for compatibility, though we load from CSV
):
    """
    Creates DataLoaders for train, validation, and test sets.
    Implements WeightedRandomSampler for training to handle class imbalance.
    """

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Calculate Class Weights for Training
    # We want to balance the batches so the model sees 'silence', 'yes', 'no' etc. as often as 'unknown'
    label_counts = df_train["label"].value_counts()

    # Calculate weight per label: 1.0 / count
    weights_per_label = {label: 1.0 / count for label, count in label_counts.items()}

    # Assign a weight to each sample in the dataframe
    # Use .map for efficiency
    sample_weights = df_train["label"].map(weights_per_label).fillna(0).values
    sample_weights = torch.from_numpy(sample_weights).double()

    # 3. Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Create Datasets
    train_dataset = SpeechCommandsDataset(df_train, mode="train")
    val_dataset = SpeechCommandsDataset(df_val, mode="val")
    test_dataset = SpeechCommandsDataset(df_test, mode="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Mutually exclusive with shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches (Balanced Sampling)")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
