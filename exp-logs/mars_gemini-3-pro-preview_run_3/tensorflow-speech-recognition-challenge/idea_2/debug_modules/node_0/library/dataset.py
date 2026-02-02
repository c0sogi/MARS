import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


def apply_spec_augment(spec):
    """
    Applies SpecAugment (Frequency and Time Masking) to a spectrogram.
    Masked regions are filled with the minimum value of the spectrogram.

    Args:
        spec (torch.Tensor): Input spectrogram of shape (channels, n_mels, time).

    Returns:
        torch.Tensor: Augmented spectrogram.
    """
    # spec shape: (C, F, T)
    min_val = spec.min()
    _, n_mels, n_steps = spec.shape

    # Frequency Masking
    f_param = Config.FREQ_MASK_PARAM
    f_width = torch.randint(0, f_param + 1, (1,)).item()
    if f_width > 0 and f_width < n_mels:
        f_start = torch.randint(0, n_mels - f_width, (1,)).item()
        spec[:, f_start : f_start + f_width, :] = min_val

    # Time Masking
    t_param = Config.TIME_MASK_PARAM
    # Ensure mask is not too large (e.g., < 20% of time) if t_param is large,
    # but here we trust Config.TIME_MASK_PARAM is set appropriately (20).
    t_width = torch.randint(0, t_param + 1, (1,)).item()
    if t_width > 0 and t_width < n_steps:
        t_start = torch.randint(0, n_steps - t_width, (1,)).item()
        spec[:, :, t_start : t_start + t_width] = min_val

    return spec


class MixUpCollate:
    """
    Collate function that applies MixUp augmentation to a batch.
    """

    def __init__(self, alpha=Config.MIXUP_ALPHA, num_classes=Config.NUM_CLASSES):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        # batch is a list of tuples (spec, label_id)
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)

        batch_size = inputs.size(0)

        # Convert targets to one-hot
        targets_one_hot = torch.zeros(batch_size, self.num_classes).scatter_(
            1, targets.view(-1, 1), 1
        )

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # MixUp
        index = torch.randperm(batch_size)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index]

        return mixed_inputs, mixed_targets


class SpeechCommandDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.target_sr = Config.SAMPLE_RATE
        self.target_len = Config.N_SAMPLES

        # Define MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Cache background noise files for training to allow efficient random cropping
        self.noise_cache = {}
        if self.mode == "train":
            silence_rows = self.df[self.df["label"] == "silence"]
            unique_noise_paths = silence_rows["filepath"].unique()
            for path in unique_noise_paths:
                full_path = os.path.join(Config.INPUT_ROOT, path)
                if os.path.exists(full_path):
                    wav, sr = torchaudio.load(full_path)
                    if sr != self.target_sr:
                        wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
                    self.noise_cache[path] = wav

    def __len__(self):
        return len(self.df)

    def _load_audio(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"]
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        waveform = None

        # 1. Handle Silence (Background Noise)
        if label == "silence":
            if self.mode == "train" and filepath in self.noise_cache:
                # Use cached full noise file
                noise_wav = self.noise_cache[filepath]
                noise_len = noise_wav.shape[1]

                if noise_len > self.target_len:
                    # Random crop
                    start = torch.randint(0, noise_len - self.target_len, (1,)).item()
                    waveform = noise_wav[:, start : start + self.target_len]
                else:
                    waveform = noise_wav
            else:
                # Load from disk (Val/Test or uncached)
                if os.path.exists(full_path):
                    waveform, sr = torchaudio.load(full_path)
                    if sr != self.target_sr:
                        waveform = torchaudio.transforms.Resample(sr, self.target_sr)(
                            waveform
                        )

                    # For validation, take a deterministic crop (e.g., center)
                    if waveform.shape[1] > self.target_len:
                        start = (waveform.shape[1] - self.target_len) // 2
                        waveform = waveform[:, start : start + self.target_len]

        # 2. Handle Regular Audio (or if silence loading failed/fell through)
        if waveform is None:
            if not os.path.exists(full_path):
                # Fallback for missing files (should not happen based on validation)
                return torch.zeros(1, self.target_len)

            waveform, sr = torchaudio.load(full_path)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

        # 3. Pad or Crop to fixed length
        current_len = waveform.shape[1]

        if current_len > self.target_len:
            if self.mode == "train":
                # Random crop (time shift augmentation)
                max_shift = current_len - self.target_len
                start = torch.randint(0, max_shift + 1, (1,)).item()
                waveform = waveform[:, start : start + self.target_len]
            else:
                # Center crop
                start = (current_len - self.target_len) // 2
                waveform = waveform[:, start : start + self.target_len]
        elif current_len < self.target_len:
            # Pad with zeros (center padding)
            pad_amt = self.target_len - current_len
            pad_left = pad_amt // 2
            pad_right = pad_amt - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))

        return waveform

    def __getitem__(self, idx):
        waveform = self._load_audio(idx)

        # Compute Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)

        # Apply SpecAugment (Train only)
        if self.mode == "train":
            log_mel_spec = apply_spec_augment(log_mel_spec)

        # Get Label ID
        label_str = self.df.iloc[idx]["label"]
        label_id = Config.LABEL2ID[label_str]

        return log_mel_spec, label_id


def get_balanced_dataloaders():
    """
    Creates balanced training and validation DataLoaders.
    Undersamples 'unknown' and oversamples 'silence' in training set.
    """
    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # --- Balance Training Data ---
    # 1. Calculate median count of core commands
    cmd_counts = df_train[df_train["label"].isin(Config.COMMANDS)][
        "label"
    ].value_counts()
    target_count = int(cmd_counts.median())

    # 2. Separate groups
    df_commands = df_train[df_train["label"].isin(Config.COMMANDS)]
    df_unknown = df_train[df_train["label"] == "unknown"]
    df_silence = df_train[df_train["label"] == "silence"]

    # 3. Undersample 'unknown'
    if len(df_unknown) > target_count:
        df_unknown = df_unknown.sample(n=target_count, random_state=Config.SEED)

    # 4. Oversample 'silence'
    # We replicate the silence rows so the Dataset class picks different random crops
    if len(df_silence) > 0:
        df_silence = df_silence.sample(
            n=target_count, replace=True, random_state=Config.SEED
        )

    # 5. Combine
    df_train_balanced = pd.concat([df_commands, df_unknown, df_silence], axis=0)
    df_train_balanced = df_train_balanced.sample(
        frac=1, random_state=Config.SEED
    ).reset_index(drop=True)

    # --- Create Datasets ---
    train_dataset = SpeechCommandDataset(df_train_balanced, mode="train")
    val_dataset = SpeechCommandDataset(df_val, mode="val")

    # --- Create DataLoaders ---
    # Train loader uses MixUpCollate
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=MixUpCollate(
            alpha=Config.MIXUP_ALPHA, num_classes=Config.NUM_CLASSES
        ),
        pin_memory=True,
    )

    # Val loader uses default collate (stacking)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates the test DataLoader.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    test_dataset = SpeechCommandDataset(df_test, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
