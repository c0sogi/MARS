import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


def get_mel_filters(sr, n_fft, n_mels, f_min, f_max):
    """
    Creates a Mel filterbank matrix using NumPy.
    """
    if f_max is None:
        f_max = sr / 2
    if f_min is None:
        f_min = 0

    # Initialize Mel points
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    # Calculate bins
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

    # Create filterbank
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for f in range(bins[i], bins[i + 1]):
            fb[i, f] = (f - bins[i]) / (bins[i + 1] - bins[i])
        for f in range(bins[i + 1], bins[i + 2]):
            fb[i, f] = (bins[i + 2] - f) / (bins[i + 2] - bins[i + 1])

    return torch.tensor(fb, dtype=torch.float32)


class LogMelSpectrogram(torch.nn.Module):
    """
    Custom implementation of Log Mel Spectrogram using native PyTorch operations.
    Replaces torchaudio.transforms.MelSpectrogram and AmplitudeToDB.
    """

    def __init__(self, sr, n_fft, hop_length, n_mels, f_min, f_max, top_db=80.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db

        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer(
            "mel_basis", get_mel_filters(sr, n_fft, n_mels, f_min, f_max)
        )

    def forward(self, x):
        # x shape: (Channels, Time)

        # STFT
        # Note: torch.stft expects inputs (Batch, Time) or (Time).
        # We process (C, T) by treating C as batch or iterating.
        # Since C=1 usually, we can just pass x.
        spec_complex = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        # spec_complex shape: (C, Freq, Time)

        # Power Spectrogram
        power_spec = spec_complex.abs().pow(2.0)

        # Apply Mel Basis
        # mel_basis: (n_mels, Freq)
        # power_spec: (C, Freq, Time)
        # Result: (C, n_mels, Time)
        mel_spec = torch.matmul(self.mel_basis, power_spec)

        # Amplitude to DB
        mel_spec = torch.clamp(mel_spec, min=1e-10)
        log_mel_spec = 10.0 * torch.log10(mel_spec)

        # Top DB clamping
        max_val = log_mel_spec.max()
        log_mel_spec = torch.max(log_mel_spec, max_val - self.top_db)

        return log_mel_spec


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
    # Cite solution_lesson_node_00011: Use random.randint for inclusive bounds to maximize regularization
    f_width = random.randint(0, f_param)
    if f_width > 0 and f_width < n_mels:
        f_start = random.randint(0, n_mels - f_width)
        spec[:, f_start : f_start + f_width, :] = min_val

    # Time Masking
    t_param = Config.TIME_MASK_PARAM
    # Cite solution_lesson_node_00011: Use random.randint for inclusive bounds
    t_width = random.randint(0, t_param)
    if t_width > 0 and t_width < n_steps:
        t_start = random.randint(0, n_steps - t_width)
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
        self.log_mel_transform = LogMelSpectrogram(
            sr=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )

        # Cache background noise files for training
        self.noise_cache = {}
        if self.mode == "train":
            silence_rows = self.df[self.df["label"] == "silence"]
            unique_noise_paths = silence_rows["filepath"].unique()
            for path in unique_noise_paths:
                full_path = os.path.join(Config.INPUT_ROOT, path)
                if os.path.exists(full_path):
                    wav = self._read_and_resample(full_path)
                    self.noise_cache[path] = wav

    def _read_and_resample(self, filepath):
        """
        Reads audio using soundfile and resamples using torch.
        Returns tensor of shape (Channels, Time).
        """
        try:
            wav_np, sr = sf.read(filepath)
        except Exception:
            return torch.zeros(1, self.target_len)

        # Handle shape: sf.read returns (Samples, Channels) or (Samples,)
        # We need (Channels, Samples)
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]  # (1, S)
        else:
            wav_np = wav_np.T  # (C, S)

        wav_tensor = torch.from_numpy(wav_np).float()

        # Resample if needed
        if sr != self.target_sr:
            # interpolate expects (Batch, Channels, Time)
            wav_tensor = wav_tensor.unsqueeze(0)
            wav_tensor = F.interpolate(
                wav_tensor,
                scale_factor=self.target_sr / sr,
                mode="linear",
                align_corners=False,
            )
            wav_tensor = wav_tensor.squeeze(0)

        return wav_tensor

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
                    waveform = self._read_and_resample(full_path)

                    # For validation, take a deterministic crop (e.g., center)
                    if waveform.shape[1] > self.target_len:
                        start = (waveform.shape[1] - self.target_len) // 2
                        waveform = waveform[:, start : start + self.target_len]

        # 2. Handle Regular Audio (or if silence loading failed/fell through)
        if waveform is None:
            if not os.path.exists(full_path):
                return torch.zeros(1, self.target_len)

            waveform = self._read_and_resample(full_path)

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
            waveform = F.pad(waveform, (pad_left, pad_right))

        return waveform

    def __getitem__(self, idx):
        waveform = self._load_audio(idx)

        # Compute Log-Mel Spectrogram
        log_mel_spec = self.log_mel_transform(waveform)

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
