import os
import torch
import soundfile as sf
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


def create_mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    """
    Creates a Mel filterbank matrix using NumPy.
    """
    # FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # Mel scale conversion
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # Mel points
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    # Map to FFT bins
    bin_pts = np.floor((n_fft + 1) * f_pts / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        start = bin_pts[i]
        center = bin_pts[i + 1]
        end = bin_pts[i + 2]

        if center > start:
            filters[i, start:center] = (np.arange(start, center) - start) / (
                center - start
            )
        if end > center:
            filters[i, center:end] = (end - np.arange(center, end)) / (end - center)

    return filters


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

        # Precompute Mel Filterbanks
        self.mel_bases = []
        for _ in Config.WINDOW_SIZES:
            # n_fft is fixed to 2048 to cover largest window
            mel_basis = create_mel_filterbank(
                Config.SAMPLE_RATE, 2048, Config.N_MELS, Config.F_MIN, Config.F_MAX
            )
            self.mel_bases.append(torch.from_numpy(mel_basis).float())

        # Cache background noise files for 'silence' class to avoid repeated IO
        self.silence_cache = {}
        if "label" in self.df.columns:
            silence_files = self.df[self.df["label"] == "silence"]["filepath"].unique()
            for rel_path in silence_files:
                full_path = os.path.join(Config.INPUT_DIR, rel_path)
                if os.path.exists(full_path):
                    try:
                        # Use soundfile instead of torchaudio
                        wav_np, sr = sf.read(full_path)
                        wav = torch.from_numpy(wav_np).float()
                        if wav.dim() == 1:
                            wav = wav.unsqueeze(0)  # (1, T)
                        elif wav.dim() == 2:
                            wav = wav.t()  # (C, T)

                        # Resample if necessary
                        if sr != Config.SAMPLE_RATE:
                            wav = torch.nn.functional.interpolate(
                                wav.unsqueeze(0),
                                size=int(wav.shape[1] * Config.SAMPLE_RATE / sr),
                                mode="linear",
                                align_corners=False,
                            ).squeeze(0)

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
                # Deterministic crop for val/test
                if wav_len > self.target_length:
                    wav = full_wav[:, : self.target_length]
                else:
                    wav = full_wav
        else:
            # Standard Audio Loading
            full_path = os.path.join(Config.INPUT_DIR, filepath)
            if not os.path.exists(full_path):
                return torch.zeros(1, self.target_length)

            # Use soundfile
            wav_np, sr = sf.read(full_path)
            wav = torch.from_numpy(wav_np).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 2:
                wav = wav.t()

            if sr != Config.SAMPLE_RATE:
                wav = torch.nn.functional.interpolate(
                    wav.unsqueeze(0),
                    size=int(wav.shape[1] * Config.SAMPLE_RATE / sr),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

        # Pad or Crop to fixed length
        c, n = wav.shape
        if n < self.target_length:
            padding = self.target_length - n
            wav = torch.nn.functional.pad(wav, (0, padding))
        elif n > self.target_length:
            start = (n - self.target_length) // 2
            wav = wav[:, start : start + self.target_length]

        return wav

    def __getitem__(self, idx):
        # 1. Load Audio
        waveform = self._get_audio(idx)

        # 2. Compute Multi-Resolution Spectrograms
        specs = []
        for i, win_size in enumerate(Config.WINDOW_SIZES):
            # Create window
            window = torch.hann_window(win_size)
            # STFT
            stft = torch.stft(
                waveform,
                n_fft=2048,
                hop_length=Config.HOP_LENGTH,
                win_length=win_size,
                window=window,
                center=True,
                return_complex=True,
            )
            # Power Spectrogram: (1, F, T)
            power_spec = stft.abs().pow(2.0)

            # Apply Mel Basis: (n_mels, F) @ (1, F, T) -> (1, n_mels, T)
            # We use broadcasting or explicit matmul.
            # mel_basis is (n_mels, F), power_spec is (1, F, T)
            # We want (1, n_mels, T).
            mel_spec = torch.matmul(self.mel_bases[i], power_spec)
            specs.append(mel_spec)

        # Stack: (3, n_mels, time)
        multi_res_spec = torch.cat(specs, dim=0)

        # Convert to Log-Mel (dB)
        # 10 * log10(x)
        multi_res_spec = 10.0 * torch.log10(torch.clamp(multi_res_spec, min=1e-10))
        top_db = 80.0
        max_val = multi_res_spec.max()
        multi_res_spec = torch.clamp(multi_res_spec, min=max_val - top_db)

        # 3. Augmentation (Train only)
        if self.mode == "train":
            # Frequency Masking
            F = Config.FREQ_MASK_PARAM
            if F > 0:
                f_len = multi_res_spec.shape[1]
                f = np.random.randint(1, F + 1)
                f0 = np.random.randint(0, f_len - f + 1)
                multi_res_spec[:, f0 : f0 + f, :] = multi_res_spec.min()

            # Time Masking
            T = Config.TIME_MASK_PARAM
            if T > 0:
                t_len = multi_res_spec.shape[2]
                t = np.random.randint(1, T + 1)
                t0 = np.random.randint(0, t_len - t + 1)
                multi_res_spec[:, :, t0 : t0 + t] = multi_res_spec.min()

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
