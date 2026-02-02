import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from library.config import Config


def get_classes():
    """
    Retrieves the list of classes from the sample_submission.csv file
    to ensure the correct order for predictions.
    """
    ss_path = os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
    df = pd.read_csv(ss_path)
    # The columns are fname, Label1, Label2, ...
    classes = [c for c in df.columns if c not in ["fname", "file_path"]]
    return classes


class MelSpectrogram(nn.Module):
    """
    Custom MelSpectrogram implementation using pure PyTorch to avoid torchaudio dependency.
    """

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min, f_max):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Generate Mel Filterbank
        mel_basis = self._create_mel_basis(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Pre-compute window
        self.register_buffer("window", torch.hann_window(n_fft))

    def _create_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        # Initialize frequencies
        # Center frequencies of each FFT bin
        n_freqs = n_fft // 2 + 1

        # Mel scale conversion functions (HTK formula)
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        # Create mel points
        m_min = hz_to_mel(fmin)
        m_max = hz_to_mel(fmax)
        m_pts = np.linspace(m_min, m_max, n_mels + 2)
        f_pts = mel_to_hz(m_pts)

        # Map to FFT bins
        bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

        # Create Filterbank matrix: (n_mels, n_freqs)
        fb = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            b_left = bins[i]
            b_center = bins[i + 1]
            b_right = bins[i + 2]

            # Left ramp
            for k in range(b_left, b_center):
                if k < n_freqs:
                    fb[i, k] = (k - b_left) / (b_center - b_left)

            # Right ramp
            for k in range(b_center, b_right):
                if k < n_freqs:
                    fb[i, k] = (b_right - k) / (b_right - b_center)

        return fb

    def forward(self, waveform):
        # waveform: (Batch/Channels, Time)

        # Compute STFT
        # return_complex=True is required in newer torch versions
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        # Compute Power Spectrogram: |STFT|^2
        spec = stft.abs().pow(2)  # (C, F, T)

        # Apply Mel Filterbank
        # mel_basis: (n_mels, F)
        # spec: (C, F, T)
        # Result: (C, n_mels, T)
        melspec = torch.matmul(self.mel_basis, spec)

        return melspec


class AmplitudeToDB(nn.Module):
    """
    Custom AmplitudeToDB implementation.
    """

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        # x is power spectrogram
        # Avoid log(0)
        x = torch.clamp(x, min=1e-10)
        x_db = 10.0 * torch.log10(x)

        if self.top_db is not None:
            x_db = torch.max(x_db, x_db.max() - self.top_db)

        return x_db


class AudioDataset(Dataset):
    def __init__(self, df, classes, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (fname, file_path, labels/encoded_labels).
            classes (list): List of class names.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.mode = mode

        # Pre-compute label lookup if labels exist
        if "encoded_labels" in self.df.columns:
            self.labels = self.df["encoded_labels"].tolist()
        else:
            self.labels = None

        # Audio transformation components
        self.mel_spec = MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=80.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        # 1. Load Audio using soundfile instead of torchaudio
        try:
            # sf.read returns (samples, channels) for multi-channel
            # or (samples,) for mono.
            # It returns float64 by default, usually normalized [-1, 1].
            waveform, sr = sf.read(full_path)
            waveform = torch.from_numpy(waveform).float()

            # Ensure shape is (Channels, Time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, Time)
            else:
                waveform = waveform.t()  # (Channels, Time)

        except Exception as e:
            # Fallback for corrupted files
            waveform = torch.zeros(1, Config.SAMPLE_RATE)
            sr = Config.SAMPLE_RATE

        # 2. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample
        if sr != Config.SAMPLE_RATE:
            # Use torch interpolation for resampling
            # Input to interpolate must be (Batch, Channels, Time)
            waveform = waveform.unsqueeze(0)
            new_len = int(waveform.shape[-1] * Config.SAMPLE_RATE / sr)
            waveform = F.interpolate(
                waveform, size=new_len, mode="linear", align_corners=False
            )
            waveform = waveform.squeeze(0)

        # 4. Length Adjustment
        current_len = waveform.shape[1]

        if self.mode == "train":
            # Fixed length for training (Random Crop or Pad)
            target_len = Config.AUDIO_LEN
            if current_len > target_len:
                start = np.random.randint(0, current_len - target_len)
                waveform = waveform[:, start : start + target_len]
            elif current_len < target_len:
                pad_amt = target_len - current_len
                waveform = F.pad(waveform, (0, pad_amt))
        else:
            # Variable length for Val/Test
            # Ensure minimum length for FFT
            if current_len < Config.N_FFT:
                pad_amt = Config.N_FFT - current_len
                waveform = F.pad(waveform, (0, pad_amt))

        # 5. Compute Log-Mel Spectrogram
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        # 6. Instance-wise Normalization
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # Apply SpecAugment (Time/Freq Masking)
        # Cite solution_lesson_node_00011
        if self.mode == "train":
            C, freq_dim, T = spec.shape

            # Frequency Masking
            if Config.SPEC_AUG_FREQ_MASK > 0:
                f_width = np.random.randint(0, Config.SPEC_AUG_FREQ_MASK)
                if f_width > 0 and (freq_dim - f_width) > 0:
                    f0 = np.random.randint(0, freq_dim - f_width)
                    spec[:, f0 : f0 + f_width, :] = 0.0

            # Time Masking
            if Config.SPEC_AUG_TIME_MASK > 0:
                t_width = np.random.randint(0, Config.SPEC_AUG_TIME_MASK)
                if t_width > 0 and (T - t_width) > 0:
                    t0 = np.random.randint(0, T - t_width)
                    spec[:, :, t0 : t0 + t_width] = 0.0

        # 7. Get Label
        if self.labels is not None:
            label_vec = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Dummy label for test
            label_vec = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        return spec, label_vec, fname


def collate_fn(batch):
    """
    Custom collate function to handle variable length spectrograms.
    Pads the time dimension (dim 2) to the maximum length in the batch.
    """
    # batch is a list of tuples: (spec, label, fname)
    # spec shape: (1, n_mels, time)

    # Find maximum time dimension in this batch
    max_time = max([x[0].shape[2] for x in batch])

    specs = []
    labels = []
    fnames = []

    for spec, label, fname in batch:
        current_time = spec.shape[2]
        pad_amt = max_time - current_time
        if pad_amt > 0:
            # Pad the last dimension (time)
            spec = F.pad(spec, (0, pad_amt))
        specs.append(spec)
        labels.append(label)
        fnames.append(fname)

    return torch.stack(specs), torch.stack(labels), fnames


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.
    Implements caching for processed metadata (dataframes with encoded labels).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")

    classes = get_classes()

    # --- Load or Create DataFrames ---
    if load_cached_data and os.path.exists(train_cache) and os.path.exists(val_cache):
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
    else:
        # Load raw metadata
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)

        # Initialize MultiLabelBinarizer with the specific vocabulary
        mlb = MultiLabelBinarizer(classes=classes)
        # Fit on classes to ensure all columns exist and are ordered correctly
        mlb.fit([classes])

        # Process Train Labels
        train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(","))
        train_encoded = mlb.transform(train_df["label_list"])
        train_df["encoded_labels"] = list(train_encoded)

        # Process Val Labels
        val_df["label_list"] = val_df["labels"].apply(lambda x: x.split(","))
        val_encoded = mlb.transform(val_df["label_list"])
        val_df["encoded_labels"] = list(val_encoded)

        # Save to cache
        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)

    # --- Load Test Data Dynamically ---
    # Always load from input/sample_submission.csv to handle runtime test sets
    ss_path = os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
    test_df = pd.read_csv(ss_path)
    test_df["file_path"] = test_df["fname"].apply(lambda x: os.path.join("test", x))

    # --- Debug / Subsampling ---
    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        # test_df = test_df.head(50) # Optional

    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
    if Config.MAX_VAL_SAMPLES:
        val_df = val_df.iloc[: Config.MAX_VAL_SAMPLES]

    # --- Create Datasets ---
    train_dataset = AudioDataset(train_df, classes, mode="train")
    val_dataset = AudioDataset(val_df, classes, mode="val")
    test_dataset = AudioDataset(test_df, classes, mode="test")

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
