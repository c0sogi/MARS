import os
import hashlib
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import AudioConfig, TrainConfig

# Ensure reproducibility
torch.manual_seed(TrainConfig.seed)
np.random.seed(TrainConfig.seed)


def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """
    Creates a Mel filterbank matrix using native NumPy.
    Cite Debug Lesson 1: Native implementation to avoid library dependency.
    Returns: (n_freqs, n_mels)
    """
    n_freqs = n_fft // 2 + 1

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Grid points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax if fmax else sr / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin mapping
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    weights = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        start = bin_points[i]
        center = bin_points[i + 1]
        end = bin_points[i + 2]

        if center > start:
            weights[i, start:center] = (np.arange(start, center) - start) / (
                center - start
            )
        if end > center:
            weights[i, center:end] = (end - np.arange(center, end)) / (end - center)

    return torch.from_numpy(weights.T).float()


class MelSpectrogram(torch.nn.Module):
    """
    Custom MelSpectrogram implementation using torch.stft.
    Replaces torchaudio.transforms.MelSpectrogram.
    """

    def __init__(
        self, sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Create Mel Basis
        self.mel_basis = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)

        # Create Window
        self.window = torch.hann_window(win_length)

    def forward(self, waveform):
        # waveform: (1, time) or (time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # STFT
        # Output: (Batch, Freq, Frames) complex tensor
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

        # Power Spectrogram
        spec = stft.abs().pow(2.0)

        # Mel Spectrogram
        # spec: (Batch, Freq, Frames)
        # mel_basis: (Freq, Mels)
        # Output: (Batch, Mels, Frames)
        mel_spec = torch.matmul(self.mel_basis.to(waveform.device).T, spec)

        return mel_spec


class AmplitudeToDB(torch.nn.Module):
    """
    Custom AmplitudeToDB implementation.
    Replaces torchaudio.transforms.AmplitudeToDB.
    """

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        # 10 * log10(x)
        x_db = 10.0 * torch.log10(torch.clamp(x, min=1e-10))
        x_max = x_db.max()
        x_db = torch.clamp(x_db, min=x_max - self.top_db)
        return x_db


class FeatureExtractor:
    """
    Computes 3-Channel Multi-Resolution Log-Mel Spectrograms.
    Channels correspond to Short, Medium, and Long STFT windows.
    """

    def __init__(self):
        self.transforms = []
        # Initialize a MelSpectrogram transform for each resolution defined in config
        for n_fft, win_length in AudioConfig.resolutions:
            t = MelSpectrogram(
                sample_rate=AudioConfig.sr,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=AudioConfig.hop_length,
                n_mels=AudioConfig.n_mels,
                f_min=AudioConfig.fmin,
                f_max=AudioConfig.fmax,
            )
            self.transforms.append(t)

        self.db_transform = AmplitudeToDB(top_db=AudioConfig.top_db)

    def __call__(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Audio waveform (1, time) or (channels, time)
        Returns:
            np.ndarray: (3, n_mels, time)
        """
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Truncate to fixed length
        target_samples = AudioConfig.n_samples
        num_samples = waveform.shape[1]

        if num_samples < target_samples:
            # Pad with zeros at the end
            padding = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > target_samples:
            # Truncate
            waveform = waveform[:, :target_samples]

        # Compute features for each resolution
        channels = []
        for transform in self.transforms:
            # Compute MelSpec: Output (1, n_mels, time)
            spec = transform(waveform)
            # Convert to DB
            spec_db = self.db_transform(spec)
            channels.append(spec_db)

        # Stack along channel dimension: (3, n_mels, time)
        multires_spec = torch.cat(channels, dim=0)

        return multires_spec.numpy()


class CachingDataset(Dataset):
    """
    Dataset wrapper to process and cache audio files.
    Used with DataLoader to parallelize preprocessing.
    """

    def __init__(self, metadata_df, input_dir, cache_dir, overwrite=False):
        self.metadata_df = metadata_df
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.overwrite = overwrite
        self.extractor = FeatureExtractor()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        rel_path = row["filepath"]

        # Generate deterministic cache filename
        file_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.npy")

        # Check if processing is needed
        if not self.overwrite and os.path.exists(cache_path):
            return 0

        # Load and Process
        full_path = os.path.join(self.input_dir, rel_path)
        try:
            # Load audio using soundfile (replaces torchaudio.load)
            # sf.read returns (samples, channels) or (samples,)
            data, sr = sf.read(full_path)

            # Convert to tensor (Channels, Time)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T).float()

            # Resample if necessary (Native PyTorch implementation)
            if sr != AudioConfig.sr:
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0),
                    scale_factor=AudioConfig.sr / sr,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            # Compute 3-channel features
            features = self.extractor(waveform)

            # Save to disk
            np.save(cache_path, features)

        except Exception as e:
            # In case of corrupt files, print error but don't crash the whole process
            print(f"Error processing {rel_path}: {e}")

        return 0


def process_dataset(metadata_path, cache_dir, load_cached_data=True):
    """
    Processes a dataset defined by a metadata CSV file.
    Uses DataLoader to utilize multiple CPU cores.
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Determine overwrite behavior
    overwrite = not load_cached_data

    print(f"Processing {len(df)} files from {metadata_path}...")
    print(f"Cache Directory: {cache_dir}")
    print(f"Overwrite Mode: {overwrite}")

    # Create Dataset and DataLoader
    dataset = CachingDataset(
        metadata_df=df,
        input_dir=TrainConfig.input_dir,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )

    # Use num_workers for parallel processing
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        collate_fn=lambda x: x,  # Dummy collate
    )

    # Iterate to trigger processing
    for _ in loader:
        pass

    print("Processing complete.")


def cache_data(load_cached_data=True):
    """
    Main function to cache Train, Validation, and Test datasets.
    """
    # Ensure directories exist
    TrainConfig.setup_directories()

    # Process Train
    process_dataset(
        TrainConfig.train_metadata_path, TrainConfig.cache_dir, load_cached_data
    )

    # Process Validation
    process_dataset(
        TrainConfig.val_metadata_path, TrainConfig.cache_dir, load_cached_data
    )

    # Process Test
    process_dataset(
        TrainConfig.test_metadata_path, TrainConfig.cache_dir, load_cached_data
    )


def get_feature_path(filepath):
    """
    Helper function to get the cached feature path for a given audio filepath.
    """
    file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
    return os.path.join(TrainConfig.cache_dir, f"{file_hash}.npy")
