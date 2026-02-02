import os
import hashlib
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import AudioConfig, TrainConfig

# Ensure reproducibility
torch.manual_seed(TrainConfig.seed)
np.random.seed(TrainConfig.seed)


class FeatureExtractor:
    """
    Computes 3-Channel Multi-Resolution Log-Mel Spectrograms.
    Channels correspond to Short, Medium, and Long STFT windows.
    """

    def __init__(self):
        self.transforms = []
        # Initialize a MelSpectrogram transform for each resolution defined in config
        for n_fft, win_length in AudioConfig.resolutions:
            t = T.MelSpectrogram(
                sample_rate=AudioConfig.sr,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=AudioConfig.hop_length,
                n_mels=AudioConfig.n_mels,
                f_min=AudioConfig.fmin,
                f_max=AudioConfig.fmax,
                center=True,  # Ensures consistent time dimension across resolutions
                pad_mode="reflect",
                power=2.0,
                normalized=False,
            )
            self.transforms.append(t)

        self.db_transform = T.AmplitudeToDB(stype="power", top_db=AudioConfig.top_db)

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
            # Load audio (normalize=True scales to [-1, 1])
            waveform, sr = torchaudio.load(full_path, normalize=True)

            # Resample if necessary
            if sr != AudioConfig.sr:
                resampler = T.Resample(orig_freq=sr, new_freq=AudioConfig.sr)
                waveform = resampler(waveform)

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
