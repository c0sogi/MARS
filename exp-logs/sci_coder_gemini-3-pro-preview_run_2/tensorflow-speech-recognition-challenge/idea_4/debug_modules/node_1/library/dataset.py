import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.audio_transforms import get_transforms


class SpeechCommandDataset(Dataset):
    def __init__(self, metadata_path, phase="train", load_cached=True, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to try loading cached waveforms.
            debug (bool): If True, use a small subset of data.
        """
        self.phase = phase
        self.df = pd.read_csv(metadata_path)

        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

        # Ensure is_background column exists (fill False for test/val if missing)
        if "is_background" not in self.df.columns:
            self.df["is_background"] = False
        else:
            self.df["is_background"] = self.df["is_background"].fillna(False)

        self.labels = self.df["label"].values
        self.fnames = self.df["fname"].values
        self.is_background = self.df["is_background"].values.astype(bool)

        # Prepare transforms
        self.transform = get_transforms(phase)

        # Separate indices for background (dynamic slicing) and regular (static cache)
        # We need to map the global index (0..len(df)) to the specific data source
        self.background_indices = np.where(self.is_background)[0]
        self.regular_indices = np.where(~self.is_background)[0]

        # Map global index to local index in the respective storage
        # map_to_local[global_idx] = local_index_in_storage
        self.map_to_local = np.zeros(len(self.df), dtype=int)

        # For regular files, we will store them in a contiguous array, so we map
        # global_idx -> 0..N_regular
        for local_i, global_i in enumerate(self.regular_indices):
            self.map_to_local[global_i] = local_i

        # For background files, we map global_idx -> global_idx (key in dict)
        for global_i in self.background_indices:
            self.map_to_local[global_i] = global_i

        # 1. Load Background Noise (Always load fresh into memory)
        self.background_data = {}
        for idx in self.background_indices:
            row = self.df.iloc[idx]
            path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                wav, sr = torchaudio.load(path)
                # Ensure 16k
                if sr != Config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
                    wav = resampler(wav)
                # Ensure mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                self.background_data[idx] = wav
            except Exception:
                # Fallback: 1 second of silence
                self.background_data[idx] = torch.zeros(1, Config.NUM_SAMPLES)

        # 2. Load Regular Files (Cacheable)
        self.regular_waveforms = self._load_regular_data(load_cached)

    def _load_regular_data(self, load_cached):
        """
        Loads regular audio files, padding/truncating to fixed length.
        Uses caching to speed up subsequent runs.
        """
        cache_name = f"{self.phase}_regular_waveforms.npy"
        if Config.DEBUG:
            cache_name = f"debug_{cache_name}"

        cache_path = os.path.join(Config.WORKING_DIR, cache_name)

        # Try loading from cache
        if load_cached and os.path.exists(cache_path):
            try:
                # allow_pickle=False ensures we are loading a standard numerical array
                data = np.load(cache_path, allow_pickle=False)
                if len(data) == len(self.regular_indices):
                    return torch.from_numpy(data)
            except Exception:
                pass  # Fallback to processing

        # Process from scratch
        waveforms = []
        target_len = Config.NUM_SAMPLES

        for idx in self.regular_indices:
            row = self.df.iloc[idx]
            path = os.path.join(Config.INPUT_DIR, row["file_path"])

            try:
                wav, sr = torchaudio.load(path)

                # Resample
                if sr != Config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
                    wav = resampler(wav)

                # Mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # Pad/Truncate
                if wav.shape[1] > target_len:
                    wav = wav[:, :target_len]
                elif wav.shape[1] < target_len:
                    pad_amt = target_len - wav.shape[1]
                    wav = torch.nn.functional.pad(wav, (0, pad_amt))

                waveforms.append(wav.numpy())
            except Exception:
                # Fallback: silence
                waveforms.append(np.zeros((1, target_len), dtype=np.float32))

        if len(waveforms) > 0:
            # Concatenate to (N, 16000) - squeeze channel dim for storage efficiency
            # wav.numpy() is (1, 16000), so we concatenate on axis 0 -> (N, 16000)
            waveforms_np = np.concatenate(waveforms, axis=0)
        else:
            waveforms_np = np.empty((0, target_len), dtype=np.float32)

        # Save to cache
        np.save(cache_path, waveforms_np, allow_pickle=False)

        return torch.from_numpy(waveforms_np)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Label
        label_str = self.labels[idx]
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])

        # 2. Get Audio
        if self.is_background[idx]:
            # Dynamic random slicing for background noise
            full_wav = self.background_data[idx]  # (1, Total_Time)
            total_samples = full_wav.shape[1]
            required = Config.NUM_SAMPLES

            if total_samples > required:
                start = torch.randint(0, total_samples - required + 1, (1,)).item()
                wav = full_wav[:, start : start + required]
            else:
                pad_amt = required - total_samples
                wav = torch.nn.functional.pad(full_wav, (0, pad_amt))
        else:
            # Retrieve from cached array
            local_idx = self.map_to_local[idx]
            wav = self.regular_waveforms[local_idx]  # (16000,)
            wav = wav.unsqueeze(0)  # (1, 16000)

        # 3. Apply Transforms (Spectrogram -> Norm -> Aug)
        features = self.transform(wav)

        return features, label_id


def get_dataloaders(debug=Config.DEBUG, load_cached=True):
    """
    Creates DataLoaders for train, val, and test sets.
    Applies WeightedRandomSampler to the training set to handle class imbalance.
    """
    # 1. Train Dataset
    train_ds = SpeechCommandDataset(
        Config.TRAIN_METADATA, phase="train", load_cached=load_cached, debug=debug
    )

    # Calculate Class Weights for Sampling
    # We want to balance the batch so that 'silence' (few files) is seen as often as 'unknown' (many files)
    label_counts = pd.Series(train_ds.labels).value_counts()

    sample_weights = []
    for label in train_ds.labels:
        count = label_counts.get(label, 0)
        if count > 0:
            sample_weights.append(1.0 / count)
        else:
            sample_weights.append(0.0)

    sample_weights = torch.DoubleTensor(sample_weights)

    # Sampler draws samples with replacement according to weights
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Validation Dataset
    val_ds = SpeechCommandDataset(
        Config.VAL_METADATA, phase="val", load_cached=load_cached, debug=debug
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test Dataset
    test_ds = SpeechCommandDataset(
        Config.TEST_METADATA, phase="test", load_cached=load_cached, debug=debug
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
