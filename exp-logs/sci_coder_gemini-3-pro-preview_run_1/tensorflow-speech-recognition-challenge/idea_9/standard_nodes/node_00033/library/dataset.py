import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config, set_seed
from library.audio_transforms import DualChannelProcessor


class SpeechCommandDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands.
    Handles metadata loading, variance-aware balancing, and fine-grained label extraction.
    """

    def __init__(self, csv_path, mode="train", transform=None, load_cached_data=True):
        self.mode = mode
        self.transform = transform
        self.label2id = Config.get_label2id()

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        if mode == "train":
            self.df = self._load_and_balance_data(csv_path, load_cached_data)
        else:
            self.df = pd.read_csv(csv_path)
            # For validation, we need fine-grained labels for accurate metrics
            if mode == "val":
                self.df["fine_label"] = self.df.apply(self._extract_fine_label, axis=1)
            else:
                # Test set has no labels
                self.df["fine_label"] = Config.UNKNOWN_LABEL

    def _extract_fine_label(self, row):
        """
        Extracts the fine-grained label from the filepath.
        e.g., 'train/audio/bed/001.wav' -> 'bed'
        """
        # Normalize path separators
        path = row["filepath"].replace("\\", "/")
        parts = path.split("/")

        # Structure is usually input/train/audio/<label>/<filename>
        # Relative path in CSV is train/audio/<label>/<filename>
        if len(parts) >= 2:
            folder = parts[-2]
            if folder == "_background_noise_":
                return Config.SILENCE_LABEL
            return folder
        return Config.UNKNOWN_LABEL

    def _load_and_balance_data(self, csv_path, load_cached_data):
        """
        Loads training data and applies Variance-Aware Balancing.
        Upsamples Targets and Silence to ~2000. Keeps Aux classes natural.
        Caches the result to Parquet.
        """
        cache_path = os.path.join(Config.CACHE_DIR, "train_balanced.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # Fallback to re-process

        # 2. Process from Scratch
        df = pd.read_csv(csv_path)
        df["fine_label"] = df.apply(self._extract_fine_label, axis=1)

        # Filter to ensure we only have known classes
        valid_labels = set(Config.ALL_LABELS)
        df = df[df["fine_label"].isin(valid_labels)]

        balanced_dfs = []
        TARGET_COUNT = 2000

        groups = df.groupby("fine_label")

        for label, group in groups:
            count = len(group)

            # Upsample Targets and Silence
            if label in Config.TARGET_LABELS or label == Config.SILENCE_LABEL:
                n_samples = max(count, TARGET_COUNT)
                # Sample with replacement to reach target count
                resampled = group.sample(
                    n=n_samples, replace=True, random_state=Config.SEED
                )
                balanced_dfs.append(resampled)
            else:
                # Keep Auxiliary classes at natural distribution (Variance Preservation)
                balanced_dfs.append(group)

        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)

        # 3. Save Cache
        balanced_df.to_parquet(cache_path)

        return balanced_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        label_str = row["fine_label"]

        # Load Audio
        try:
            waveform, sr = torchaudio.load(filepath)

            # Resample if needed (robustness)
            if sr != Config.SR:
                resampler = torchaudio.transforms.Resample(sr, Config.SR)
                waveform = resampler(waveform)

            # Ensure Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

        except Exception:
            # Return silent tensor if file is corrupt
            waveform = torch.zeros(1, Config.AUDIO_LEN_SAMPLES)

        # Apply Dual-Channel Processing
        if self.transform:
            # label_str is passed to handle silence synthesis logic
            spec = self.transform(waveform, mode=self.mode, label=label_str)
        else:
            spec = torch.zeros(2, Config.N_MELS, 101)

        # Map Label to ID
        label_id = self.label2id.get(label_str, 0)

        return spec, label_id


class MixupCollator:
    """
    Collator that applies Mixup regularization to the batch.
    """

    def __init__(self, num_classes, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, batch):
        # Stack inputs: (B, 2, F, T)
        inputs = torch.stack([item[0] for item in batch])
        # Stack targets: (B,)
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)

        batch_size = inputs.size(0)

        # Convert targets to One-Hot: (B, NumClasses)
        targets_one_hot = torch.zeros(batch_size, self.num_classes)
        targets_one_hot.scatter_(1, targets.view(-1, 1), 1)

        # Sample lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Shuffle indices
        index = torch.randperm(batch_size)

        # Mix
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index]

        return mixed_inputs, mixed_targets


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)

    # Initialize Processor (handles noise bank caching internally)
    processor = DualChannelProcessor(load_cached_data=load_cached_data)

    # Create Datasets
    train_ds = SpeechCommandDataset(
        Config.TRAIN_CSV,
        mode="train",
        transform=processor,
        load_cached_data=load_cached_data,
    )

    val_ds = SpeechCommandDataset(
        Config.VAL_CSV,
        mode="val",
        transform=processor,
        load_cached_data=load_cached_data,
    )

    test_ds = SpeechCommandDataset(
        Config.TEST_CSV,
        mode="test",
        transform=processor,
        load_cached_data=load_cached_data,
    )

    # Apply Debugging Subset if requested
    if debug_subset_size is not None:
        train_ds.df = train_ds.df.iloc[:debug_subset_size]
        val_ds.df = val_ds.df.iloc[:debug_subset_size]
        test_ds.df = test_ds.df.iloc[:debug_subset_size]

    # Create Loaders
    # Train: Shuffle + Mixup
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=MixupCollator(
            num_classes=Config.NUM_CLASSES, alpha=Config.MIXUP_ALPHA
        ),
        pin_memory=True,
        drop_last=True,
    )

    # Val: No Shuffle + Standard Collate
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test: No Shuffle + Standard Collate
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
