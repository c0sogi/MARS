import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import AUDIO_CONFIG, TRAIN_CONFIG, LABEL_CONFIG, PATH_CONFIG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpeechDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path: Path to the metadata CSV.
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cached processed metadata.
        """
        self.mode = mode
        self.sample_rate = AUDIO_CONFIG.sample_rate
        self.target_length = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.duration)

        # Caching Logic for Metadata
        cache_name = f"processed_{mode}_metadata.parquet"
        cache_path = os.path.join(PATH_CONFIG.cache_dir, cache_name)

        if load_cached_data and os.path.exists(cache_path):
            self.df = pd.read_parquet(cache_path)
        else:
            self.df = pd.read_csv(metadata_path)
            self.df = self._process_metadata(self.df)
            # Save to cache
            os.makedirs(PATH_CONFIG.cache_dir, exist_ok=True)
            self.df.to_parquet(cache_path, index=False)

        self.label2id = LABEL_CONFIG.label2id
        self.id2label = LABEL_CONFIG.id2label

        # Pre-filter valid classes for training/val
        if self.mode != "test":
            # Ensure we only keep rows where the label is in our taxonomy
            # (Though _process_metadata should have handled this, this is a safety check)
            valid_mask = self.df["fine_grained_label"].isin(LABEL_CONFIG.all_classes)
            self.df = self.df[valid_mask].reset_index(drop=True)

    def _process_metadata(self, df):
        """
        Parses filepath to extract fine-grained labels and handles mapping.
        """
        fine_grained_labels = []

        for idx, row in df.iterrows():
            filepath = row["filepath"]
            # Extract parent folder name
            # filepath format: train/audio/<label>/<file>
            parts = filepath.split(os.sep)

            # Robust way to find the label folder: it's the directory containing the file
            # usually parts[-2]
            folder_name = parts[-2]

            # Map folder name to class label
            if folder_name == "_background_noise_":
                label = LABEL_CONFIG.silence_label
            elif folder_name in LABEL_CONFIG.all_classes:
                label = folder_name
            else:
                # Fallback for anything not in our taxonomy (shouldn't happen with provided aux list)
                # If it's not in all_classes, we can't train on it as a specific class.
                # However, for 'test', we don't care.
                label = "unknown_out_of_vocab"

            fine_grained_labels.append(label)

        df["fine_grained_label"] = fine_grained_labels
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(PATH_CONFIG.input_root, row["filepath"])
        label_str = row["fine_grained_label"]

        # Determine Label ID
        if self.mode == "test":
            label_id = -1  # Dummy
        else:
            label_id = self.label2id[label_str]

        # Load Audio
        # Special handling for Silence (Long files)
        if label_str == LABEL_CONFIG.silence_label and self.mode == "train":
            waveform = self._load_silence_crop(filepath)
        else:
            waveform = self._load_standard_clip(filepath)

        return waveform, label_id

    def _load_silence_crop(self, filepath):
        """
        Loads a random 1-second crop from a long background noise file.
        Efficiently loads only the necessary frames.
        """
        try:
            info = torchaudio.info(filepath)
            total_frames = info.num_frames

            if total_frames <= self.target_length:
                # If file is short, load all and pad later
                waveform, sr = torchaudio.load(filepath)
            else:
                # Random crop
                max_start = total_frames - self.target_length
                start_frame = random.randint(0, max_start)
                waveform, sr = torchaudio.load(
                    filepath, frame_offset=start_frame, num_frames=self.target_length
                )

            return self._process_waveform(waveform, sr)
        except Exception as e:
            # Fallback to zeros if file load fails
            return torch.zeros(1, self.target_length)

    def _load_standard_clip(self, filepath):
        """
        Loads a standard audio clip.
        """
        try:
            waveform, sr = torchaudio.load(filepath)
            return self._process_waveform(waveform, sr)
        except Exception:
            return torch.zeros(1, self.target_length)

    def _process_waveform(self, waveform, sr):
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure correct length (Pad or Crop)
        current_len = waveform.shape[1]

        if current_len < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            # Crop
            if self.mode == "train":
                # Random Crop
                start = random.randint(0, current_len - self.target_length)
            else:
                # Center Crop
                start = (current_len - self.target_length) // 2

            waveform = waveform[:, start : start + self.target_length]

        return waveform


def get_balanced_sampler(dataset):
    """
    Implements Variance-Aware Balancing.
    Upsamples Targets and Silence to ~2000.
    Keeps Aux classes at natural frequency.
    """
    df = dataset.df
    labels = df["fine_grained_label"].values

    # Calculate current counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique_labels, counts))

    # Define Target Counts
    TARGET_COUNT = 2000

    weights = []
    for label in labels:
        current_count = count_dict[label]

        if label in LABEL_CONFIG.target_labels or label == LABEL_CONFIG.silence_label:
            # Upsample logic: Weight = Target / Current
            # We want the probability of picking this class to result in TARGET_COUNT samples per epoch
            # Standard weight formula for BalancedSampler is 1/count.
            # Here we want specific ratios.
            # Weight * Count ~ Desired_Count
            # Weight = Desired_Count / Current_Count
            weight = TARGET_COUNT / current_count
        else:
            # Keep natural frequency (relative to other aux classes)
            # If we used 1.0 here, the aux classes would dominate because their total count is high.
            # We want Aux classes to appear as they are.
            # If we set weight = 1.0, then effective count is current_count.
            weight = 1.0

        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.double)

    # Create Sampler
    # num_samples: We can set this to the sum of desired counts or len(dataset).
    # If we want exactly 2000 per target (11 classes) + ~9000 aux, total ~31000.
    # Dataset size is ~46k.
    # Let's set num_samples to len(dataset) to keep epoch length standard,
    # the weights will handle the distribution.
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    return sampler


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    set_seed(TRAIN_CONFIG.seed)

    # 1. Datasets
    train_dataset = SpeechDataset(
        PATH_CONFIG.train_metadata, mode="train", load_cached_data=load_cached_data
    )

    val_dataset = SpeechDataset(
        PATH_CONFIG.val_metadata, mode="val", load_cached_data=load_cached_data
    )

    test_dataset = SpeechDataset(
        PATH_CONFIG.test_metadata, mode="test", load_cached_data=load_cached_data
    )

    # Debugging: Subset
    if debug:
        indices = list(range(100))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        # Hack to expose df for sampler on subset
        train_dataset.df = train_dataset.dataset.df.iloc[indices].reset_index(drop=True)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    # 2. Sampler (Train only)
    train_sampler = get_balanced_sampler(train_dataset)

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        sampler=train_sampler,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=True,
    )

    # Print Stats
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset:   {len(val_dataset)} samples")
    print(f"Test Dataset:  {len(test_dataset)} samples")
    print(f"Classes:       {LABEL_CONFIG.num_classes}")

    return train_loader, val_loader, test_loader
