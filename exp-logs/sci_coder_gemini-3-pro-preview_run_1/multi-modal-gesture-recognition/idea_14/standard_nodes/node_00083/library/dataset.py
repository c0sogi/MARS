import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import BACKGROUND_LABEL, SKELETON_INPUT_SIZE, AUDIO_INPUT_SIZE, SEED
from library.data_utils import process_sample

# Set fixed seed for reproducibility in augmentations
np.random.seed(SEED)
torch.manual_seed(SEED)


def augment_channel_mask(features, mask_ratio=0.1):
    """
    Randomly masks out a percentage of channels.
    Args:
        features: (T, C) numpy array
        mask_ratio: float, fraction of channels to mask
    Returns:
        masked_features: (T, C) numpy array
    """
    if features is None or features.size == 0:
        return features

    T, C = features.shape
    num_mask = max(1, int(C * mask_ratio))

    # Choose random channels to mask
    mask_indices = np.random.choice(C, num_mask, replace=False)

    masked_features = features.copy()
    masked_features[:, mask_indices] = 0.0
    return masked_features


def augment_time_mask(features, min_len=5, max_len=15):
    """
    Randomly masks out a contiguous chunk of time.
    Args:
        features: (T, C) numpy array
        min_len: int, minimum frames to mask
        max_len: int, maximum frames to mask
    Returns:
        masked_features: (T, C) numpy array
    """
    if features is None or features.size == 0:
        return features

    T, C = features.shape
    if T <= min_len:
        return features

    # Determine mask length
    mask_len = np.random.randint(min_len, min(max_len, T) + 1)

    # Determine start index
    start_idx = np.random.randint(0, T - mask_len + 1)

    masked_features = features.copy()
    masked_features[start_idx : start_idx + mask_len, :] = 0.0
    return masked_features


class GestureDataset(Dataset):
    def __init__(self, dataframe, stats=None, is_train=True, augment=False):
        """
        Args:
            dataframe: pandas DataFrame containing metadata.
            stats: dict containing 'skel_mean', 'skel_std', 'audio_mean', 'audio_std'.
            is_train: bool, indicates if this is a training set.
            augment: bool, whether to apply data augmentation.
        """
        self.df = dataframe.reset_index(drop=True)
        self.stats = stats
        self.is_train = is_train
        self.augment = augment

        # Pre-convert stats to float32 for efficiency
        if self.stats:
            self.skel_mean = self.stats["skel_mean"].astype(np.float32)
            self.skel_std = self.stats["skel_std"].astype(np.float32)
            self.audio_mean = self.stats["audio_mean"].astype(np.float32)
            self.audio_std = self.stats["audio_std"].astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load data (uses caching internally)
        sample = process_sample(row, load_cached_data=True)

        # Handle corruption or missing data
        if sample is None or sample["skeleton"] is None:
            # Return a dummy dict that collate_fn can filter out
            return None

        skeleton = sample["skeleton"].astype(np.float32)  # (T, 60)
        audio = sample["audio"].astype(np.float32)  # (T, 20)
        labels = sample["labels"]  # (L,)

        # 1. Normalization
        if self.stats:
            # Z-score normalization: (X - mean) / std
            # Handle potential shape mismatches if stats were computed differently,
            # though config ensures fixed sizes.
            skeleton = (skeleton - self.skel_mean) / (self.skel_std + 1e-6)
            audio = (audio - self.audio_mean) / (self.audio_std + 1e-6)

        # 2. Augmentation (Training only)
        if self.is_train and self.augment:
            # Channel Masking
            if np.random.rand() < 0.5:
                skeleton = augment_channel_mask(skeleton, mask_ratio=0.1)
                audio = augment_channel_mask(audio, mask_ratio=0.1)

            # Time Masking
            if np.random.rand() < 0.5:
                # We apply the same time mask to both modalities to simulate missing data
                # or we can apply independently. Independent is more robust for multi-modal learning.
                # Let's apply independently as per "Input Injection" lesson.
                skeleton = augment_time_mask(skeleton, min_len=5, max_len=15)
                audio = augment_time_mask(audio, min_len=5, max_len=15)

        # Convert to Torch Tensors
        return {
            "skeleton": torch.from_numpy(skeleton),
            "audio": torch.from_numpy(audio),
            "labels": torch.from_numpy(labels).long(),
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences.
    Args:
        batch: List of dictionary samples.
    Returns:
        dict containing padded tensors and lengths.
    """
    # Filter out None samples (failed loads)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Sort by sequence length (descending) for pack_padded_sequence efficiency
    # (Optional, but good practice for RNNs)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [b["skeleton"] for b in batch]
    audios = [b["audio"] for b in batch]
    labels = [b["labels"] for b in batch]
    ids = [b["sample_id"] for b in batch]

    # Get lengths
    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    # Pad Sequences
    # Features padded with 0
    padded_skeletons = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Labels padded with BACKGROUND_LABEL (0)
    # This allows calculating loss on padding as "Background" class
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=BACKGROUND_LABEL
    )

    return {
        "skeleton": padded_skeletons,  # (B, T_max, 60)
        "audio": padded_audios,  # (B, T_max, 20)
        "labels": padded_labels,  # (B, L_max)
        "lengths": lengths,  # (B,)
        "label_lengths": label_lengths,  # (B,)
        "sample_ids": ids,  # List[str]
    }
