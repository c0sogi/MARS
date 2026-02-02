import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.feature_engineering import FeatureEngineer

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Gesture Recognition.

    Modes:
    - Train: Returns fixed-size sliding windows (Config.WINDOW_SIZE).
    - Val/Test: Returns full unsegmented sequences.
    """

    def __init__(
        self,
        data_dict,
        is_train=True,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
    ):
        self.features = data_dict["features"]
        self.labels = data_dict["labels"]
        self.seq_lengths = data_dict["seq_lengths"]
        self.sample_ids = data_dict["sample_ids"]
        self.is_train = is_train
        self.window_size = window_size
        self.stride = stride

        # Calculate start index of each sequence in the flattened arrays
        self.seq_start_indices = np.zeros(len(self.seq_lengths), dtype=np.int64)
        if len(self.seq_lengths) > 0:
            self.seq_start_indices[1:] = np.cumsum(self.seq_lengths)[:-1]

        # Pre-calculate indices for __getitem__
        self.indices = []
        self._prepare_indices()

    def _prepare_indices(self):
        """
        Generates the list of accessible items.
        - Training: List of (seq_idx, start_offset) for every valid window.
        - Inference: List of (seq_idx) for every sequence.
        """
        self.indices = []

        if self.is_train:
            # Sliding window generation
            for seq_idx, length in enumerate(self.seq_lengths):
                # If sequence is shorter than window, we can pad or skip.
                # Given average length ~45 frames and window 64, we might need handling.
                # However, prompt says "average atomic gesture duration ~45",
                # but sequences contain multiple gestures and are usually longer.
                # If a sequence is shorter than window_size, we skip it or take it once if we implement padding.
                # Here we assume sequences are generally long enough or we take at least one crop if possible.

                if length < self.window_size:
                    # Option: Pad or just take what we have?
                    # For simplicity and consistency with fixed input size models, we skip strictly short seqs
                    # or we could pad. Let's assume data is sufficient or we skip.
                    # Actually, let's pad if short, or just skip to be safe against noise.
                    # Given the prompt implies continuous streams, we'll stick to valid windows.
                    pass

                # Generate windows
                # Range: [0, length - window_size] with step stride
                # We ensure we cover the end by checking the last fit
                max_start = length - self.window_size
                if max_start >= 0:
                    for start in range(0, max_start + 1, self.stride):
                        self.indices.append((seq_idx, start))
        else:
            # Full sequences
            for seq_idx in range(len(self.seq_lengths)):
                self.indices.append(seq_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.is_train:
            seq_idx, start_offset = self.indices[idx]
            global_start = self.seq_start_indices[seq_idx] + start_offset
            global_end = global_start + self.window_size

            # Extract window
            feat_window = self.features[global_start:global_end]
            label_window = self.labels[global_start:global_end]

            # Convert to tensors
            # Features: (Window, Input_Dim)
            # Labels: (Window,)
            return (
                torch.from_numpy(feat_window).float(),
                torch.from_numpy(label_window).long(),
            )

        else:
            seq_idx = self.indices[idx]
            global_start = self.seq_start_indices[seq_idx]
            length = self.seq_lengths[seq_idx]
            global_end = global_start + length

            # Extract full sequence
            feat_seq = self.features[global_start:global_end]
            label_seq = self.labels[global_start:global_end]
            sample_id = self.sample_ids[seq_idx]

            # Return tuple with sample_id for submission generation
            return (
                torch.from_numpy(feat_seq).float(),
                torch.from_numpy(label_seq).long(),
                sample_id,
            )


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    fe = FeatureEngineer()

    # 1. Load/Process Data
    # Train
    train_data = fe.process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    # Val
    val_data = fe.process_dataset(
        Config.VAL_METADATA_PATH,
        Config.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    # Test
    test_data = fe.process_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    # 2. Create Datasets
    train_dataset = GestureDataset(train_data, is_train=True)
    val_dataset = GestureDataset(val_data, is_train=False)
    test_dataset = GestureDataset(test_data, is_train=False)

    # 3. Create DataLoaders
    # Train: Shuffle, Batch Size
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain shape consistency
    )

    # Val/Test: No Shuffle, Batch Size 1 (variable lengths)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    return train_loader, val_loader, test_loader
