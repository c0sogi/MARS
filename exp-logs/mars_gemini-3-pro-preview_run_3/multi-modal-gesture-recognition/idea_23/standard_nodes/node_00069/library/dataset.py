import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library import config, features


class GestureDataset(Dataset):
    """
    PyTorch Dataset for WES-KN (Idea 23).
    Handles sliding window generation, feature fusion, and label alignment.
    """

    def __init__(
        self,
        skeletons,
        audios,
        labels_list,
        sample_ids,
        stats,
        augment=False,
        stride=None,
    ):
        """
        Args:
            skeletons: List of np.ndarray (T, 20, 3)
            audios: List of np.ndarray (T, 13)
            labels_list: List of lists of dicts (parsed labels)
            sample_ids: List of string IDs
            stats: Dict containing normalization statistics
            augment: Boolean, whether to apply kinematic augmentation
            stride: Integer, sliding window stride. Defaults to config.STRIDE.
        """
        self.skeletons = skeletons
        self.audios = audios
        self.labels_list = labels_list
        self.sample_ids = sample_ids
        self.stats = stats
        self.augment = augment
        self.window_size = config.WINDOW_SIZE
        self.stride = stride if stride is not None else config.STRIDE

        # Pre-compute features for validation/test to speed up inference
        # For training, we compute on-the-fly to allow random augmentation
        self.precomputed_features = None
        if not self.augment:
            self.precomputed_features = []
            for i in range(len(self.skeletons)):
                feat = features.process_sample(
                    self.skeletons[i], self.audios[i], self.stats, augment=False
                )
                self.precomputed_features.append(feat)

        # Build the index of windows
        # List of tuples: (sample_index, start_frame)
        self.windows = self._build_window_index()

    def _build_window_index(self):
        windows = []
        for idx, skel in enumerate(self.skeletons):
            num_frames = skel.shape[0]

            # Case 1: Sequence shorter than window
            if num_frames < self.window_size:
                # We will pad this single window in __getitem__
                windows.append((idx, 0))
                continue

            # Case 2: Standard sliding window
            # Ensure we cover the end of the sequence
            # range(start, stop, step)
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                windows.append((idx, start))

            # Handle remainder: if the last window didn't reach the very end
            last_start = windows[-1][1]
            if last_start + self.window_size < num_frames:
                # Add a window ending exactly at the last frame
                windows.append((idx, num_frames - self.window_size))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.windows[idx]

        # 1. Get Full Sequence Features
        if self.augment:
            # Compute on-the-fly with random augmentation
            full_features = features.process_sample(
                self.skeletons[sample_idx],
                self.audios[sample_idx],
                self.stats,
                augment=True,
            )
        else:
            # Retrieve pre-computed deterministic features
            full_features = self.precomputed_features[sample_idx]

        num_frames = full_features.shape[0]

        # 2. Extract Window
        # Handle padding for short sequences
        if num_frames < self.window_size:
            # Create padded array
            window_feat = np.zeros(
                (self.window_size, config.INPUT_DIR), dtype=np.float32
            )  # Wait, INPUT_DIR is path.
            # Correction: Use full_features.shape[1] which is INPUT_DIM (193)
            window_feat = np.zeros(
                (self.window_size, full_features.shape[1]), dtype=np.float32
            )

            # Fill with available data
            window_feat[:num_frames] = full_features

            # Update start_frame for label logic (it's 0)
            actual_start = 0
            actual_end = num_frames
        else:
            actual_start = start_frame
            actual_end = start_frame + self.window_size
            window_feat = full_features[actual_start:actual_end]

        # 3. Generate Targets
        # Initialize with Background class (0)
        targets = np.full(
            (self.window_size,), config.BACKGROUND_CLASS_ID, dtype=np.int64
        )

        sample_labels = self.labels_list[sample_idx]

        for label in sample_labels:
            # Label: {name, id, begin, end} (1-based from metadata)
            # Adjust to 0-based indexing for calculation if necessary,
            # but metadata 'begin'/'end' are frame indices.
            # Assuming inclusive range [begin, end]

            l_start = label["begin"]
            l_end = label["end"]
            l_id = label["id"]

            # Check intersection with window [actual_start, actual_start + window_size)
            # Window end is exclusive in Python slicing logic
            w_start = actual_start
            w_end = actual_start + self.window_size

            # Intersection range
            inter_start = max(l_start, w_start)
            inter_end = min(l_end, w_end - 1)  # l_end is inclusive, w_end is exclusive

            if inter_start <= inter_end:
                # Map to window local coordinates
                loc_start = inter_start - w_start
                loc_end = inter_end - w_start + 1  # +1 for python slice inclusion

                # Fill target
                targets[loc_start:loc_end] = l_id

        # 4. Convert to Tensor
        feat_tensor = torch.from_numpy(window_feat)
        target_tensor = torch.from_numpy(targets)

        # Return metadata for inference reconstruction
        sample_id = self.sample_ids[sample_idx]

        return feat_tensor, target_tensor, sample_id, start_frame


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    # 1. Load Data & Stats
    train_data, val_data, test_data, stats = features.load_data_and_stats(
        load_cached_data=load_cached_data
    )

    # 2. Create Datasets
    # Train: Augmentation ON, Standard Stride
    train_dataset = GestureDataset(
        skeletons=train_data["skeleton"],
        audios=train_data["audio"],
        labels_list=train_data["labels"],
        sample_ids=train_data["sample_ids"],
        stats=stats,
        augment=True,
        stride=config.STRIDE,
    )

    # Val: Augmentation OFF, Standard Stride (for consistent metric evaluation)
    val_dataset = GestureDataset(
        skeletons=val_data["skeleton"],
        audios=val_data["audio"],
        labels_list=val_data["labels"],
        sample_ids=val_data["sample_ids"],
        stats=stats,
        augment=False,
        stride=config.STRIDE,
    )

    # Test: Augmentation OFF, Standard Stride (50% overlap as per config)
    test_dataset = GestureDataset(
        skeletons=test_data["skeleton"],
        audios=test_data["audio"],
        labels_list=test_data["labels"],
        sample_ids=test_data["sample_ids"],
        stats=stats,
        augment=False,
        stride=config.STRIDE,
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
