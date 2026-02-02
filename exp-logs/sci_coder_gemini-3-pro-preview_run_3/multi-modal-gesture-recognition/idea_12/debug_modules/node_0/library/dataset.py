import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

from library.config import Config
from library.data_utils import load_cached_dataset
from library.features import extract_features, FeatureNormalizer


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the SA-AKN model.
    Handles sliding window generation, on-the-fly augmentation (train),
    and feature normalization.
    """

    def __init__(self, split, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading raw data from cache.
        """
        self.split = split
        self.is_train = split == "train"

        # 1. Load Raw Data
        # Returns dict with 'features' (list of T,20,3), 'labels' (list of list of dicts), 'sample_ids'
        data_dict = load_cached_dataset(split, load_cached_data=load_cached_data)

        self.raw_skeletons = data_dict["features"]  # List of np.ndarray (T, 20, 3)
        self.raw_labels_meta = data_dict["labels"]  # List of parsed label lists
        self.sample_ids = data_dict["sample_ids"]  # List of strings

        # 2. Process Labels (Convert sparse dicts to dense frame-wise arrays)
        self.dense_labels = self._process_labels()

        # 3. Normalization Setup
        self.normalizer = FeatureNormalizer()
        norm_path = os.path.join(Config.CACHE_DIR, "normalizer_stats.npz")

        if self.is_train:
            # For training, we need to fit the normalizer on the training set.
            # We extract features without augmentation for the whole set to calculate stats.
            print("Computing features for normalization stats...")
            temp_features = []
            for skel in self.raw_skeletons:
                # Extract features (no augmentation)
                feat = extract_features(skel, augment=False)
                temp_features.append(feat)

            self.normalizer.fit(temp_features)
            self.normalizer.save(norm_path)
            print(f"Normalizer stats saved to {norm_path}")

            # For training, we do NOT store pre-computed features to save RAM
            # and allow on-the-fly augmentation. We keep self.raw_skeletons.
            self.processed_features = None

        else:
            # For val/test, load the normalizer
            if os.path.exists(norm_path):
                self.normalizer.load(norm_path)
            else:
                # Fallback: if val is loaded before train (unlikely pipeline), fit on current data
                print(
                    "Warning: Normalizer not found. Fitting on current split (suboptimal for Val/Test)."
                )
                temp_features = [
                    extract_features(s, augment=False) for s in self.raw_skeletons
                ]
                self.normalizer.fit(temp_features)

            # Pre-compute and normalize features for Val/Test to ensure consistency
            print(f"Pre-computing normalized features for {split}...")
            self.processed_features = []
            for skel in self.raw_skeletons:
                feat = extract_features(skel, augment=False)
                feat_norm = self.normalizer.transform(feat)
                self.processed_features.append(feat_norm)

        # 4. Generate Sliding Windows
        self.window_size = Config.WINDOW_SIZE
        self.stride = (
            Config.WINDOW_STRIDE_TRAIN if self.is_train else Config.WINDOW_STRIDE_TEST
        )

        # List of tuples: (sample_index, start_frame)
        self.windows = self._create_windows()

        print(
            f"Dataset {split} initialized. Samples: {len(self.sample_ids)}, Windows: {len(self.windows)}"
        )

    def _process_labels(self):
        """
        Converts list of label metadata into list of dense numpy arrays.
        """
        dense_labels_list = []

        for i, skel in enumerate(self.raw_skeletons):
            num_frames = skel.shape[0]
            # Initialize with Background Class ID (0)
            labels = np.full(num_frames, Config.BACKGROUND_CLASS_ID, dtype=np.int64)

            meta_list = self.raw_labels_meta[i]
            for m in meta_list:
                # m is dict: {'name': str, 'id': int, 'begin': int, 'end': int}
                # 'begin' and 'end' are 1-based indices from Matlab.

                # Clamp to valid range and convert to 0-based
                start = max(0, int(m["begin"]) - 1)
                end = min(num_frames, int(m["end"]))

                label_id = int(m["id"])
                if 0 <= start < end:
                    labels[start:end] = label_id

            dense_labels_list.append(labels)

        return dense_labels_list

    def _create_windows(self):
        """
        Generates a list of (sample_idx, start_frame) for sliding windows.
        """
        windows = []
        for sample_idx, skel in enumerate(self.raw_skeletons):
            num_frames = skel.shape[0]

            # If sequence is shorter than window, we take one window starting at 0 (will be padded)
            if num_frames <= self.window_size:
                windows.append((sample_idx, 0))
                continue

            # Sliding window logic
            # Range: 0 to num_frames - window_size, step = stride
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                windows.append((sample_idx, start))

            # Handle remainder: ensure the last frames are covered
            # If the last generated window doesn't reach the exact end, add one window ending at the very end.
            last_start = windows[-1][1]
            if last_start + self.window_size < num_frames:
                windows.append((sample_idx, num_frames - self.window_size))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns:
            features: Tensor (Window, InputDim)
            labels: Tensor (Window)
            sample_idx: int
            start_frame: int
        """
        sample_idx, start_frame = self.windows[idx]

        if self.is_train:
            # On-the-fly processing for Training
            full_skel = self.raw_skeletons[sample_idx]
            seq_len = full_skel.shape[0]

            # Slice raw skeleton
            end_frame = min(start_frame + self.window_size, seq_len)
            skel_slice = full_skel[start_frame:end_frame]

            # Augment & Extract
            feat_slice = extract_features(skel_slice, augment=True)

            # Normalize
            feat_slice = self.normalizer.transform(feat_slice)

        else:
            # Pre-computed features for Val/Test
            full_feat = self.processed_features[sample_idx]
            seq_len = full_feat.shape[0]

            end_frame = min(start_frame + self.window_size, seq_len)
            feat_slice = full_feat[start_frame:end_frame]

        # Get Labels
        full_labels = self.dense_labels[sample_idx]
        label_slice = full_labels[start_frame:end_frame]

        # Padding if necessary (for sequences shorter than window_size)
        current_len = feat_slice.shape[0]
        if current_len < self.window_size:
            pad_len = self.window_size - current_len

            # Pad features with 0
            feat_pad = np.zeros((pad_len, feat_slice.shape[1]), dtype=np.float32)
            feat_slice = np.concatenate([feat_slice, feat_pad], axis=0)

            # Pad labels with Background (0)
            label_pad = np.full(pad_len, Config.BACKGROUND_CLASS_ID, dtype=np.int64)
            label_slice = np.concatenate([label_slice, label_pad], axis=0)

        # Convert to Tensors
        features_tensor = torch.from_numpy(feat_slice).float()
        labels_tensor = torch.from_numpy(label_slice).long()

        return features_tensor, labels_tensor, sample_idx, start_frame
