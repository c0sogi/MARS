import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library import config, utils


class GestureDataset(Dataset):
    """
    Dataset class for the Gesture Recognition task.
    Handles loading, caching, windowing, and on-the-fly augmentation of multi-modal data.
    """

    def __init__(
        self, split="train", load_cached_data=True, augment=False, debug=False
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            augment (bool): Whether to apply kinematic augmentation (only for training).
            debug (bool): If True, uses a small subset of data for debugging.
        """
        self.split = split
        self.augment = augment
        self.debug = debug

        # Determine metadata path
        if split == "train":
            self.metadata_path = config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Cache file path
        cache_filename = f"dataset_{split}{'_debug' if debug else ''}.npz"
        self.cache_path = os.path.join(config.WORKING_DIR, cache_filename)

        # Load data
        self.skeletons = []  # List of (T, J, 3) arrays
        self.audios = []  # List of (T, MFCC) arrays
        self.labels = []  # List of (T,) arrays
        self.sample_ids = []  # List of strings

        if load_cached_data and os.path.exists(self.cache_path):
            self._load_from_cache()
        else:
            self._process_and_cache()

        # Generate Sliding Windows
        self.windows = []
        self._create_windows()

        if self.split == "train":
            print(
                f"[{self.split.upper()}] Loaded {len(self.windows)} windows from {len(self.sample_ids)} sequences."
            )

    def _process_and_cache(self):
        """
        Reads metadata, parses raw files, extracts features, and saves to cache.
        """
        print(f"[{self.split.upper()}] Processing data from scratch...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        if self.debug:
            df = df.head(config.DEBUG_SUBSET_SIZE)

        skeletons_list = []
        audios_list = []
        labels_list = []
        ids_list = []

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

            # 1. Parse Skeleton
            # Returns (T, J, 3) in meters or None
            skel_data = utils.safe_parse_skeleton(data_path)

            if skel_data is None:
                # Skip broken samples
                continue

            num_frames = skel_data.shape[0]

            # 2. Parse Audio
            # Returns (T, MFCC) aligned to num_frames
            audio_data = utils.extract_audio_features(audio_path, num_frames)

            # 3. Parse Labels
            # Create frame-wise label array initialized to background (0)
            label_seq = np.full(num_frames, config.BACKGROUND_CLASS_ID, dtype=np.int64)

            if self.split != "test":
                try:
                    labels_meta = json.loads(row["labels"])
                    for l in labels_meta:
                        lid = int(l["id"])
                        start = int(l["begin"])
                        end = int(l["end"])
                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames - 1, end)
                        if start <= end:
                            label_seq[start : end + 1] = lid
                except Exception as e:
                    print(f"Error parsing labels for {sample_id}: {e}")

            skeletons_list.append(skel_data)
            audios_list.append(audio_data)
            labels_list.append(label_seq)
            ids_list.append(sample_id)

        # Save to cache
        # We use object arrays to store lists of variable-length arrays
        np.savez_compressed(
            self.cache_path,
            skeletons=np.array(skeletons_list, dtype=object),
            audios=np.array(audios_list, dtype=object),
            labels=np.array(labels_list, dtype=object),
            sample_ids=np.array(ids_list, dtype=object),
        )

        self.skeletons = skeletons_list
        self.audios = audios_list
        self.labels = labels_list
        self.sample_ids = ids_list

    def _load_from_cache(self):
        """
        Loads pre-processed lists from .npz file.
        """
        print(f"[{self.split.upper()}] Loading from cache: {self.cache_path}")
        data = np.load(self.cache_path, allow_pickle=True)
        self.skeletons = list(data["skeletons"])
        self.audios = list(data["audios"])
        self.labels = list(data["labels"])
        self.sample_ids = list(data["sample_ids"])

    def _create_windows(self):
        """
        Generates metadata for sliding windows: (seq_idx, start_frame, end_frame).
        """
        window_size = config.WINDOW_SIZE
        stride = config.STRIDE

        # For validation and test, we might want a smaller stride or different logic,
        # but consistent sliding window is standard for TCNs.
        # For test, we use 50% overlap as per prompt description for inference.
        if self.split in ["val", "test"]:
            stride = window_size // 2

        self.windows = []

        for seq_idx, skel in enumerate(self.skeletons):
            num_frames = skel.shape[0]

            # If sequence is shorter than window, pad it (handled in __getitem__)
            if num_frames <= window_size:
                self.windows.append((seq_idx, 0, num_frames))
                continue

            # Slide window
            for start in range(0, num_frames, stride):
                end = start + window_size
                # If we go slightly over, we still take the window and pad/clip in getitem
                # However, usually we stop if start >= num_frames
                if start >= num_frames:
                    break

                # Ensure we don't have a tiny window at the end unless it's substantial
                # But for coverage, we include it.
                self.windows.append((seq_idx, start, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns:
            features (torch.Tensor): (Time, InputDim)
            labels (torch.Tensor): (Time,)
            mask (torch.Tensor): (Time,) - 1 for valid frames, 0 for padded
        """
        seq_idx, start, end = self.windows[idx]

        raw_skel = self.skeletons[seq_idx]  # (T_full, J, 3)
        raw_audio = self.audios[seq_idx]  # (T_full, MFCC)
        raw_labels = self.labels[seq_idx]  # (T_full,)

        full_len = raw_skel.shape[0]
        window_size = config.WINDOW_SIZE

        # --- Context Handling for Kinematics ---
        # To compute valid velocity/acceleration for the first frame of the window,
        # we need context (previous frames).
        # Velocity needs 1 prev frame, Accel needs 2 prev frames.
        context_frames = 2

        # Determine fetch range
        fetch_start = start - context_frames
        fetch_end = end  # We don't need future context for causality, just current

        # Handle boundary conditions for fetch
        pad_front = 0
        if fetch_start < 0:
            pad_front = abs(fetch_start)
            fetch_start = 0

        # Slice raw data
        skel_slice = raw_skel[fetch_start:fetch_end]

        # Pad front if needed (replicate first frame)
        if pad_front > 0:
            first_frame = skel_slice[0:1]
            skel_slice = np.concatenate(
                [np.repeat(first_frame, pad_front, axis=0), skel_slice], axis=0
            )

        # --- Augmentation ---
        # Apply augmentation on the raw positions BEFORE computing derivatives
        if self.augment:
            skel_slice = utils.augment_skeleton(skel_slice)

        # --- Compute Kinematics ---
        # Returns (T_slice, FeatureDim)
        # Note: compute_kinematics sets first frame deriv to 0.
        # Because we added 2 context frames, the derivatives at index 2 (which corresponds to 'start')
        # will be valid relative to the context.
        kinematics = utils.compute_kinematics(skel_slice)

        # Remove context frames to get back to target window
        kinematics = kinematics[context_frames:]

        # --- Audio Slice ---
        # Audio doesn't need derivative context, just direct slicing
        audio_slice = raw_audio[start:end]
        label_slice = raw_labels[start:end]

        # --- Padding to Window Size ---
        # If the slice is shorter than window_size (end of sequence), pad with zeros/background
        current_len = kinematics.shape[0]
        pad_len = window_size - current_len

        mask = np.ones(window_size, dtype=np.float32)

        if pad_len > 0:
            # Pad Features with 0
            k_pad = np.zeros((pad_len, kinematics.shape[1]), dtype=np.float32)
            kinematics = np.concatenate([kinematics, k_pad], axis=0)

            a_pad = np.zeros((pad_len, audio_slice.shape[1]), dtype=np.float32)
            audio_slice = np.concatenate([audio_slice, a_pad], axis=0)

            # Pad Labels with Background (0)
            l_pad = np.full(pad_len, config.BACKGROUND_CLASS_ID, dtype=np.int64)
            label_slice = np.concatenate([label_slice, l_pad], axis=0)

            # Mask
            mask[current_len:] = 0.0

        elif pad_len < 0:
            # Should not happen with current logic, but safety clip
            kinematics = kinematics[:window_size]
            audio_slice = audio_slice[:window_size]
            label_slice = label_slice[:window_size]

        # --- Fusion ---
        # Concatenate Kinematics and Audio
        # Shape: (Window, InputDim)
        features = np.concatenate([kinematics, audio_slice], axis=1)

        # Convert to Torch Tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(label_slice).long()
        mask = torch.from_numpy(mask).float()

        return features, labels, mask
