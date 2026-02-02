import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library import config, utils


def augment_skeleton(skeleton_data):
    """
    Applies random rotation (Y-axis) and scaling to raw skeleton data.
    Args:
        skeleton_data: (T, 20, 3) numpy array in mm.
    Returns:
        augmented_data: (T, 20, 3) numpy array.
    """
    # Random rotation angle (radians)
    # Approx +/- 17 degrees
    theta = np.random.uniform(-0.3, 0.3)

    # Rotation matrix around Y-axis
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    # Random scaling factor
    scale = np.random.uniform(0.9, 1.1)

    # Apply transformations
    # Reshape to (N, 3) for matrix multiplication
    T, J, C = skeleton_data.shape
    flat_data = skeleton_data.reshape(-1, 3)

    # Rotate
    rotated_data = np.dot(flat_data, R.T)

    # Scale
    augmented_data = rotated_data * scale

    return augmented_data.reshape(T, J, C)


def load_raw_data(mode, load_cached_data=True):
    """
    Loads raw skeleton and audio data with caching.
    Does NOT compute kinematics yet, to allow for augmentation.
    """
    cache_file = os.path.join(config.WORKING_DIR, f"dataset_raw_{mode}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            raw_skel = list(data["raw_skel"])
            raw_audio = list(data["raw_audio"])
            labels = list(data["labels"])
            sample_ids = list(data["sample_ids"])
            print(f"Loaded raw {mode} data from cache: {len(raw_skel)} samples.")
            return raw_skel, raw_audio, labels, sample_ids
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch
    metadata_path = os.path.join(config.METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if config.DEBUG:
        df = df.head(config.SUBSET_SIZE)
        print(f"DEBUG MODE: Processing subset of {len(df)} samples.")

    raw_skel_list = []
    raw_audio_list = []
    labels_list = []
    ids_list = []

    print(f"Processing raw {mode} data...")

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # Load raw skeleton (T, 20, 3)
        skeleton_pos = utils.load_robust_mat(mat_path)
        if skeleton_pos is None:
            continue

        num_frames = skeleton_pos.shape[0]

        # Load audio features (T, 13) - Audio is pre-processed as it's not augmented spatially
        audio_feats = utils.extract_audio_features(audio_path, num_frames)

        # Parse Labels
        # Ensure lengths match exactly for safety
        min_len = min(num_frames, audio_feats.shape[0])

        # Truncate to min_len
        skeleton_pos = skeleton_pos[:min_len]
        audio_feats = audio_feats[:min_len]

        seq_labels = np.zeros(min_len, dtype=np.int64)
        if mode != "test":
            label_info = json.loads(row["labels"])
            for l in label_info:
                gid = l["id"]
                # 1-based index in metadata to 0-based
                start = max(0, l["begin"] - 1)
                end = min(min_len, l["end"])
                if start < end:
                    seq_labels[start:end] = gid

        raw_skel_list.append(skeleton_pos.astype(np.float32))
        raw_audio_list.append(audio_feats.astype(np.float32))
        labels_list.append(seq_labels)
        ids_list.append(sample_id)

    # 3. Save Cache
    # Use object array for ragged lists
    np.savez_compressed(
        cache_file,
        raw_skel=np.array(raw_skel_list, dtype=object),
        raw_audio=np.array(raw_audio_list, dtype=object),
        labels=np.array(labels_list, dtype=object),
        sample_ids=np.array(ids_list, dtype=object),
    )

    print(f"Processed and cached {len(raw_skel_list)} raw samples for {mode}.")
    return raw_skel_list, raw_audio_list, labels_list, ids_list


class GestureDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        self.mode = mode
        self.window_size = config.WINDOW_SIZE
        self.stride = config.STRIDE

        # Load raw data
        self.raw_skel, self.raw_audio, self.labels, self.ids = load_raw_data(
            mode, load_cached_data
        )

        # Generate sliding windows
        self.windows = []
        for sample_idx, (skel, audio) in enumerate(zip(self.raw_skel, self.raw_audio)):
            seq_len = skel.shape[0]

            # If sequence is shorter than window, take one window (will be padded)
            if seq_len <= self.window_size:
                self.windows.append((sample_idx, 0))
                continue

            # Sliding window
            # We want to ensure we cover the end of the sequence too
            num_windows = (seq_len - self.window_size) // self.stride + 1

            for w in range(num_windows):
                start_frame = w * self.stride
                self.windows.append((sample_idx, start_frame))

            # Check if we missed the tail
            last_start = (num_windows - 1) * self.stride
            if last_start + self.window_size < seq_len:
                # Add one final window aligned to the end?
                # Or just stride one more time.
                # Standard approach: add a window starting such that it covers the end
                # But simple striding is usually sufficient if stride is small enough.
                # Let's add one last window if significant portion remains
                remaining = seq_len - (last_start + self.window_size)
                if remaining > 0:
                    self.windows.append((sample_idx, seq_len - self.window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.windows[idx]

        # Retrieve raw sequences
        raw_skel_seq = self.raw_skel[sample_idx]  # (T, 20, 3)
        audio_seq = self.raw_audio[sample_idx]  # (T, 13)
        label_seq = self.labels[sample_idx]  # (T,)

        seq_len = raw_skel_seq.shape[0]
        end_frame = start_frame + self.window_size

        # Initialize buffers
        skel_window = np.zeros((self.window_size, 20, 3), dtype=np.float32)
        audio_window = np.zeros(
            (self.window_size, audio_seq.shape[1]), dtype=np.float32
        )
        label_window = np.zeros((self.window_size,), dtype=np.int64)

        # Calculate copy ranges
        # If sequence is shorter than window (padding needed)
        # Or if we are at the end and need padding (though logic above tries to avoid this for long seqs)

        actual_end = min(end_frame, seq_len)
        copy_len = actual_end - start_frame

        # Copy data
        skel_window[:copy_len] = raw_skel_seq[start_frame:actual_end]
        audio_window[:copy_len] = audio_seq[start_frame:actual_end]
        label_window[:copy_len] = label_seq[start_frame:actual_end]

        # Padding (Repeat last frame or zero pad? Zero pad is standard for RNNs usually,
        # but for kinematics repeating might be safer to avoid huge velocity spikes at end.
        # However, utils.compute_kinematics handles 0s fine. Let's stick to zero padding for simplicity
        # as initialized.)

        # Augmentation (Train only)
        if self.mode == "train":
            skel_window = augment_skeleton(skel_window)

        # Compute Kinematics
        # utils.compute_kinematics expects (T, 20, 3) and returns (T, 180)
        # It handles unit conversion (mm -> m) and derivatives
        kinematic_feats = utils.compute_kinematics(skel_window)

        # Concatenate with Audio
        # (T, 180) + (T, 13) -> (T, 193)
        features = np.concatenate([kinematic_feats, audio_window], axis=1)

        return torch.from_numpy(features), torch.from_numpy(label_window)
