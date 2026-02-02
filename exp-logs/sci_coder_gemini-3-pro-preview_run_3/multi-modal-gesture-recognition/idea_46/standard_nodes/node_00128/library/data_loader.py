import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Paths, DataConfig, LABEL_MAP
from library.utils import set_seed

# Ensure reproducibility
set_seed(42)


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Gesture Recognition.
    Supports both windowed data (training) and full sequences (validation/testing).
    """

    def __init__(self, features, labels=None, sample_ids=None):
        """
        Args:
            features (list of np.ndarray): List of feature arrays.
            labels (list of np.ndarray, optional): List of label arrays.
            sample_ids (list of str, optional): List of sample IDs.
        """
        self.features = features
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x = torch.from_numpy(self.features[idx]).float()

        item = {"features": x}

        if self.labels is not None:
            y = torch.from_numpy(self.labels[idx]).long()
            item["labels"] = y

        if self.sample_ids is not None:
            item["sample_id"] = self.sample_ids[idx]

        return item


def polymorphic_extract(mat_struct, field_name):
    """
    Robustly extracts a field from a MATLAB struct, handling differences
    between scalar structs, struct arrays, and object references.
    """
    if hasattr(mat_struct, field_name):
        return getattr(mat_struct, field_name)
    elif isinstance(mat_struct, dict) and field_name in mat_struct:
        return mat_struct[field_name]
    else:
        return None


def load_mat_file(path):
    """
    Loads a .mat file with error handling.
    """
    try:
        # struct_as_record=False loads structs as objects with attributes
        # squeeze_me=True simplifies arrays
        return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_skeleton_features(mat_path):
    """
    Extracts skeleton features from .mat file.
    Implements:
    1. Polymorphic parsing.
    2. Raw Millimeter retention.
    3. Root-Relative Centering.
    4. Central Difference Kinematics.
    """
    mat = load_mat_file(mat_path)
    if mat is None:
        return None

    # Navigate to frames
    video = polymorphic_extract(mat, "Video")
    if video is None:
        return None

    frames = polymorphic_extract(video, "Frames")
    if frames is None:
        return None

    # Handle single frame vs list of frames
    if not isinstance(frames, np.ndarray):
        frames = np.array([frames])

    num_frames = len(frames)
    num_joints = DataConfig.NUM_JOINTS

    # Pre-allocate: (T, J, 3)
    # We use raw millimeters, so values are ~1000s
    skeleton_pos = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    # Parse frames
    for t, frame in enumerate(frames):
        skel = polymorphic_extract(frame, "Skeleton")
        if skel is None:
            continue

        # Skeleton might be an array of joints or a struct with fields
        # The description says "Skeletons array" containing "Skeleton structures"
        # But usually in these datasets, frame.Skeleton is a struct array of 20 joints

        joints = skel
        if not isinstance(joints, (list, np.ndarray)):
            # If it's a single object, it might be malformed or single joint
            # Try to treat as iterable or skip
            try:
                joints = [joints]
            except:
                continue

        # Extract joints
        # Order is assumed fixed as per description (HipCenter is index 0)
        for j_idx, joint in enumerate(joints):
            if j_idx >= num_joints:
                break

            wp = polymorphic_extract(joint, "WorldPosition")
            if wp is not None:
                # WorldPosition has X, Y, Z attributes
                try:
                    skeleton_pos[t, j_idx, 0] = float(wp.X)
                    skeleton_pos[t, j_idx, 1] = float(wp.Y)
                    skeleton_pos[t, j_idx, 2] = float(wp.Z)
                except:
                    pass

    # 1. Root-Relative Centering
    # HipCenter is index 0. Subtract its position from all joints for each frame.
    if DataConfig.CENTER_SKELETON:
        hip_center = skeleton_pos[:, 0:1, :]  # (T, 1, 3)
        skeleton_pos = skeleton_pos - hip_center

    # 2. Kinematics using Central Differences
    # Position: (T, J, 3)
    pos = skeleton_pos

    # Velocity: (T, J, 3)
    if DataConfig.USE_CENTRAL_DIFFERENCE:
        vel = np.gradient(pos, axis=0)
        acc = np.gradient(vel, axis=0)
    else:
        # Fallback to forward difference (padded)
        vel = np.diff(pos, axis=0, prepend=pos[0:1])
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

    # Flatten joints and coordinates: (T, J*3)
    pos_flat = pos.reshape(num_frames, -1)
    vel_flat = vel.reshape(num_frames, -1)
    acc_flat = acc.reshape(num_frames, -1)

    # Concatenate: (T, Features) -> 20*3*3 = 180 features
    features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)

    return features


def extract_audio_features(audio_path, target_num_frames):
    """
    Extracts MFCC features and aligns them to the video frame count.
    """
    if not os.path.exists(audio_path):
        # Return zeros if audio missing
        return np.zeros((target_num_frames, DataConfig.N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary (though config says 16000 is expected)
        if sample_rate != DataConfig.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sample_rate, DataConfig.AUDIO_SR)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=DataConfig.AUDIO_SR,
            n_mfcc=DataConfig.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # Shape: (1, n_mfcc, time_steps)

        # Interpolate to match video frames
        # Input to interpolate needs to be (Batch, Channels, Length)
        # We treat n_mfcc as channels
        mfcc = mfcc.permute(
            0, 2, 1
        )  # (1, time, n_mfcc) -> wait, interpolate expects (N, C, L)

        # Correct for interpolate: (1, n_mfcc, time_source)
        mfcc_in = mfcc.permute(0, 2, 1)  # (1, n_mfcc, time_source)

        mfcc_aligned = F.interpolate(
            mfcc_in, size=target_num_frames, mode="linear", align_corners=False
        )  # (1, n_mfcc, target_frames)

        # Transpose back to (target_frames, n_mfcc)
        mfcc_out = mfcc_aligned.squeeze(0).permute(1, 0).numpy()

        return mfcc_out

    except Exception as e:
        # print(f"Audio processing error {audio_path}: {e}")
        return np.zeros((target_num_frames, DataConfig.N_MFCC), dtype=np.float32)


def create_dense_labels(labels_meta, num_frames):
    """
    Converts sparse label metadata (start, end, id) to a dense frame-wise label array.
    Background class is 0.
    """
    dense_labels = np.zeros(num_frames, dtype=np.int64)  # Class 0 is background

    for label in labels_meta:
        gid = label["id"]
        start = max(0, label["begin"] - 1)  # 1-based to 0-based
        end = min(num_frames, label["end"])

        if start < end:
            dense_labels[start:end] = gid

    return dense_labels


def process_dataset(csv_path, mode="train", debug_size=None):
    """
    Process raw files listed in CSV into features and labels.
    """
    df = pd.read_csv(csv_path)

    if debug_size is not None:
        df = df.head(debug_size)

    all_features = []
    all_labels = []
    all_ids = []

    input_dir = Paths.INPUT

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(input_dir, row["data_path"])
        audio_path = os.path.join(input_dir, row["audio_path"])

        # 1. Extract Skeleton
        skel_features = extract_skeleton_features(data_path)
        if skel_features is None:
            continue

        num_frames = skel_features.shape[0]

        # 2. Extract Audio
        audio_features = extract_audio_features(audio_path, num_frames)

        # 3. Early Fusion (Concatenation)
        # Skeleton: 180 dims, Audio: 13 dims
        combined_features = np.concatenate([skel_features, audio_features], axis=1)

        # 4. Labels
        if mode != "test":
            labels_meta = json.loads(row["labels"])
            dense_labels = create_dense_labels(labels_meta, num_frames)
        else:
            dense_labels = np.zeros(num_frames, dtype=np.int64)

        # 5. Sampling Strategy
        if mode == "train":
            # Sliding Window
            win_size = DataConfig.WINDOW_SIZE
            stride = DataConfig.STRIDE

            # If sequence is shorter than window, pad it
            if num_frames < win_size:
                pad_len = win_size - num_frames
                # Pad features with zeros (or edge)
                feat_pad = np.pad(
                    combined_features, ((0, pad_len), (0, 0)), mode="constant"
                )
                lbl_pad = np.pad(
                    dense_labels, (0, pad_len), mode="constant", constant_values=0
                )

                all_features.append(feat_pad)
                all_labels.append(lbl_pad)
                all_ids.append(sample_id)
            else:
                # Generate windows
                for start_idx in range(0, num_frames - win_size + 1, stride):
                    end_idx = start_idx + win_size
                    all_features.append(combined_features[start_idx:end_idx])
                    all_labels.append(dense_labels[start_idx:end_idx])
                    all_ids.append(sample_id)
        else:
            # Full Sequence (Val/Test)
            all_features.append(combined_features)
            all_labels.append(dense_labels)
            all_ids.append(sample_id)

    return all_features, all_labels, all_ids


def load_data(load_cached_data=True, debug_size=DataConfig.DEBUG_SAMPLE_SIZE):
    """
    Main entry point to load data. Handles caching.

    Args:
        load_cached_data (bool): Whether to try loading from .npz cache.
        debug_size (int or None): Limit number of samples for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    cache_dir = Paths.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    suffix = f"_debug{debug_size}" if debug_size else ""
    train_cache = os.path.join(cache_dir, f"train{suffix}.npz")
    val_cache = os.path.join(cache_dir, f"val{suffix}.npz")
    test_cache = os.path.join(cache_dir, f"test{suffix}.npz")

    datasets = {}
    modes = [
        ("train", Paths.TRAIN_CSV, train_cache),
        ("val", Paths.VAL_CSV, val_cache),
        ("test", Paths.TEST_CSV, test_cache),
    ]

    for mode, csv_path, cache_path in modes:
        loaded = False
        features, labels, ids = [], [], []

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                # np.load with object arrays needs careful handling
                # We stored them as object arrays of arrays
                # Explicitly cast to float32/int64 to avoid object dtype issues
                features = [f.astype(np.float32) for f in data["features"]]
                labels = [l.astype(np.int64) for l in data["labels"]]
                ids = list(data["ids"])
                loaded = True
                print(f"Loaded {mode} data from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                loaded = False

        # Process from scratch if not loaded
        if not loaded:
            print(f"Processing {mode} data from scratch...")
            features, labels, ids = process_dataset(
                csv_path, mode=mode, debug_size=debug_size
            )

            # Save to cache
            # Use dtype=object for variable length arrays (val/test) or consistent lists
            np.savez_compressed(
                cache_path,
                features=np.array(features, dtype=object),
                labels=np.array(labels, dtype=object),
                ids=np.array(ids, dtype=object),
            )
            print(f"Saved {mode} data to cache: {cache_path}")

        # Create Dataset
        datasets[mode] = GestureDataset(features, labels, ids)

        # Print stats
        # Calculate total frames
        total_frames = sum([f.shape[0] for f in features])
        print(
            f"{mode.capitalize()} Set: {len(features)} samples, {total_frames} total frames."
        )
        if len(features) > 0:
            print(f"  Feature Shape (Example): {features[0].shape}")

    return datasets["train"], datasets["val"], datasets["test"]


def get_dataloaders(batch_size=32, num_workers=2, debug_size=None):
    """
    Factory function to get PyTorch DataLoaders.
    """
    train_ds, val_ds, test_ds = load_data(load_cached_data=True, debug_size=debug_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Val and Test use batch_size=1 for full sequence inference
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
