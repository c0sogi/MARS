import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import (
    PATHS,
    GESTURE_MAP,
    SELECTED_JOINT_INDICES,
    JOINT_NAMES,
    SEED,
    get_hyperparams,
)
from library.utils import set_seed

# Ensure deterministic behavior across the module
set_seed(SEED)


class GestureDataset(Dataset):
    def __init__(self, data, is_train=False):
        """
        Dataset class for the Gesture Recognition task.

        Args:
            data (list of dict): List containing preprocessed samples.
                                 Each item must contain:
                                 - 'features_pos': (T, 12, 3) Normalized Joint Positions
                                 - 'features_audio': (T, 13) Aligned MFCCs
                                 - 'targets': (T,) Frame-wise labels
                                 - 'sample_id': str
            is_train (bool): If True, applies physically consistent augmentation.
        """
        self.data = data
        self.is_train = is_train
        self.hp = get_hyperparams()

        # Augmentation hyperparameters
        self.aug_sigma = self.hp.get("aug_sigma", 0.01)
        self.filter_width = self.hp.get("aug_temp_filter_width", 5)

    def __len__(self):
        return len(self.data)

    def _apply_augmentation(self, pos):
        """
        Applies temporally correlated noise to joint positions.
        Logic:
        1. Generate Gaussian Noise.
        2. Apply Temporal Low-Pass Filter (Moving Average) to smooth the noise.
        3. Add to positions.

        Args:
            pos (np.ndarray): (T, J, 3)
        Returns:
            np.ndarray: Augmented positions (T, J, 3)
        """
        T, J, C = pos.shape
        # Generate Gaussian Noise
        noise = np.random.normal(0, self.aug_sigma, size=(T, J, C))

        # Apply Temporal Low-Pass Filter (Simple 1D Convolution)
        kernel = np.ones(self.filter_width) / self.filter_width
        noise_smooth = np.zeros_like(noise)

        # Apply filter independently per joint/channel
        for j in range(J):
            for c in range(C):
                noise_smooth[:, j, c] = np.convolve(noise[:, j, c], kernel, mode="same")

        return pos + noise_smooth

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load Raw Normalized Positions: (T, 12, 3)
        pos = sample["features_pos"].copy()
        audio = sample["features_audio"].copy()  # (T, 13)
        targets = sample["targets"]

        # 1. Augmentation (Training Only)
        # Applied to positions BEFORE velocity calculation to maintain consistency
        if self.is_train:
            pos = self._apply_augmentation(pos)

        # 2. Derive Explicit Temporal Velocity
        # V[t] = P[t] - P[t-1], with V[0] = 0
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # 3. Flatten and Concatenate Features
        T = pos.shape[0]
        # Pos: (T, 12, 3) -> (T, 36)
        pos_flat = pos.reshape(T, -1)
        # Vel: (T, 12, 3) -> (T, 36)
        vel_flat = vel.reshape(T, -1)

        # Feature Vector: [Pos, Vel, Audio] -> (T, 36 + 36 + 13) = (T, 85)
        features = np.concatenate([pos_flat, vel_flat, audio], axis=1)

        # Convert to PyTorch Tensors
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)

        return {
            "features": features,
            "targets": targets,
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences and generate masks.
    Transposes features to (B, C, T) for Conv1D compatibility.
    """
    # Sort by length for efficient packing (optional)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    targets = [x["targets"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    # Pad sequences
    # features: (B, T_max, D)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    # targets: (B, T_max) - Padding with 0 (Background)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    # Generate Valid Frame Mask
    # (B, T_max)
    lengths = torch.tensor([x.shape[0] for x in features], dtype=torch.long)
    mask = torch.arange(features_padded.shape[1])[None, :] < lengths[:, None]
    mask = mask.float()

    # Transpose features for Multi-Granularity Stem (Conv1D)
    # Input: (B, T, C) -> Output: (B, C, T)
    features_padded = features_padded.transpose(1, 2)

    return features_padded, targets_padded, mask, sample_ids


def defensive_load_mat(mat_path):
    """
    Robustly parses .mat file to extract skeleton frames and labels.
    Handles variable struct/array formats in the provided dataset.
    """
    try:
        # Load mat with squeeze_me to simplify struct access
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, [], 0

        video = mat["Video"]
        frames = video.Frames
        num_frames = getattr(video, "NumFrames", 0)

        # Ensure frames is iterable
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        # Pre-allocate Skeleton Data: (T, 12, 3)
        # We only extract the 12 selected upper-body joints
        skeleton_data = np.zeros(
            (len(frames), len(SELECTED_JOINT_INDICES), 3), dtype=np.float32
        )

        for t, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton
            # Handle case where Skeleton is an array (multiple users) vs single object
            # We assume the first skeleton is the target user
            if isinstance(skel, np.ndarray):
                if skel.size > 0:
                    joints = (
                        skel[0]
                        if isinstance(skel[0], (scipy.io.matlab.mat_struct, np.void))
                        else skel
                    )
                else:
                    continue
            else:
                joints = skel

            # Extract WorldPosition for selected joints
            # Assuming joints is an array/struct where index corresponds to Joint ID
            # or it has a WorldPosition field.
            # Based on dataset description, Skeleton is an array of joint structures.

            # If joints is a single struct with a WorldPosition array:
            if hasattr(joints, "WorldPosition") and not isinstance(joints, np.ndarray):
                # This case is rare in Kinect data but possible
                pass

            # Standard Kinect format: joints is an array of 20 structs
            if isinstance(joints, np.ndarray) and len(joints) >= 20:
                for i, joint_idx in enumerate(SELECTED_JOINT_INDICES):
                    try:
                        wp = joints[joint_idx].WorldPosition
                        skeleton_data[t, i, 0] = wp.X
                        skeleton_data[t, i, 1] = wp.Y
                        skeleton_data[t, i, 2] = wp.Z
                    except AttributeError:
                        pass

        # Extract Label Intervals (for Train/Val)
        labels_list = []
        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            # Normalize to list
            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0:
                    raw_labels = np.array([raw_labels])
            elif not isinstance(raw_labels, list):
                raw_labels = np.array([raw_labels])

            for l in raw_labels:
                try:
                    name = l.Name
                    start = l.Begin
                    end = l.End
                    if name in GESTURE_MAP:
                        labels_list.append((GESTURE_MAP[name], start, end))
                except AttributeError:
                    pass

        return skeleton_data, labels_list, num_frames

    except Exception as e:
        # print(f"Error parsing {mat_path}: {e}")
        return None, [], 0


def process_audio(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCCs, and aligns them to the video frame count.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Extract MFCCs
        # n_fft=400 (25ms at 16kHz), hop_length=160 (10ms)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # Shape: (time, n_mfcc)

        # Align to Video Frames using Linear Interpolation
        audio_len = mfcc.shape[0]
        if audio_len == 0:
            return np.zeros((target_num_frames, 13), dtype=np.float32)

        # Generate indices for interpolation
        indices = np.linspace(0, audio_len - 1, target_num_frames)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.ceil(indices).astype(int)
        weights = indices - indices_floor

        # Clamp indices
        indices_floor = np.clip(indices_floor, 0, audio_len - 1)
        indices_ceil = np.clip(indices_ceil, 0, audio_len - 1)

        mfcc_np = mfcc.numpy()

        # Interpolate
        mfcc_aligned = (
            mfcc_np[indices_floor] * (1 - weights[:, None])
            + mfcc_np[indices_ceil] * weights[:, None]
        )

        return mfcc_aligned.astype(np.float32)

    except Exception as e:
        # Return zeros on failure
        return np.zeros((target_num_frames, 13), dtype=np.float32)


def load_data_with_cache(metadata_path, split, load_cached_data=True, sample_size=None):
    """
    Loads dataset from metadata. Implements caching to .npz format to speed up
    subsequent runs and avoid re-parsing MAT/Audio files.
    """
    cache_file = os.path.join(PATHS["cache"], f"{split}_data.npz")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                # Reconstruct list of dicts from concatenated arrays
                all_pos = data["all_pos"]
                all_audio = data["all_audio"]
                all_targets = data["all_targets"]
                indices = data["indices"]
                lengths = data["lengths"]
                sample_ids = data["sample_ids"]

                dataset = []
                count = len(sample_ids)
                if sample_size:
                    count = min(count, sample_size)

                for i in range(count):
                    start = indices[i]
                    length = lengths[i]
                    end = start + length

                    item = {
                        "features_pos": all_pos[start:end],
                        "features_audio": all_audio[start:end],
                        "targets": all_targets[start:end],
                        "sample_id": str(sample_ids[i]),
                    }
                    dataset.append(item)
                return dataset
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing from scratch...")

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")
    df = pd.read_csv(metadata_path)
    if sample_size:
        df = df.iloc[:sample_size]

    # Temporary lists for batch concatenation
    list_pos = []
    list_audio = []
    list_targets = []
    list_ids = []
    list_lengths = []

    dataset = []

    for _, row in df.iterrows():
        sid = row["sample_id"]
        mat_path = os.path.join(PATHS["input"], row["data_path"])
        audio_path = os.path.join(PATHS["input"], row["audio_path"])

        # Parse MAT file
        skel_data, labels_intervals, num_frames = defensive_load_mat(mat_path)
        if skel_data is None or num_frames == 0:
            continue

        # Feature Engineering: Normalization
        # Center relative to HipCenter (Index 0 in SELECTED_JOINTS)
        hip_center = skel_data[:, 0:1, :]  # (T, 1, 3)
        skel_norm = (skel_data - hip_center) * 0.001  # Scale mm to meters
        skel_norm = skel_norm.astype(np.float32)

        # Feature Engineering: Audio
        audio_data = process_audio(audio_path, num_frames)

        # Build Targets (Frame-wise)
        targets = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

        # Apply labels if available (Train/Val)
        if split != "test":
            for lbl, start, end in labels_intervals:
                # Matlab 1-based indices -> Python 0-based
                # Start is inclusive, End is inclusive in Matlab
                s = max(0, start - 1)
                e = min(num_frames, end)
                targets[s:e] = lbl

        # Store
        item = {
            "features_pos": skel_norm,
            "features_audio": audio_data,
            "targets": targets,
            "sample_id": sid,
        }
        dataset.append(item)

        # Append to lists for caching
        list_pos.append(skel_norm)
        list_audio.append(audio_data)
        list_targets.append(targets)
        list_ids.append(sid)
        list_lengths.append(num_frames)

    # 3. Save to Cache
    if len(list_pos) > 0:
        all_pos = np.concatenate(list_pos, axis=0)
        all_audio = np.concatenate(list_audio, axis=0)
        all_targets = np.concatenate(list_targets, axis=0)
        indices = np.cumsum([0] + list_lengths[:-1])
        lengths = np.array(list_lengths)
        sample_ids = np.array(list_ids)

        np.savez(
            cache_file,
            all_pos=all_pos,
            all_audio=all_audio,
            all_targets=all_targets,
            indices=indices,
            lengths=lengths,
            sample_ids=sample_ids,
        )
        print(f"Saved cache to {cache_file}")

    return dataset


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create Train, Val, and Test dataloaders.
    """
    hp = get_hyperparams()

    # Load Datasets
    train_data = load_data_with_cache(
        PATHS["train_meta"], "train", load_cached_data, hp["sample_size"]
    )
    val_data = load_data_with_cache(
        PATHS["val_meta"], "val", load_cached_data, hp["sample_size"]
    )
    test_data = load_data_with_cache(
        PATHS["test_meta"], "test", load_cached_data, hp["sample_size"]
    )

    # Initialize Dataset Objects
    train_ds = GestureDataset(train_data, is_train=True)
    val_ds = GestureDataset(val_data, is_train=False)
    test_ds = GestureDataset(test_data, is_train=False)

    # Configure Dataloaders
    # Pin memory for faster GPU transfer
    pin_memory = True if hp["device"] == "cuda" else False

    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp["num_workers"],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["num_workers"],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["num_workers"],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
