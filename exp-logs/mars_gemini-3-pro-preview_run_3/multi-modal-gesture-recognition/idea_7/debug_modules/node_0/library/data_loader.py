import os
import json
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# ==========================================
# Helper Functions
# ==========================================


def load_mat_file(path):
    """Safely load .mat file."""
    try:
        # struct_as_record=False allows accessing fields as attributes
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_audio_features(audio_path, target_num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    """
    try:
        # Load audio
        waveform, sample_rate = sf.read(audio_path)
        # Convert to torch tensor: (Channels, Time)
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        else:
            waveform = waveform.T

        waveform = torch.from_numpy(waveform).float()

        # Resample if necessary
        if sample_rate != config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Compute MFCC
        # We use a hop length that roughly corresponds to video frame rate,
        # but we will force align via interpolation anyway.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.AUDIO_MFCC_DIM,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time_steps)

        # Average over channels if stereo
        if mfcc.size(0) > 1:
            mfcc = mfcc.mean(dim=0, keepdim=True)

        # Align to video frames using interpolation
        # Input to interpolate must be (Batch, Channels, Length)
        # Current: (1, n_mfcc, time)
        if mfcc.size(-1) > 0:
            mfcc = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )
        else:
            # Handle empty audio case
            mfcc = torch.zeros(1, config.AUDIO_MFCC_DIM, target_num_frames)

        # Output shape: (NumFrames, n_mfcc)
        return mfcc.squeeze(0).transpose(0, 1).numpy()

    except Exception as e:
        # print(f"Warning: Audio extraction failed for {audio_path}: {e}")
        return np.zeros((target_num_frames, config.AUDIO_MFCC_DIM), dtype=np.float32)


def extract_skeleton_data(mat_data):
    """
    Extracts raw 3D joint positions from the .mat structure.
    Returns: (NumFrames, NumJoints, 3)
    """
    try:
        video = mat_data.Video
        num_frames = video.NumFrames
        frames = video.Frames

        # Initialize container
        # 20 joints, 3 coordinates
        skeleton_data = np.zeros((num_frames, config.NUM_JOINTS, 3), dtype=np.float32)

        # Check if frames is iterable
        if isinstance(frames, (np.ndarray, list)):
            iter_frames = frames
        elif isinstance(frames, scipy.io.matlab.mat_struct):
            iter_frames = [frames]
        else:
            return skeleton_data

        for i, frame in enumerate(iter_frames):
            if i >= num_frames:
                break
            if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
                wp = frame.Skeleton.WorldPosition
                # wp should be 20x3 or struct with X,Y,Z arrays
                # Based on description: "X value represents x-component..."
                # Usually in these datasets it's a struct array or matrix.
                # Let's try to handle common formats.
                try:
                    # If it's a struct with X, Y, Z fields
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # Assuming X, Y, Z are arrays of length 20
                        x = np.array(wp.X, dtype=np.float32)
                        y = np.array(wp.Y, dtype=np.float32)
                        z = np.array(wp.Z, dtype=np.float32)

                        # Stack: (20, 3)
                        if x.size == config.NUM_JOINTS:
                            skeleton_data[i, :, 0] = x
                            skeleton_data[i, :, 1] = y
                            skeleton_data[i, :, 2] = z
                    # If it's a matrix
                    elif isinstance(wp, np.ndarray) and wp.shape == (
                        config.NUM_JOINTS,
                        3,
                    ):
                        skeleton_data[i] = wp
                except:
                    pass

        return skeleton_data
    except Exception as e:
        # print(f"Warning: Skeleton extraction failed: {e}")
        # Return zeros if failed, but try to respect NumFrames if known
        nf = 0
        if (
            mat_data
            and hasattr(mat_data, "Video")
            and hasattr(mat_data.Video, "NumFrames")
        ):
            nf = mat_data.Video.NumFrames
        return np.zeros((nf, config.NUM_JOINTS, 3), dtype=np.float32)


def process_labels(labels_json, num_frames):
    """
    Converts JSON label list to frame-wise label array.
    """
    label_array = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

    if not isinstance(labels_json, list):
        return label_array

    for label in labels_json:
        start = max(0, int(label["begin"]) - 1)  # 1-based to 0-based
        end = min(num_frames, int(label["end"]))
        lid = int(label["id"])

        if start < end:
            label_array[start:end] = lid

    return label_array


def augment_and_derive_kinematics(positions, augment=True):
    """
    Applies augmentation to positions, then computes velocity and acceleration.

    Args:
        positions: (T, Joints, 3) numpy array
        augment: bool

    Returns:
        features: (T, Joints * 9) flattened vector of [Pos, Vel, Acc]
    """
    # Work on a copy
    pos = positions.copy()

    if augment:
        # 1. Rotation around Y-axis
        # Angle in radians
        angle_deg = np.random.uniform(
            -config.AUGMENT_ROTATION_RANGE, config.AUGMENT_ROTATION_RANGE
        )
        angle_rad = np.deg2rad(angle_deg)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix for Y-axis:
        # [ cos  0  sin]
        # [  0   1   0 ]
        # [-sin  0  cos]

        # Apply rotation: x' = x*cos + z*sin, z' = -x*sin + z*cos
        x = pos[:, :, 0]
        z = pos[:, :, 2]

        new_x = x * cos_a + z * sin_a
        new_z = -x * sin_a + z * cos_a

        pos[:, :, 0] = new_x
        pos[:, :, 2] = new_z

        # 2. Scaling
        scale = 1.0 + np.random.uniform(
            -config.AUGMENT_SCALE_RANGE, config.AUGMENT_SCALE_RANGE
        )
        pos = pos * scale

    # Derive Kinematics
    # Velocity: P[t] - P[t-1]
    # Pad first frame with 0
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]

    # Acceleration: V[t] - V[t-1]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]

    # Concatenate features: (T, Joints, 9)
    # Structure: [Pos_x, Pos_y, Pos_z, Vel_x, ..., Acc_z] per joint
    features = np.concatenate([pos, vel, acc], axis=2)

    # Flatten joints and channels: (T, Joints * 9)
    T, J, C = features.shape
    features = features.reshape(T, J * C)

    return features.astype(np.float32)


# ==========================================
# Data Loading & Caching Logic
# ==========================================


def load_and_process_split(metadata_path, cache_name, load_cached_data=True):
    """
    Loads raw data for a split, processes it into arrays, and caches it.
    Returns:
        all_skeletons: (TotalFrames, J, 3)
        all_audio: (TotalFrames, MFCC_Dim)
        all_labels: (TotalFrames,)
        sequence_index: (NumSequences, 3) -> [start_idx, length, sample_id_hash/idx]
        sample_ids: List of sample ID strings
    """
    cache_path = os.path.join(config.CACHE_DIR, f"{cache_name}.npz")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(
                cache_path, allow_pickle=True
            )  # allow_pickle needed for object arrays (sample_ids)
            return (
                data["skeletons"],
                data["audio"],
                data["labels"],
                data["indices"],
                data["sample_ids"].tolist(),
            )
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Parse labels column
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    skeletons_list = []
    audio_list = []
    labels_list = []
    indices = []
    sample_ids = []

    current_idx = 0

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # Load Mat
        mat = load_mat_file(data_path)
        if mat is None:
            continue

        # Extract Skeleton
        skel = extract_skeleton_data(mat)
        num_frames = skel.shape[0]

        if num_frames == 0:
            continue

        # Extract Audio
        aud = extract_audio_features(audio_path, num_frames)

        # Generate Labels
        lbl = process_labels(row["parsed_labels"], num_frames)

        # Store
        skeletons_list.append(skel)
        audio_list.append(aud)
        labels_list.append(lbl)
        sample_ids.append(sample_id)

        indices.append([current_idx, num_frames])
        current_idx += num_frames

    # Concatenate
    if not skeletons_list:
        print("Warning: No data found!")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    all_skeletons = np.concatenate(skeletons_list, axis=0)
    all_audio = np.concatenate(audio_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    sequence_index = np.array(indices, dtype=np.int64)

    # Save to cache
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(
        cache_path,
        skeletons=all_skeletons,
        audio=all_audio,
        labels=all_labels,
        indices=sequence_index,
        sample_ids=np.array(sample_ids),
    )

    return all_skeletons, all_audio, all_labels, sequence_index, sample_ids


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, split_name, mode="train", load_cached=True, debug_size=None
    ):
        """
        Args:
            metadata_path: Path to CSV.
            split_name: 'train', 'val', or 'test'.
            mode: 'train' (sliding windows) or 'inference' (full sequences).
            load_cached: Bool.
            debug_size: Int, limit number of sequences.
        """
        self.mode = mode
        self.split_name = split_name

        # Load Raw Data
        self.skeletons, self.audio, self.labels, self.indices, self.sample_ids = (
            load_and_process_split(metadata_path, f"dataset_{split_name}", load_cached)
        )

        # Debugging
        if debug_size is not None and len(self.indices) > 0:
            limit = min(debug_size, len(self.indices))
            self.indices = self.indices[:limit]
            self.sample_ids = self.sample_ids[:limit]
            # Note: We don't slice the big arrays to save memory logic complexity, just restrict indices

        # Build Access Index
        self.access_list = []  # List of (seq_idx, start_frame)

        if self.mode == "train":
            # Create sliding windows
            for i, (start_global, length) in enumerate(self.indices):
                # If sequence is shorter than window, pad or take what we can?
                # We'll skip very short ones or pad in getitem.
                # Ideally, we want valid windows.
                if length < config.WINDOW_SIZE:
                    # For training, maybe skip or just take one padded window
                    self.access_list.append((i, 0))
                else:
                    # Stride
                    for t in range(0, length - config.WINDOW_SIZE + 1, config.STRIDE):
                        self.access_list.append((i, t))
        else:
            # Inference: One entry per sequence
            for i in range(len(self.indices)):
                self.access_list.append((i, 0))

    def __len__(self):
        return len(self.access_list)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.access_list[idx]
        global_start, seq_len = self.indices[seq_idx]

        # Determine slice range
        if self.mode == "train":
            # Fixed window size
            end_frame = start_frame + config.WINDOW_SIZE

            # Handle short sequences (padding)
            if seq_len < config.WINDOW_SIZE:
                pad_len = config.WINDOW_SIZE - seq_len
                sl_start = global_start
                sl_end = global_start + seq_len

                # Extract raw
                raw_skel = self.skeletons[sl_start:sl_end]
                raw_audio = self.audio[sl_start:sl_end]
                lbl = self.labels[sl_start:sl_end]

                # Pad
                # Pad skeleton with last frame or zeros? Zeros is safer for masking, but last frame preserves continuity.
                # Let's use zero padding for features, ignore labels.
                raw_skel = np.pad(raw_skel, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
                raw_audio = np.pad(raw_audio, ((0, pad_len), (0, 0)), mode="constant")
                lbl = np.pad(
                    lbl,
                    (0, pad_len),
                    mode="constant",
                    constant_values=config.BACKGROUND_CLASS_ID,
                )

            else:
                sl_start = global_start + start_frame
                sl_end = global_start + end_frame

                raw_skel = self.skeletons[sl_start:sl_end]
                raw_audio = self.audio[sl_start:sl_end]
                lbl = self.labels[sl_start:sl_end]

            # Augment Skeleton & Derive Kinematics
            # Only augment if training
            skel_features = augment_and_derive_kinematics(raw_skel, augment=True)

        else:
            # Inference: Full sequence
            sl_start = global_start
            sl_end = global_start + seq_len

            raw_skel = self.skeletons[sl_start:sl_end]
            raw_audio = self.audio[sl_start:sl_end]
            lbl = self.labels[sl_start:sl_end]

            # No augmentation for inference
            skel_features = augment_and_derive_kinematics(raw_skel, augment=False)

        # Concatenate Audio (Early Fusion)
        # skel_features: (T, 180), raw_audio: (T, 13)
        features = np.concatenate([skel_features, raw_audio], axis=1)  # (T, 193)

        # Convert to Tensor
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(lbl).long()

        if self.mode == "inference":
            return features, labels, self.sample_ids[seq_idx]
        else:
            return features, labels


def get_dataloaders(load_cached=True):
    """
    Factory function to create dataloaders.
    """
    # Train Set
    train_ds = GestureDataset(
        config.TRAIN_METADATA_PATH,
        "train",
        mode="train",
        load_cached=load_cached,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set
    val_ds = GestureDataset(
        config.VAL_METADATA_PATH,
        "val",
        mode="inference",  # Evaluate on full sequences
        load_cached=load_cached,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    # Batch size 1 for variable length inference
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # Test Set
    test_ds = GestureDataset(
        config.TEST_METADATA_PATH,
        "test",
        mode="inference",
        load_cached=load_cached,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
