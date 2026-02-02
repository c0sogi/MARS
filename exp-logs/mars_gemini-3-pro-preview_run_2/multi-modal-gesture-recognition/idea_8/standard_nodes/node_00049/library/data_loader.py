import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import configuration and utilities
from library import config
from library import utils


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the Gesture Recognition task.
    Handles multi-modal feature extraction (Skeleton + Audio) and frame-wise labeling.
    """

    def __init__(self, data_list, is_train=True, transform=None):
        """
        Args:
            data_list (list): List of dictionaries containing processed samples.
            is_train (bool): Flag to enable augmentation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data_list
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Features: [Seq_Len, Input_Dim]
        features = torch.tensor(sample["features"], dtype=torch.float32)

        # Targets: [Seq_Len]
        # Test samples might not have targets, return dummy if so
        if "targets" in sample and sample["targets"] is not None:
            targets = torch.tensor(sample["targets"], dtype=torch.long)
        else:
            targets = torch.zeros(features.shape[0], dtype=torch.long)

        # Augmentation: Gaussian Noise on Skeleton Features
        # Skeleton features are the first TOTAL_SKELETON_FEATURES columns
        if self.is_train:
            noise = (
                torch.randn_like(features[:, : config.TOTAL_SKELETON_FEATURES])
                * config.GAUSSIAN_NOISE_STD
            )
            features[:, : config.TOTAL_SKELETON_FEATURES] += noise

        return {
            "id": sample["id"],
            "features": features,
            "targets": targets,
            "length": features.shape[0],
        }


def extract_skeleton_features(mat_path, num_frames):
    """
    Extracts and normalizes skeleton joint positions and velocities from .mat file.
    """
    try:
        # Load .mat file
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return np.zeros((num_frames, config.TOTAL_SKELETON_FEATURES))

        video = mat["Video"]
        frames_data = getattr(video, "Frames", None)

        # Handle cases where Frames might be missing or empty
        if frames_data is None:
            return np.zeros((num_frames, config.TOTAL_SKELETON_FEATURES))

        # Ensure frames_data is iterable (array) even if single frame
        if not isinstance(frames_data, np.ndarray):
            frames_data = np.array([frames_data])

        # Pre-allocate array: [NumFrames, NumJoints, 3]
        # We limit by num_frames in case metadata and mat mismatch slightly
        actual_frames = min(len(frames_data), num_frames)
        skeleton_data = np.zeros((actual_frames, config.NUM_SELECTED_JOINTS, 3))

        # Joint Indices
        joint_indices = config.UPPER_BODY_JOINTS
        hip_center_idx = (
            0  # In our selected list, HipCenter is at index 0 (original index 0)
        )

        for i in range(actual_frames):
            frame_obj = frames_data[i]
            # Check if Skeleton exists
            if not hasattr(frame_obj, "Skeleton") or frame_obj.Skeleton is None:
                continue

            skeleton = frame_obj.Skeleton

            # Skeleton might be an array if multiple users, we usually take the first tracked one
            # or the structure might differ. Based on description: "Skeleton Frame: An array... contained within Skeletons array"
            # But the provided struct description says "Skeleton" contains "WorldPosition".
            # We assume single user or take primary.

            # If skeleton is an array (multiple users), pick the one with non-zero data or first one
            curr_skel = None
            if isinstance(skeleton, np.ndarray):
                if len(skeleton) > 0:
                    curr_skel = skeleton[0]
            else:
                curr_skel = skeleton

            if curr_skel is None or not hasattr(curr_skel, "WorldPosition"):
                continue

            world_pos = curr_skel.WorldPosition

            # Robust dimension handling for numeric arrays (Cite solution_lesson_node_00048)
            # If data is (3, N) instead of (N, 3), transpose it.
            if isinstance(world_pos, np.ndarray) and world_pos.dtype.kind in "iuf":
                if (
                    world_pos.ndim == 2
                    and world_pos.shape[0] == 3
                    and world_pos.shape[1] >= 20
                ):
                    world_pos = world_pos.T

            # Extract coordinates for selected joints
            # If WorldPosition is an array of structs (one per joint) or a numeric matrix
            if isinstance(world_pos, np.ndarray) and len(world_pos) >= 20:
                for local_idx, joint_idx in enumerate(joint_indices):
                    # joint_idx is the Kinect index (0-19)
                    joint_data = world_pos[joint_idx]
                    # Check if it has x, y, z attributes or is an array
                    if hasattr(joint_data, "X"):
                        skeleton_data[i, local_idx, 0] = joint_data.X
                        skeleton_data[i, local_idx, 1] = joint_data.Y
                        skeleton_data[i, local_idx, 2] = joint_data.Z
                    elif (
                        isinstance(joint_data, (list, tuple, np.ndarray))
                        and len(joint_data) >= 3
                    ):
                        skeleton_data[i, local_idx, :] = joint_data[:3]
            # If WorldPosition is a single struct containing arrays for all joints (unlikely based on desc but possible)
            else:
                pass  # Fallback to zeros if format unexpected

        # Normalization: Relative to HipCenter
        # HipCenter is at index 0 in our extracted skeleton_data
        hip_positions = skeleton_data[:, 0:1, :]  # [T, 1, 3]
        skeleton_data_norm = skeleton_data - hip_positions

        # Flatten spatial features: [T, NumJoints * 3]
        spatial_features = skeleton_data_norm.reshape(actual_frames, -1)

        # Compute Velocity: V_t = P_t - P_{t-1}
        # Pad first frame with 0
        velocity = np.zeros_like(spatial_features)
        velocity[1:] = spatial_features[1:] - spatial_features[:-1]

        # Concatenate: [T, Spatial + Velocity]
        full_skeleton_features = np.concatenate([spatial_features, velocity], axis=1)

        # Pad if actual frames < num_frames
        if actual_frames < num_frames:
            padding = np.zeros(
                (num_frames - actual_frames, full_skeleton_features.shape[1])
            )
            full_skeleton_features = np.concatenate(
                [full_skeleton_features, padding], axis=0
            )

        return full_skeleton_features

    except Exception as e:
        # print(f"Error processing skeleton for {mat_path}: {e}")
        return np.zeros((num_frames, config.TOTAL_SKELETON_FEATURES))


def extract_audio_features(audio_path, num_frames):
    """
    Extracts MFCC features from audio and aligns them to video frames.
    """
    try:
        if not os.path.exists(audio_path):
            return np.zeros((num_frames, config.AUDIO_FEATURES))

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # [1, n_mfcc, time]

        # Remove channel dim
        mfcc = mfcc.squeeze(0)  # [n_mfcc, time]

        # Align to video frames using interpolation
        # Input to interpolate must be [Batch, Channels, Time]
        mfcc = mfcc.unsqueeze(0)  # [1, n_mfcc, time]

        mfcc_aligned = F.interpolate(
            mfcc, size=num_frames, mode="linear", align_corners=False
        )  # [1, n_mfcc, num_frames]

        mfcc_aligned = mfcc_aligned.squeeze(0).transpose(0, 1)  # [num_frames, n_mfcc]

        return mfcc_aligned.numpy()

    except Exception as e:
        # print(f"Error processing audio for {audio_path}: {e}")
        return np.zeros((num_frames, config.AUDIO_FEATURES))


def extract_labels(mat_path, num_frames):
    """
    Constructs frame-wise label array from .mat annotation.
    """
    targets = np.zeros(num_frames, dtype=int)  # Default 0 (Background)

    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return targets

        video = mat["Video"]
        labels_raw = getattr(video, "Labels", [])

        def process_label_obj(obj):
            try:
                name = obj.Name
                start = int(obj.Begin)
                end = int(obj.End)
                if name in config.GESTURE_MAP:
                    gid = config.GESTURE_MAP[name]
                    # Matlab 1-based indexing to Python 0-based
                    # Range is inclusive in description, so start-1 to end
                    s = max(0, start - 1)
                    e = min(num_frames, end)
                    targets[s:e] = gid
            except AttributeError:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                process_label_obj(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label_obj(l)
        else:
            process_label_obj(labels_raw)

        return targets
    except Exception as e:
        return targets


def process_dataset(metadata_df, cache_file, load_cached_data=True):
    """
    Main function to process data or load from cache.
    """
    # Try loading cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data_list = list(loaded["data"])
            print(f"Loaded {len(data_list)} samples.")
            return data_list
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print("Processing dataset from scratch...")
    data_list = []

    # Sanitize DataFrame: Remove rows with missing critical data
    metadata_df = metadata_df.dropna(subset=["data_path"])

    for _, row in metadata_df.iterrows():
        sample_id = row["sample_id"]
        num_frames = int(row["num_frames"])

        # Paths
        mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(config.INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else ""
        )

        # 1. Skeleton Features
        skel_feats = extract_skeleton_features(mat_path, num_frames)

        # 2. Audio Features
        audio_feats = extract_audio_features(audio_path, num_frames)

        # Concatenate
        # Ensure lengths match exactly (robustness)
        min_len = min(len(skel_feats), len(audio_feats))
        features = np.concatenate([skel_feats[:min_len], audio_feats[:min_len]], axis=1)

        # 3. Targets (only if labels column implies ground truth exists, i.e., not test)
        # However, for consistency, we try to extract from MAT if available.
        # Test MAT files don't have labels, so it returns zeros.
        targets = extract_labels(mat_path, num_frames)[:min_len]

        data_list.append({"id": sample_id, "features": features, "targets": targets})

    # Save to cache
    print(f"Saving processed data to {cache_file}...")
    np.savez_compressed(cache_file, data=np.array(data_list, dtype=object))

    return data_list


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    """
    ids = [item["id"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    # Pad features
    features_list = [item["features"] for item in batch]
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Pad targets
    targets_list = [item["targets"] for item in batch]
    targets_padded = pad_sequence(
        targets_list, batch_first=True, padding_value=0
    )  # 0 is background

    # Create Mask (True for valid positions, False for padding)
    # Shape: [Batch, Max_Len]
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return {
        "ids": ids,
        "features": features_padded,  # [B, T, D]
        "targets": targets_padded,  # [B, T]
        "mask": mask,  # [B, T]
        "lengths": lengths,
    }


def get_data_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for Train, Val, and Test sets.
    """
    utils.set_seed(config.SEED)

    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Cache Paths
    train_cache = os.path.join(config.CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(config.CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(config.CACHE_DIR, "test_data.npz")

    # Process Data
    print("Preparing Training Data...")
    train_data = process_dataset(train_df, train_cache, load_cached_data)

    print("Preparing Validation Data...")
    val_data = process_dataset(val_df, val_cache, load_cached_data)

    print("Preparing Test Data...")
    test_data = process_dataset(test_df, test_cache, load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(train_data, is_train=True)
    val_dataset = GestureDataset(val_data, is_train=False)  # No augmentation for val
    test_dataset = GestureDataset(test_data, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
