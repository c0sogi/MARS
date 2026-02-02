import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


def load_mat_file(path):
    """
    Robustly loads the .mat file containing skeleton and label data.
    """
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_skeleton_data(mat, num_frames):
    """
    Extracts raw world positions for the selected upper-body joints.
    Returns: numpy array of shape (num_frames, num_joints, 3)
    """
    if "Video" not in mat:
        return np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    video = mat["Video"]

    # Handle Frames structure
    if not hasattr(video, "Frames"):
        return np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    frames = video.Frames

    # Initialize container
    # We use the actual length of frames found in the struct, truncated/padded to num_frames if mismatch
    actual_frames = len(frames) if isinstance(frames, np.ndarray) else 1
    if actual_frames == 0:
        return np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    # Pre-allocate
    skeleton_data = np.zeros((actual_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    # Indices of joints to keep
    indices = Config.SELECTED_JOINTS_INDICES

    # Helper to extract position from a skeleton object
    def get_pos(skel_obj):
        # skel_obj should be a Skeleton structure
        # We need WorldPosition
        if not hasattr(skel_obj, "Skeleton"):
            return np.zeros((Config.NUM_JOINTS, 3))

        joints = skel_obj.Skeleton

        # joints might be an array of structures or a single structure
        # The dataset description says "Skeleton" contains "WorldPosition"
        # Usually joints is an array of size 20 (for 20 joints)

        positions = []
        for idx in indices:
            if isinstance(joints, np.ndarray) and idx < len(joints):
                j = joints[idx]
                if hasattr(j, "WorldPosition"):
                    wp = j.WorldPosition
                    positions.append([wp.X, wp.Y, wp.Z])
                else:
                    positions.append([0.0, 0.0, 0.0])
            else:
                positions.append([0.0, 0.0, 0.0])
        return np.array(positions)

    # Iterate over frames
    if isinstance(frames, np.ndarray):
        for i, frame_obj in enumerate(frames):
            # frame_obj contains 'Skeleton' field which might be the user
            # The structure is Video.Frames(i).Skeleton.Skeleton(j).WorldPosition
            # Wait, the description says: "Skeleton Frame: An array of Skeleton structures... contained within a Skeletons array"
            # But the export script says: "Sample00001_X.mat ... containing ... Skeleton."
            # The provided mat file is the aggregated one.

            # Let's try to access the skeleton for the tracked user.
            # Usually there is a 'Skeleton' field in the frame object.
            if hasattr(frame_obj, "Skeleton"):
                # There might be multiple skeletons, we need the valid one.
                # Simplification: Take the first one or the one with non-zero data.
                # In this dataset, usually single user or we trust the first valid one.
                skel = frame_obj.Skeleton

                # If skel is an array (multiple users), take the first one
                if isinstance(skel, np.ndarray) and len(skel) > 0:
                    skel = skel[0]  # Assume primary user is 0

                # Now extract joints from this skeleton
                # This 'skel' object should have 'WorldPosition' fields for joints?
                # No, the structure is usually: skel.Joints or similar.
                # Re-reading description: "Skeleton structure ... contains JointsType, WorldPosition..."
                # It seems 'skel' IS the array of joints if it's a struct array?
                # Or 'skel' contains a field 'Joints'?

                # Based on standard MSR Daily Activity / similar datasets:
                # Video.Frames(i).Skeleton is the skeleton info.
                # If it's a struct array of size 20, then it is the joints.

                if isinstance(skel, np.ndarray) and len(skel) >= 20:
                    # This looks like the joints array
                    current_pose = []
                    for idx in indices:
                        joint = skel[idx]
                        if hasattr(joint, "WorldPosition"):
                            wp = joint.WorldPosition
                            current_pose.append([wp.X, wp.Y, wp.Z])
                        else:
                            current_pose.append([0.0, 0.0, 0.0])
                    skeleton_data[i] = np.array(current_pose)
                else:
                    pass  # Unknown structure

    # Handle length mismatch with num_frames
    if actual_frames != num_frames:
        if actual_frames > num_frames:
            skeleton_data = skeleton_data[:num_frames]
        else:
            padding = np.zeros(
                (num_frames - actual_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )
            skeleton_data = np.concatenate([skeleton_data, padding], axis=0)

    return skeleton_data


def extract_framewise_labels(mat, num_frames):
    """
    Constructs the frame-wise label vector from the .mat annotations.
    Returns: numpy array of shape (num_frames,)
    """
    labels = np.zeros(num_frames, dtype=np.int64)

    if "Video" not in mat:
        return labels
    video = mat["Video"]

    if not hasattr(video, "Labels"):
        return labels

    raw_labels = video.Labels

    def process_label(lbl_obj):
        try:
            name = lbl_obj.Name
            start = int(lbl_obj.Begin) - 1  # Matlab 1-based
            end = int(
                lbl_obj.End
            )  # Exclusive in Python slicing? No, usually inclusive in Matlab.
            # Let's assume End is inclusive in Matlab, so in Python slice it's End.

            if name in Config.GESTURE_MAP:
                gid = Config.GESTURE_MAP[name]
                # Clip to bounds
                start = max(0, start)
                end = min(num_frames, end)
                if start < end:
                    labels[start:end] = gid
        except AttributeError:
            pass

    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 0:
            process_label(raw_labels.item())
        else:
            for l in raw_labels:
                process_label(l)
    else:
        process_label(raw_labels)

    return labels


def load_and_process_audio(audio_path, target_frames):
    """
    Loads audio, extracts MFCCs, and aligns to video frames.
    Returns: numpy array of shape (target_frames, n_mfcc)
    """
    if not os.path.exists(audio_path):
        return np.zeros((target_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_MFCC_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, Time)
        mfcc = mfcc.mean(dim=0)  # Average over channels: (n_mfcc, Time)

        # Align to video frames via interpolation
        # Input to interpolate needs to be (Batch, Channels, Length)
        mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, Time)

        aligned_mfcc = F.interpolate(
            mfcc, size=target_frames, mode="linear", align_corners=False
        )  # (1, n_mfcc, target_frames)

        aligned_mfcc = aligned_mfcc.squeeze(0).transpose(
            0, 1
        )  # (target_frames, n_mfcc)
        return aligned_mfcc.numpy()

    except Exception as e:
        # print(f"Audio error {audio_path}: {e}")
        return np.zeros((target_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)


def process_dataset_to_memory(df, cache_path):
    """
    Iterates through the dataframe, loads raw data, and aggregates into a dictionary.
    """
    data_cache = {
        "sample_ids": [],
        "skeletons": [],  # List of (T, J, 3) arrays
        "audios": [],  # List of (T, C) arrays
        "labels": [],  # List of (T,) arrays
        "num_frames": [],
    }

    print(f"Processing {len(df)} samples...")

    for _, row in df.iterrows():
        sid = row["sample_id"]
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        n_frames = int(row["num_frames"])

        if n_frames == 0:
            # Fallback if metadata was 0 (shouldn't happen often)
            n_frames = 50

        # Load Mat
        mat = load_mat_file(mat_path)
        if mat is None:
            continue

        # Extract Skeleton
        skel = extract_skeleton_data(mat, n_frames)

        # Extract Labels (if available)
        lbls = extract_framewise_labels(mat, n_frames)

        # Extract Audio
        aud = load_and_process_audio(audio_path, n_frames)

        data_cache["sample_ids"].append(sid)
        data_cache["skeletons"].append(skel)
        data_cache["audios"].append(aud)
        data_cache["labels"].append(lbls)
        data_cache["num_frames"].append(n_frames)

    # Convert lists to object arrays for saving
    # We use object arrays because lengths differ
    np.savez_compressed(
        cache_path,
        sample_ids=np.array(data_cache["sample_ids"]),
        skeletons=np.array(data_cache["skeletons"], dtype=object),
        audios=np.array(data_cache["audios"], dtype=object),
        labels=np.array(data_cache["labels"], dtype=object),
        num_frames=np.array(data_cache["num_frames"]),
    )
    print(f"Saved cache to {cache_path}")
    return data_cache


def load_cached_dataset(cache_path):
    print(f"Loading cache from {cache_path}...")
    loaded = np.load(cache_path, allow_pickle=True)
    data = {
        "sample_ids": loaded["sample_ids"],
        "skeletons": loaded["skeletons"],
        "audios": loaded["audios"],
        "labels": loaded["labels"],
        "num_frames": loaded["num_frames"],
    }
    return data


class GestureDataset(Dataset):
    def __init__(self, data_dict, is_train=True):
        self.sample_ids = data_dict["sample_ids"]
        self.skeletons = data_dict["skeletons"]
        self.audios = data_dict["audios"]
        self.labels = data_dict["labels"]
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_ids)

    def _augment_skeleton(self, skeleton):
        """
        Applies physically consistent geometric augmentation.
        skeleton: (T, J, 3)
        """
        T, J, C = skeleton.shape

        # 1. Random Rotation (around Y axis - vertical, or general)
        # Let's do a general small rotation to simulate camera angle changes
        # Euler angles in degrees
        rx = np.random.uniform(-Config.AUG_ROTATION_RANGE, Config.AUG_ROTATION_RANGE)
        ry = np.random.uniform(-Config.AUG_ROTATION_RANGE, Config.AUG_ROTATION_RANGE)
        rz = np.random.uniform(-Config.AUG_ROTATION_RANGE, Config.AUG_ROTATION_RANGE)

        # Convert to radians
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

        # Rotation matrices
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )
        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )
        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Apply rotation: (T*J, 3) @ R.T -> (T*J, 3)
        flat_skel = skeleton.reshape(-1, 3)
        rotated_skel = flat_skel @ R.T

        # 2. Random Scaling
        scale = 1.0 + np.random.uniform(-Config.AUG_SCALE_RANGE, Config.AUG_SCALE_RANGE)
        augmented_skel = rotated_skel * scale

        return augmented_skel.reshape(T, J, 3)

    def __getitem__(self, idx):
        # Load raw data
        skel = self.skeletons[idx].astype(np.float32)  # (T, J, 3)
        audio = self.audios[idx].astype(np.float32)  # (T, MFCC)
        label = self.labels[idx].astype(np.int64)  # (T,)

        # Augmentation
        if self.is_train:
            skel = self._augment_skeleton(skel)

        # Feature Engineering
        # 1. Velocity (Physically Consistent: computed AFTER augmentation)
        # Pad first frame with 0
        velocity = np.zeros_like(skel)
        velocity[1:] = skel[1:] - skel[:-1]

        # 2. Normalization (Root-Relative)
        # HipCenter is usually index 0 in our selection (if it maps to HipCenter)
        # Config.SELECTED_JOINTS_INDICES = [0, 1, ..., 11]
        # Dataset description: 0:HipCenter.
        root = skel[:, 0:1, :]  # (T, 1, 3)
        skel_norm = skel - root

        # Flatten Spatial Dimensions
        # Skel: (T, J*3), Vel: (T, J*3)
        T = skel.shape[0]
        skel_flat = skel_norm.reshape(T, -1)
        vel_flat = velocity.reshape(T, -1)

        # Concatenate all features
        # [Skeleton, Velocity, Audio]
        features = np.concatenate([skel_flat, vel_flat, audio], axis=1)  # (T, 85)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(T, dtype=torch.long),
            "sample_id": self.sample_ids[idx],
        }


def collate_fn(batch):
    # Sort by length for packing (optional but good practice)
    batch.sort(key=lambda x: x["length"], reverse=True)

    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    lengths = torch.stack([x["length"] for x in batch])
    sample_ids = [x["sample_id"] for x in batch]

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    # Create Mask (Batch, Time)
    mask = torch.zeros(
        features_padded.shape[0], features_padded.shape[1], dtype=torch.float32
    )
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    return {
        "features": features_padded,  # (B, T_max, D)
        "labels": labels_padded,  # (B, T_max)
        "mask": mask,  # (B, T_max)
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get data loaders.
    Handles caching logic.
    """
    set_seed(Config.SEED)

    # Paths
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")

    train_cache = os.path.join(Config.CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_data.npz")

    # Load Dataframes
    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Sanitize DataFrames (Cite debug_lesson_15, debug_lesson_9)
    df_train = df_train.dropna(subset=["data_path"])
    df_val = df_val.dropna(subset=["data_path"])
    df_test = df_test.dropna(subset=["data_path"])

    # --- Train Data ---
    if load_cached_data and os.path.exists(train_cache):
        train_data = load_cached_dataset(train_cache)
    else:
        train_data = process_dataset_to_memory(df_train, train_cache)

    # --- Val Data ---
    if load_cached_data and os.path.exists(val_cache):
        val_data = load_cached_dataset(val_cache)
    else:
        val_data = process_dataset_to_memory(df_val, val_cache)

    # --- Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        test_data = load_cached_dataset(test_cache)
    else:
        test_data = process_dataset_to_memory(df_test, test_cache)

    # Create Datasets
    train_dataset = GestureDataset(train_data, is_train=True)
    val_dataset = GestureDataset(val_data, is_train=False)
    test_dataset = GestureDataset(test_data, is_train=False)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
