import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def load_mat_file(path):
    """Safely loads a .mat file."""
    try:
        # struct_as_record=False and squeeze_me=True allow accessing fields as attributes
        return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading MAT file {path}: {e}")
        return None


def process_audio(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCC features, and aligns them to the video frame count.
    """
    full_path = os.path.join(Config.INPUT_DIR, audio_path)
    if not os.path.exists(full_path) or target_num_frames <= 0:
        return np.zeros(
            (max(1, target_num_frames), Config.AUDIO_MFCC_N_MFCC), dtype=np.float32
        )

    try:
        waveform, sample_rate = torchaudio.load(full_path)

        # Extract MFCCs
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_MFCC_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)

        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        mfcc = F.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()
        return mfcc.astype(np.float32)

    except Exception as e:
        return np.zeros(
            (max(1, target_num_frames), Config.AUDIO_MFCC_N_MFCC), dtype=np.float32
        )


def process_skeleton(mat_data, num_frames):
    """
    Extracts, normalizes, and selects skeleton joints.
    """
    skeleton_data = np.zeros(
        (num_frames, Config.NUM_SELECTED_JOINTS, 3), dtype=np.float32
    )

    if mat_data is None or not hasattr(mat_data, "Video"):
        return skeleton_data

    video = mat_data.Video
    frames = getattr(video, "Frames", [])

    if isinstance(frames, np.ndarray) and frames.size == 0:
        return skeleton_data

    limit = min(num_frames, len(frames) if isinstance(frames, np.ndarray) else 1)

    for i in range(limit):
        frame_obj = frames[i] if isinstance(frames, np.ndarray) else frames

        if not hasattr(frame_obj, "Skeleton") or frame_obj.Skeleton is None:
            if i > 0:
                skeleton_data[i] = skeleton_data[i - 1]
            continue

        skel = frame_obj.Skeleton
        target_skel = None
        if isinstance(skel, np.ndarray):
            if skel.size > 0:
                target_skel = skel[0]
        else:
            target_skel = skel

        if target_skel is None or not hasattr(target_skel, "WorldPosition"):
            if i > 0:
                skeleton_data[i] = skeleton_data[i - 1]
            continue

        try:
            wp = target_skel.WorldPosition

            # Robust extraction
            current_frame_joints = np.zeros(
                (Config.NUM_SELECTED_JOINTS, 3), dtype=np.float32
            )

            # Check if wp is iterable/indexable
            if hasattr(wp, "__len__") and len(wp) >= 20:
                for idx, joint_idx in enumerate(Config.SELECTED_JOINTS):
                    joint_node = wp[joint_idx]
                    current_frame_joints[idx, 0] = joint_node.X
                    current_frame_joints[idx, 1] = joint_node.Y
                    current_frame_joints[idx, 2] = joint_node.Z
            else:
                if i > 0:
                    skeleton_data[i] = skeleton_data[i - 1]
                continue

            hip_center = current_frame_joints[0].copy()
            current_frame_joints -= hip_center
            current_frame_joints *= Config.SCALE_FACTOR
            skeleton_data[i] = current_frame_joints

        except Exception:
            if i > 0:
                skeleton_data[i] = skeleton_data[i - 1]

    for i in range(limit, num_frames):
        if i > 0:
            skeleton_data[i] = skeleton_data[i - 1]

    return skeleton_data


def generate_frame_labels(mat_data, num_frames, label_seq):
    """
    Generates frame-wise classification labels and boundary labels.

    Args:
        mat_data: Loaded MAT structure.
        num_frames (int): Total frames.
        label_seq (list): List of gesture IDs from metadata.

    Returns:
        tuple: (labels, boundaries)
            labels: (num_frames,) int array
            boundaries: (num_frames,) float array
    """
    labels = np.zeros(num_frames, dtype=np.int64)  # 0 is background
    boundaries = np.zeros(num_frames, dtype=np.float32)

    if mat_data is None or not hasattr(mat_data, "Video"):
        return labels, boundaries

    video = mat_data.Video
    raw_labels = getattr(video, "Labels", [])

    # Helper to process a label object
    def process_label_obj(obj):
        try:
            name = obj.Name
            start = int(obj.Begin) - 1  # 1-based to 0-based
            end = int(obj.End) - 1

            if name in Config.GESTURE_MAP:
                gid = Config.GESTURE_MAP[name]

                # Clip to valid range
                start = max(0, start)
                end = min(num_frames - 1, end)

                if start <= end:
                    labels[start : end + 1] = gid
                    # Set boundaries at transition points
                    boundaries[start] = 1.0
                    if end + 1 < num_frames:
                        boundaries[end + 1] = 1.0
        except AttributeError:
            pass

    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 0:
            process_label_obj(raw_labels.item())
        else:
            for l in raw_labels:
                process_label_obj(l)
    else:
        process_label_obj(raw_labels)

    # Refine boundaries: 1 where label changes
    # This captures internal transitions if gestures are adjacent
    # and ensures background-to-gesture transitions are marked
    diff = np.diff(labels, prepend=labels[0])
    boundaries = (diff != 0).astype(np.float32)
    boundaries[0] = 1.0  # Start of sequence is always a boundary

    return labels, boundaries


def process_dataset(metadata_path, is_train=True):
    df = pd.read_csv(metadata_path)

    # Cite debug_lesson_15: Drop rows with missing critical paths
    # We rely on generate_metadata to produce valid paths, but filter NaNs just in case.
    df = df.dropna(subset=["data_path", "audio_path", "num_frames"])

    if "labels" in df.columns:
        df["labels"] = df["labels"].apply(
            lambda x: (
                [int(i) for i in str(x).split()]
                if pd.notna(x) and str(x).strip() != ""
                else []
            )
        )

    all_positions = []
    all_audio = []
    all_labels = []
    all_boundaries = []
    all_ids = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        num_frames = int(row["num_frames"])

        # Cite debug_lesson_2: Filter invalid data explicitly
        if num_frames <= 0:
            continue

        if is_train and len(row["labels"]) == 0:
            continue

        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        mat_data = load_mat_file(mat_path)

        # Skip if MAT file is corrupt or missing
        if mat_data is None:
            continue

        positions = process_skeleton(mat_data, num_frames)
        audio = process_audio(row["audio_path"], num_frames)

        if is_train:
            gesture_list = row["labels"]
            lbls, bnds = generate_frame_labels(mat_data, num_frames, gesture_list)
        else:
            lbls = np.zeros(num_frames, dtype=np.int64)
            bnds = np.zeros(num_frames, dtype=np.float32)

        all_positions.append(positions)
        all_audio.append(audio)
        all_labels.append(lbls)
        all_boundaries.append(bnds)
        all_ids.append(sample_id)

    return {
        "positions": all_positions,
        "audio": all_audio,
        "labels": all_labels,
        "boundaries": all_boundaries,
        "ids": all_ids,
    }


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode (train, val, test).
    Uses caching to speed up subsequent loads.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Processed data dictionary.
    """
    if mode == "train":
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        cache_path = Config.TRAIN_CACHE_FILE
        is_train = True
    elif mode == "val":
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_path = Config.VAL_CACHE_FILE
        is_train = True  # Val has labels
    else:
        csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
        cache_path = Config.TEST_CACHE_FILE
        is_train = False

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "positions": data["positions"],
                "audio": data["audio"],
                "labels": data["labels"],
                "boundaries": data["boundaries"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {mode} data from raw files...")
    data_dict = process_dataset(csv_path, is_train=is_train)

    # Save to cache (using object array for variable length sequences)
    np.savez_compressed(
        cache_path,
        positions=np.array(data_dict["positions"], dtype=object),
        audio=np.array(data_dict["audio"], dtype=object),
        labels=np.array(data_dict["labels"], dtype=object),
        boundaries=np.array(data_dict["boundaries"], dtype=object),
        ids=np.array(data_dict["ids"], dtype=object),
    )

    return data_dict


def physically_consistent_augmentation(positions):
    """
    Applies temporally correlated noise to positions and derives consistent velocity.

    Args:
        positions (np.ndarray): Shape (T, J, 3).

    Returns:
        tuple: (aug_positions, aug_velocities)
    """
    T, J, C = positions.shape

    # 1. Generate Gaussian Noise
    noise = np.random.normal(0, 0.005, size=(T, J, C)).astype(
        np.float32
    )  # Small noise in meters

    # 2. Temporal Low-Pass Filter (Moving Average)
    # Simple window-3 smoothing along time axis
    # Pad noise to keep shape
    padded_noise = np.pad(noise, ((1, 1), (0, 0), (0, 0)), mode="edge")
    smoothed_noise = np.zeros_like(noise)
    for t in range(T):
        smoothed_noise[t] = np.mean(padded_noise[t : t + 3], axis=0)

    # 3. Add to positions
    aug_positions = positions + smoothed_noise

    # 4. Derive Velocity
    # V_t = P_t - P_{t-1}
    # Pad first frame with 0 velocity (or replicate first frame pos difference which is 0)
    aug_velocities = np.zeros_like(aug_positions)
    aug_velocities[1:] = aug_positions[1:] - aug_positions[:-1]

    return aug_positions, aug_velocities


class GestureDataset(Dataset):
    def __init__(self, data_dict, augment=False):
        """
        Args:
            data_dict (dict): Dictionary containing positions, audio, labels, etc.
            augment (bool): Whether to apply augmentation.
        """
        self.positions = data_dict["positions"]
        self.audio = data_dict["audio"]
        self.labels = data_dict["labels"]
        self.boundaries = data_dict["boundaries"]
        self.ids = data_dict["ids"]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get sequences
        pos = self.positions[idx]  # (T, 12, 3)
        aud = self.audio[idx]  # (T, 13)
        lbl = self.labels[idx]  # (T,)
        bnd = self.boundaries[idx]  # (T,)

        # Ensure float32
        pos = pos.astype(np.float32)
        aud = aud.astype(np.float32)

        if self.augment:
            # Apply physically consistent augmentation
            aug_pos, aug_vel = physically_consistent_augmentation(pos)
        else:
            aug_pos = pos
            # Compute clean velocity
            aug_vel = np.zeros_like(pos)
            aug_vel[1:] = pos[1:] - pos[:-1]

        # Flatten spatial dimensions for model input
        # Pos: (T, 12, 3) -> (T, 36)
        # Vel: (T, 12, 3) -> (T, 36)
        T = aug_pos.shape[0]
        flat_pos = aug_pos.reshape(T, -1)
        flat_vel = aug_vel.reshape(T, -1)

        # Concatenate features: [Pos, Vel, Audio]
        # Dim: 36 + 36 + 13 = 85
        features = np.concatenate([flat_pos, flat_vel, aud], axis=1)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "boundaries": torch.tensor(bnd, dtype=torch.float32),
            "id": self.ids[idx],
        }


def collate_fn(batch):
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)
    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    boundaries = [x["boundaries"] for x in batch]
    ids = [x["id"] for x in batch]

    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    max_len = lengths.max().item() if lengths.numel() > 0 else 0

    if max_len == 0:
        # Handle empty batch case to prevent crash
        return {
            "features": torch.zeros(
                len(batch), 1, features[0].shape[1] if features else 85
            ),
            "labels": torch.zeros(len(batch), 1, dtype=torch.long),
            "boundaries": torch.zeros(len(batch), 1, dtype=torch.float32),
            "mask": torch.zeros(len(batch), 1, dtype=torch.float32),
            "lengths": torch.ones(len(batch), dtype=torch.long),
            "ids": ids,
        }

    padded_features = torch.zeros(len(batch), max_len, features[0].shape[1])
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_boundaries = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, (f, l, b, length) in enumerate(zip(features, labels, boundaries, lengths)):
        padded_features[i, :length] = f
        padded_labels[i, :length] = l
        padded_boundaries[i, :length] = b
        mask[i, :length] = 1.0

    return {
        "features": padded_features,
        "labels": padded_labels,
        "boundaries": padded_boundaries,
        "mask": mask,
        "lengths": lengths,
        "ids": ids,
    }
