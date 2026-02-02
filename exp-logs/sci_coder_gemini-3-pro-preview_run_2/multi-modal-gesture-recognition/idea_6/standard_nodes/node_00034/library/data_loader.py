import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import Config

# Ensure reproducible behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def load_mat_file(file_path):
    """Safely loads a .mat file."""
    try:
        return scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_audio_features(audio_path, target_num_frames):
    """
    Extracts MFCC features from audio and aligns them to the video frame count.

    Args:
        audio_path (str): Path to the .wav file.
        target_num_frames (int): Number of video frames to align to.

    Returns:
        np.ndarray: MFCC features of shape (target_num_frames, n_mfcc).
    """
    if not os.path.exists(audio_path):
        # Return zeros if audio is missing
        return np.zeros((target_num_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_MFCC_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # Shape: (time, n_mfcc)

        # Interpolate to match video frames
        current_frames = mfcc.shape[0]
        if current_frames != target_num_frames:
            # Input needs to be (Batch, Channels, Time) for interpolate
            mfcc_t = mfcc.transpose(0, 1).unsqueeze(0)
            mfcc_interp = F.interpolate(
                mfcc_t, size=target_num_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc_interp.squeeze(0).transpose(0, 1)

        return mfcc.numpy()

    except Exception as e:
        # Fallback to zeros on error
        return np.zeros((target_num_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)


def extract_skeleton_features(mat_data, num_frames):
    """
    Extracts normalized skeleton features and velocity.

    Returns:
        np.ndarray: Features of shape (num_frames, num_joints*3 * 2).
    """
    if mat_data is None or "Video" not in mat_data:
        return np.zeros((num_frames, Config.NUM_JOINTS * 6), dtype=np.float32)

    video = mat_data["Video"]
    frames = getattr(video, "Frames", [])

    # Handle cases where Frames might be missing or not an array
    if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
        # Single frame or empty
        if hasattr(frames, "Skeleton"):
            frames = [frames]
        else:
            frames = []

    # Pre-allocate
    # 12 joints * 3 coords = 36
    skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    # Indices for selected joints
    joint_indices = Config.SELECTED_JOINTS

    # Process frames
    limit = min(num_frames, len(frames))
    for i in range(limit):
        frame_obj = frames[i]
        if hasattr(frame_obj, "Skeleton"):
            skel = frame_obj.Skeleton
            if hasattr(skel, "WorldPosition"):
                wp = skel.WorldPosition
                # wp is usually 20 structs or objects.
                # We need to extract X, Y, Z for specific indices.
                # Assuming wp is an array of structs corresponding to joints

                # Check if WorldPosition is an array of objects/structs
                if isinstance(wp, np.ndarray) and len(wp) >= 20:
                    for j_idx, joint_id in enumerate(joint_indices):
                        joint = wp[joint_id]
                        if hasattr(joint, "X"):
                            skeleton_data[i, j_idx, 0] = joint.X
                            skeleton_data[i, j_idx, 1] = joint.Y
                            skeleton_data[i, j_idx, 2] = joint.Z

    # Fill missing frames with previous frame data (simple imputation)
    for i in range(1, num_frames):
        if np.all(skeleton_data[i] == 0):
            skeleton_data[i] = skeleton_data[i - 1]

    # Normalization: Subtract HipCenter (Index 0 in SELECTED_JOINTS is HipCenter, which is index 0 in raw)
    # Config.SELECTED_JOINTS[0] is 0 (HipCenter).
    # So in our reduced array, index 0 is HipCenter.
    hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
    normalized_data = skeleton_data - hip_center

    # Flatten joints: (T, 12*3)
    flat_pos = normalized_data.reshape(num_frames, -1)

    # Compute Velocity
    velocity = np.zeros_like(flat_pos)
    velocity[1:] = flat_pos[1:] - flat_pos[:-1]

    # Concatenate Position and Velocity
    features = np.concatenate([flat_pos, velocity], axis=1)  # (T, 72)

    return features


def extract_labels(mat_data, num_frames):
    """
    Constructs dense frame-wise labels from annotation.

    Returns:
        np.ndarray: Label array of shape (num_frames,).
    """
    labels = np.zeros(num_frames, dtype=np.int64)  # 0 is background

    if mat_data is None or "Video" not in mat_data:
        return labels

    video = mat_data["Video"]
    if not hasattr(video, "Labels"):
        return labels

    raw_labels = video.Labels

    # Helper to process single label entry
    def process_entry(entry):
        try:
            name = entry.Name
            start = int(entry.Begin) - 1  # Matlab 1-based to Python 0-based
            end = int(
                entry.End
            )  # Exclusive in Python slice? No, inclusive in logic, so end

            if name in Config.GESTURE_MAP:
                gid = Config.GESTURE_MAP[name]
                # Clip to valid range
                start = max(0, start)
                end = min(num_frames, end)
                if start < end:
                    labels[start:end] = gid
        except AttributeError:
            pass

    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 0:
            process_entry(raw_labels.item())
        else:
            for l in raw_labels:
                process_entry(l)
    else:
        process_entry(raw_labels)

    return labels


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Loads, processes, and caches the dataset.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return list(data["features"]), list(data["labels"]), list(data["ids"])
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Sanitize missing values to avoid TypeErrors in path construction
    # Cite debug_lesson_9
    df = df.fillna("")

    # Debug subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    all_features = []
    all_labels = []
    all_ids = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]

        # Ensure paths are valid strings
        if not row["data_path"] or not row["audio_path"]:
            continue

        # Construct full paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Load MAT
        mat_data = load_mat_file(mat_path)

        # Determine num_frames
        if mat_data and "Video" in mat_data:
            num_frames = getattr(mat_data["Video"], "NumFrames", 0)
        else:
            # Fallback if MAT fails (shouldn't happen often)
            num_frames = row["num_frames"]

        if num_frames == 0:
            # Skip invalid
            continue

        # Extract Features
        skel_feats = extract_skeleton_features(mat_data, num_frames)
        audio_feats = extract_audio_features(audio_path, num_frames)

        # Concatenate: (T, 72) + (T, 13) -> (T, 85)
        combined_feats = np.concatenate([skel_feats, audio_feats], axis=1)

        # Extract Labels
        # For test set, this will just return zeros, which is fine
        dense_labels = extract_labels(mat_data, num_frames)

        all_features.append(combined_feats.astype(np.float32))
        all_labels.append(dense_labels.astype(np.int64))
        all_ids.append(sample_id)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=np.array(all_features, dtype=object),
        labels=np.array(all_labels, dtype=object),
        ids=np.array(all_ids, dtype=object),
    )
    print(f"Saved processed data to {cache_path}")

    return all_features, all_labels, all_ids


class GestureDataset(Dataset):
    def __init__(self, features, labels, is_train=False):
        self.features = features
        self.labels = labels
        self.is_train = is_train
        self.noise_std = Config.GAUSSIAN_NOISE_STD

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # (T, D)
        lbl = self.labels[idx]  # (T,)

        # Augmentation
        if self.is_train:
            noise = np.random.normal(0, self.noise_std, feat.shape).astype(np.float32)
            feat = feat + noise

        return torch.from_numpy(feat), torch.from_numpy(lbl)


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    """
    features, labels = zip(*batch)

    # Store original lengths for masking/packing if needed
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)

    # Pad features with 0
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Pad labels with -100 (standard ignore index for CrossEntropy)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return padded_features, padded_labels, lengths


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    Config.ensure_dirs()

    # Load Data
    train_feats, train_lbls, _ = process_dataset(
        Config.TRAIN_METADATA, "train_data.npz", load_cached_data
    )
    val_feats, val_lbls, _ = process_dataset(
        Config.VAL_METADATA, "val_data.npz", load_cached_data
    )
    test_feats, test_lbls, test_ids = process_dataset(
        Config.TEST_METADATA, "test_data.npz", load_cached_data
    )

    # Create Datasets
    train_ds = GestureDataset(train_feats, train_lbls, is_train=True)
    val_ds = GestureDataset(val_feats, val_lbls, is_train=False)
    test_ds = GestureDataset(test_feats, test_lbls, is_train=False)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
