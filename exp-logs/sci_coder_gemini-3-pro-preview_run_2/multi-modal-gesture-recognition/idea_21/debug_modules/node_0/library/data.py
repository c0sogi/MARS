import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    SEED,
    GESTURE_MAP,
    NUM_CLASSES,
    SELECTED_JOINTS,
    SCALE_FACTOR,
    AUDIO_MFCC_N_MFCC,
    DEBUG_SUBSET_SIZE,
    BATCH_SIZE,
    DEVICE,
)
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


def load_mat_file(path):
    """Safely loads a .mat file using scipy."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def process_skeleton(mat_data, num_frames):
    """
    Extracts, normalizes, and computes velocity for skeleton data.

    Args:
        mat_data: The loaded .mat object.
        num_frames: Expected number of frames.

    Returns:
        positions: (T, NumJoints, 3) normalized positions.
        valid: Boolean indicating success.
    """
    if mat_data is None or "Video" not in mat_data.__dict__:
        return None, False

    video = mat_data.Video

    # Handle Frames structure
    if not hasattr(video, "Frames"):
        return None, False

    frames = video.Frames

    # Check if frames is a single object or array
    if not isinstance(frames, np.ndarray):
        frames = [frames]

    # If num_frames doesn't match array length, trust the array length
    actual_frames = len(frames)

    # Initialize container: T x J x 3
    num_joints = len(SELECTED_JOINTS)
    raw_positions = np.zeros((actual_frames, num_joints, 3), dtype=np.float32)

    for t, frame in enumerate(frames):
        # frame.Skeleton might be an array (multiple users) or single object
        skeletons = frame.Skeleton

        target_skel = None

        # Heuristic: Pick the first skeleton found
        if isinstance(skeletons, np.ndarray):
            if skeletons.size > 0:
                target_skel = skeletons[0]
        else:
            target_skel = skeletons

        if target_skel is None:
            continue

        # Extract joint positions
        # WorldPosition is usually a struct array or object with x,y,z
        # Based on description: WorldPosition.X, .Y, .Z

        # Check if JointsType/WorldPosition exists
        if not hasattr(target_skel, "WorldPosition"):
            continue

        w_pos = target_skel.WorldPosition

        # w_pos should be an array of structs corresponding to joints
        # or a struct of arrays.
        # Usually in Kinect mat files: w_pos is an array of size 20

        if isinstance(w_pos, np.ndarray):
            # Iterate through selected joints
            for i, joint_idx in enumerate(SELECTED_JOINTS):
                if joint_idx < len(w_pos):
                    joint = w_pos[joint_idx]
                    raw_positions[t, i, 0] = joint.X
                    raw_positions[t, i, 1] = joint.Y
                    raw_positions[t, i, 2] = joint.Z
        else:
            # Fallback if structure is different (unlikely based on description)
            pass

    # Normalization
    # 1. Scale to meters
    raw_positions = raw_positions * SCALE_FACTOR

    # 2. Center at HipCenter (Index 0 in SELECTED_JOINTS is HipCenter based on config)
    # We assume SELECTED_JOINTS[0] is the root.
    # raw_positions shape: (T, J, 3)
    hip_center = raw_positions[:, 0:1, :]  # (T, 1, 3)
    centered_positions = raw_positions - hip_center

    return centered_positions, True


def extract_audio_features(audio_path, target_num_frames):
    """
    Computes MFCCs and aligns them to the video frame count.

    Args:
        audio_path: Path to .wav file.
        target_num_frames: Number of video frames to align to.

    Returns:
        mfcc_features: (T, n_mfcc)
    """
    if not os.path.exists(audio_path):
        return np.zeros((target_num_frames, AUDIO_MFCC_N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=AUDIO_MFCC_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channel, n_mfcc, Time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # Shape: (1, n_mfcc, Time)

        # Interpolate to match video frames
        # Input to interpolate needs to be (Batch, Channels, Length)
        # We treat n_mfcc as channels
        if mfcc.shape[-1] > 0:
            mfcc_resampled = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )
            # Shape: (1, n_mfcc, T) -> (T, n_mfcc)
            mfcc_features = mfcc_resampled.squeeze(0).permute(1, 0).numpy()
        else:
            mfcc_features = np.zeros(
                (target_num_frames, AUDIO_MFCC_N_MFCC), dtype=np.float32
            )

        return mfcc_features

    except Exception as e:
        # print(f"Audio processing error {audio_path}: {e}")
        return np.zeros((target_num_frames, AUDIO_MFCC_N_MFCC), dtype=np.float32)


def augment_physically_consistent(positions):
    """
    Applies temporally correlated noise to positions and derives velocity.

    Args:
        positions: (T, J, 3)

    Returns:
        features: (T, FeatureDim) where FeatureDim = J*3 + J*3
    """
    T, J, D = positions.shape

    # 1. Generate Gaussian Noise
    noise = np.random.normal(0, 0.005, size=(T, J, D))  # 5mm std dev

    # 2. Temporal Low-Pass Filter (Smooth the noise)
    # Butterworth filter
    b, a = scipy.signal.butter(N=2, Wn=0.3, btype="low")
    smooth_noise = scipy.signal.lfilter(b, a, noise, axis=0)

    # 3. Add to positions
    aug_positions = positions + smooth_noise

    # 4. Derive Velocity
    # Pad first frame to keep shape T
    velocity = np.zeros_like(aug_positions)
    velocity[1:] = aug_positions[1:] - aug_positions[:-1]

    # Flatten features
    # (T, J, 3) -> (T, J*3)
    pos_flat = aug_positions.reshape(T, -1)
    vel_flat = velocity.reshape(T, -1)

    return np.concatenate([pos_flat, vel_flat], axis=1).astype(np.float32)


def compute_velocity_no_aug(positions):
    """Computes velocity without augmentation."""
    T = positions.shape[0]
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    pos_flat = positions.reshape(T, -1)
    vel_flat = velocity.reshape(T, -1)

    return np.concatenate([pos_flat, vel_flat], axis=1).astype(np.float32)


def generate_frame_labels(labels_list, num_frames, mat_data):
    """
    Generates frame-wise classification and boundary labels.

    Args:
        labels_list: List of gesture IDs.
        num_frames: Total frames.
        mat_data: Loaded mat file containing timing info.

    Returns:
        cls_labels: (T,) int array (0-20)
        bnd_labels: (T,) float array (0.0 or 1.0)
    """
    cls_labels = np.zeros(num_frames, dtype=np.int64)

    if mat_data is None or not hasattr(mat_data, "Video"):
        return cls_labels, np.zeros(num_frames, dtype=np.float32)

    video = mat_data.Video
    if not hasattr(video, "Labels"):
        return cls_labels, np.zeros(num_frames, dtype=np.float32)

    raw_labels = video.Labels

    # Convert to list if single object
    if not isinstance(raw_labels, np.ndarray):
        raw_labels = [raw_labels]
    elif raw_labels.ndim == 0:
        raw_labels = [raw_labels.item()]

    for lbl in raw_labels:
        try:
            name = lbl.Name
            start = int(lbl.Begin) - 1  # 1-based to 0-based
            end = int(lbl.End)  # inclusive in matlab, exclusive for python slice?
            # Usually Matlab 1:10 means 10 frames. Python 0:10.
            # Let's assume End is the index of the last frame.

            if name in GESTURE_MAP:
                gid = GESTURE_MAP[name]
                # Clip to valid range
                start = max(0, start)
                end = min(num_frames, end)

                if start < end:
                    cls_labels[start:end] = gid
        except AttributeError:
            continue

    # Boundary Labels: 1 where label changes
    bnd_labels = np.zeros(num_frames, dtype=np.float32)
    # Change from t-1 to t
    diff = cls_labels[1:] != cls_labels[:-1]
    bnd_labels[1:][diff] = 1.0
    # Also mark start of first gesture if it's not background
    if cls_labels[0] != 0:
        bnd_labels[0] = 1.0

    return cls_labels, bnd_labels


class GestureDataset(Dataset):
    def __init__(self, metadata_df, is_train=True, augment=False):
        self.metadata = metadata_df
        self.is_train = is_train
        self.augment = augment
        self.data_cache = {}

    def preload_data(self, cache_file):
        """Loads data from cache file."""
        print(f"Loading data from {cache_file}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            self.data_cache = loaded["data"].item()
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.data_cache = {}

    def save_data(self, cache_file):
        """Saves processed data to cache."""
        print(f"Saving data to {cache_file}...")
        np.savez_compressed(cache_file, data=self.data_cache)

    def process_and_cache(self, cache_path, load_cached_data=True):
        if load_cached_data and os.path.exists(cache_path):
            self.preload_data(cache_path)
            return

        print(f"Processing {len(self.metadata)} samples...")

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]

            # Paths
            data_path = os.path.join(INPUT_DIR, row["data_path"])
            audio_path = os.path.join(INPUT_DIR, row["audio_path"])

            # Load MAT
            mat_data = load_mat_file(data_path)

            # Determine NumFrames
            # Prefer metadata num_frames, fallback to mat data
            num_frames = row["num_frames"]
            if num_frames == 0 and mat_data is not None:
                try:
                    num_frames = mat_data.Video.NumFrames
                except:
                    pass

            # Process Skeleton
            positions, valid = process_skeleton(mat_data, num_frames)

            if not valid:
                # Handle corrupted/missing data by creating dummy
                # print(f"Warning: Invalid skeleton for {sample_id}")
                actual_frames = num_frames if num_frames > 0 else 100
                positions = np.zeros(
                    (actual_frames, len(SELECTED_JOINTS), 3), dtype=np.float32
                )

            actual_frames = positions.shape[0]

            # Process Audio
            audio_feats = extract_audio_features(audio_path, actual_frames)

            # Process Labels (Train/Val only)
            if (
                self.is_train or "labels" in row
            ):  # 'labels' column exists in test csv but is empty
                # For test set, labels list is empty, generate_frame_labels returns zeros
                # For train/val, we parse the MAT file again or use the list?
                # The MAT file has precise start/end frames. The CSV only has sequence order.
                # We MUST use the MAT file for frame-level ground truth.
                labels_list = row[
                    "labels"
                ]  # Not used for frame generation, but good for check
                cls_labels, bnd_labels = generate_frame_labels(
                    labels_list, actual_frames, mat_data
                )
            else:
                cls_labels = np.zeros(actual_frames, dtype=np.int64)
                bnd_labels = np.zeros(actual_frames, dtype=np.float32)

            # Store in cache
            # We store positions separately to allow dynamic augmentation
            self.data_cache[sample_id] = {
                "positions": positions,  # (T, J, 3)
                "audio": audio_feats,  # (T, MFCC)
                "cls_labels": cls_labels,
                "bnd_labels": bnd_labels,
            }

        self.save_data(cache_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]

        data = self.data_cache.get(sample_id)
        if data is None:
            # Fallback if cache miss (shouldn't happen if processed)
            return self.__getitem__(0)  # Return first item to avoid crash

        positions = data["positions"]
        audio = data["audio"]
        cls_labels = data["cls_labels"]
        bnd_labels = data["bnd_labels"]

        # Augmentation
        if self.augment:
            skel_features = augment_physically_consistent(positions)
        else:
            skel_features = compute_velocity_no_aug(positions)

        # Concatenate Skeleton + Audio
        # skel: (T, J*6), audio: (T, MFCC)
        features = np.concatenate([skel_features, audio], axis=1)

        return {
            "features": torch.from_numpy(features).float(),
            "cls_labels": torch.from_numpy(cls_labels).long(),
            "bnd_labels": torch.from_numpy(bnd_labels).float(),
            "sample_id": sample_id,
        }


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    """
    features = [b["features"] for b in batch]
    cls_labels = [b["cls_labels"] for b in batch]
    bnd_labels = [b["bnd_labels"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]

    # Pad
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    cls_labels_padded = pad_sequence(
        cls_labels, batch_first=True, padding_value=0
    )  # 0 is background
    bnd_labels_padded = pad_sequence(bnd_labels, batch_first=True, padding_value=0.0)

    # Create Mask (Batch, Time)
    lengths = torch.tensor([f.size(0) for f in features]).long()
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return {
        "features": features_padded,
        "cls_labels": cls_labels_padded,
        "bnd_labels": bnd_labels_padded,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_loaders(load_cached_data=True):
    """
    Initializes datasets and dataloaders.
    """
    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Debug Subset
    if DEBUG_SUBSET_SIZE is not None:
        train_df = train_df.head(DEBUG_SUBSET_SIZE)
        val_df = val_df.head(DEBUG_SUBSET_SIZE)
        test_df = test_df.head(DEBUG_SUBSET_SIZE)

    # Initialize Datasets
    train_ds = GestureDataset(train_df, is_train=True, augment=True)
    val_ds = GestureDataset(
        val_df, is_train=True, augment=False
    )  # Validation is deterministic
    test_ds = GestureDataset(test_df, is_train=False, augment=False)

    # Process and Cache
    # We use separate cache files for each split to avoid huge files
    train_cache = os.path.join(CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(CACHE_DIR, "test_data.npz")

    train_ds.process_and_cache(train_cache, load_cached_data)
    val_ds.process_and_cache(val_cache, load_cached_data)
    test_ds.process_and_cache(test_cache, load_cached_data)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
