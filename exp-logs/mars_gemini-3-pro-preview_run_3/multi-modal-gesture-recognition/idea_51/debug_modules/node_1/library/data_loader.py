import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


# ==========================================
# Robust Skeleton Parser
# ==========================================
class SkeletonParser:
    """
    Polymorphic parser for .mat files to extract 3D joint positions robustly.
    Handles variations in MATLAB struct formats.
    """

    @staticmethod
    def parse(mat_path):
        # Load mat file with squeeze_me to simplify arrays
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception:
            return None

        # Fix: Access dict key directly (Cite debug_lesson_2)
        if "Video" not in mat:
            return None

        video = mat["Video"]
        # Fix: Unwrap 0-d array if necessary (Cite debug_lesson_16)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        # Handle Frames structure
        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames
        # Fix: Unwrap 0-d array for frames (Cite debug_lesson_16)
        if isinstance(frames, np.ndarray) and frames.ndim == 0:
            frames = frames.item()

        # Ensure frames is iterable
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        num_frames = len(frames)
        if num_frames == 0:
            return None

        skeleton_data = []

        for f in frames:
            # Check for Skeleton attribute
            if not hasattr(f, "Skeleton"):
                skeleton_data.append(None)
                continue

            skel = f.Skeleton

            # Handle case where Skeleton might be an array (multiple users)
            # We take the first tracked user (index 0) or the object itself if not an array
            if isinstance(skel, (np.ndarray, list)):
                if len(skel) > 0:
                    skel = skel[0]
                else:
                    skeleton_data.append(None)
                    continue
            # Fix: Unwrap 0-d array for Skeleton (Cite debug_lesson_16)
            elif isinstance(skel, np.ndarray) and skel.ndim == 0:
                skel = skel.item()

            # Check for WorldPosition
            if not hasattr(skel, "WorldPosition"):
                skeleton_data.append(None)
                continue

            wp = skel.WorldPosition
            # Fix: Unwrap 0-d array for WorldPosition (Cite debug_lesson_10)
            if isinstance(wp, np.ndarray) and wp.ndim == 0:
                wp = wp.item()

            # Extract X, Y, Z
            # Case A: WorldPosition is a 20x3 or 3x20 matrix
            if isinstance(wp, np.ndarray):
                if wp.shape == (20, 3):
                    skeleton_data.append(wp)
                elif wp.shape == (3, 20):
                    skeleton_data.append(wp.T)
                else:
                    skeleton_data.append(None)

            # Case B: WorldPosition is a struct with X, Y, Z fields
            elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                # Ensure they are arrays of length 20
                x = np.atleast_1d(wp.X)
                y = np.atleast_1d(wp.Y)
                z = np.atleast_1d(wp.Z)

                if len(x) == 20 and len(y) == 20 and len(z) == 20:
                    joint_pos = np.stack([x, y, z], axis=1)  # (20, 3)
                    skeleton_data.append(joint_pos)
                else:
                    skeleton_data.append(None)
            else:
                skeleton_data.append(None)

        # Post-process to handle missing frames (interpolation or nearest fill)
        # Convert to numpy array
        processed_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Forward fill
        last_valid = np.zeros((20, 3))
        for i in range(num_frames):
            if skeleton_data[i] is not None:
                processed_data[i] = skeleton_data[i]
                last_valid = skeleton_data[i]
            else:
                processed_data[i] = last_valid

        return processed_data


# ==========================================
# Data Processing Functions
# ==========================================
def extract_audio_features(audio_path, target_num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        # We use standard parameters, but ensure we get enough frames
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

        # Interpolate to match video frames
        # Input to interpolate must be (Batch, Channels, Time)
        # mfcc is (1, n_mfcc, time)
        if mfcc.shape[-1] > 0:
            mfcc_aligned = F.interpolate(
                mfcc.unsqueeze(0),
                size=target_num_frames,
                mode="linear",
                align_corners=False,
            )  # (1, 1, n_mfcc, target_frames)

            mfcc_aligned = (
                mfcc_aligned.squeeze(0).squeeze(0).permute(1, 0)
            )  # (target_frames, n_mfcc)
        else:
            # Handle silent or empty audio
            mfcc_aligned = torch.zeros((target_num_frames, Config.AUDIO_N_MFCC))

        return mfcc_aligned.numpy()

    except Exception:
        # Fallback for missing audio files
        return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)


def process_sample(row, input_dir, augment=False):
    """
    Full pipeline: Parse -> Augment -> Derive -> Align -> Label.
    Returns: features (T, 250), labels (T,)
    """
    # 1. Load Skeleton
    mat_path = os.path.join(input_dir, row["data_path"])
    positions = SkeletonParser.parse(mat_path)

    if positions is None:
        # Return None to indicate failure
        return None, None

    num_frames = positions.shape[0]

    # 2. Augmentation (Noise Injection)
    # Apply BEFORE derivation as per requirements
    if augment:
        noise = np.random.normal(0, Config.NOISE_SIGMA, positions.shape)
        positions = positions + noise

    # 3. Root-Relative Centering
    # HipCenter is index 0
    root_pos = positions[:, 0:1, :]  # (T, 1, 3)
    positions_centered = positions - root_pos  # (T, 20, 3)

    # 4. Explicit Bone Vectors
    # Calculate vector P_child - P_parent
    bone_vectors = []
    for p1, p2 in Config.BONE_PAIRS:
        # p1 is parent, p2 is child usually, or connected joints
        # Vector direction doesn't strictly matter as long as consistent
        vec = positions_centered[:, p2, :] - positions_centered[:, p1, :]
        bone_vectors.append(vec)
    bone_vectors = np.stack(bone_vectors, axis=1)  # (T, 19, 3)

    # 5. Kinematic Derivation (Velocity & Acceleration)
    # Using central differences (np.gradient)
    velocity = np.gradient(positions_centered, axis=0)  # (T, 20, 3)
    acceleration = np.gradient(velocity, axis=0)  # (T, 20, 3)

    # 6. Audio Features
    audio_path = os.path.join(input_dir, row["audio_path"])
    audio_feats = extract_audio_features(audio_path, num_frames)  # (T, 13)

    # 7. Flatten and Concatenate
    # Shapes: (T, 60), (T, 57), (T, 60), (T, 60), (T, 13)
    feat_pos = positions_centered.reshape(num_frames, -1)
    feat_bone = bone_vectors.reshape(num_frames, -1)
    feat_vel = velocity.reshape(num_frames, -1)
    feat_acc = acceleration.reshape(num_frames, -1)

    features = np.concatenate(
        [feat_pos, feat_bone, feat_vel, feat_acc, audio_feats], axis=1
    )
    # Expected dim: 60+57+60+60+13 = 250

    # 8. Generate Labels
    labels = np.zeros(num_frames, dtype=np.int64)
    if isinstance(row["labels"], str):
        gesture_list = json.loads(row["labels"])
    else:
        gesture_list = row["labels"]

    for g in gesture_list:
        start = max(0, g["begin"] - 1)  # 1-based to 0-based
        end = min(num_frames, g["end"])
        labels[start:end] = g["id"]

    return features.astype(np.float32), labels


def create_windows(features, labels, window_size, stride):
    """
    Slices sequence into windows.
    """
    num_frames = features.shape[0]
    windows_x = []
    windows_y = []

    # Handle short sequences by padding
    if num_frames < window_size:
        pad_len = window_size - num_frames
        # Pad features with 0, labels with 0 (background)
        feat_pad = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
        label_pad = np.pad(labels, (0, pad_len), mode="constant")
        return [feat_pad], [label_pad]

    # Sliding window
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        windows_x.append(features[start:end])
        windows_y.append(labels[start:end])

    # Handle remainder if significant?
    # Usually strictly sliding is fine.

    return windows_x, windows_y


def get_data(mode, load_cached_data=True):
    """
    Main data retrieval function with caching logic.
    """
    # Determine paths and settings
    if mode == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PATH
        augment = True  # Apply noise for training set generation
    elif mode == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PATH
        augment = False
    else:  # test
        metadata_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PATH
        augment = False

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            X = data["X"]
            Y = data["Y"]
            return X, Y
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from Scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    all_X = []
    all_Y = []

    for _, row in df.iterrows():
        feats, labs = process_sample(row, Config.INPUT_DIR, augment=augment)

        if feats is None:
            continue

        # Windowing
        wins_x, wins_y = create_windows(feats, labs, Config.WINDOW_SIZE, Config.STRIDE)
        all_X.extend(wins_x)
        all_Y.extend(wins_y)

    # Stack
    if len(all_X) > 0:
        X_final = np.stack(all_X)  # (N, 64, 250)
        Y_final = np.stack(all_Y)  # (N, 64)
    else:
        # Empty dataset fallback
        X_final = np.zeros((0, Config.WINDOW_SIZE, Config.INPUT_DIM), dtype=np.float32)
        Y_final = np.zeros((0, Config.WINDOW_SIZE), dtype=np.int64)

    # 3. Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X_final, Y=Y_final)

    return X_final, Y_final


# ==========================================
# Dataset Class
# ==========================================
class GestureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ==========================================
# DataLoader Factory
# ==========================================
def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    set_seed(Config.SEED)

    # Load Data
    X_train, Y_train = get_data("train", load_cached_data)
    X_val, Y_val = get_data("val", load_cached_data)
    X_test, Y_test = get_data("test", load_cached_data)

    # Create Datasets
    train_ds = GestureDataset(X_train, Y_train)
    val_ds = GestureDataset(X_val, Y_val)
    test_ds = GestureDataset(X_test, Y_test)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Sequence inference usually done 1 by 1 or batched windows
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
