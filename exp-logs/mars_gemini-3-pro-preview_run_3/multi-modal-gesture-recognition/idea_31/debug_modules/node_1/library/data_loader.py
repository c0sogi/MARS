import os
import json
import random
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def polymorphic_skeleton_parser(mat_path):
    """
    Robustly parses the .mat file to extract skeleton joint positions.
    Handles variations in data structure (struct vs cell vs object).
    Returns:
        numpy.ndarray: Shape (T, 20, 3) in meters.
    """
    try:
        # Load with struct_as_record=False to access fields as attributes
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat._fieldnames:
            raise ValueError("Missing 'Video' struct")

        video = mat.Video
        num_frames = int(video.NumFrames)

        # Initialize container (T, Joints, 3)
        # 20 Joints is standard for this Kinect version
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        if not hasattr(video, "Frames"):
            return skeleton_data  # Return zeros if no frames

        frames = video.Frames

        # Handle case where Frames is a single object, list, or array
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        # Iterate through frames
        # Note: The loop is bounded by min(num_frames, len(frames)) to be safe
        iter_len = min(
            num_frames, len(frames) if isinstance(frames, (list, np.ndarray)) else 1
        )

        for t in range(iter_len):
            frame_obj = frames[t]

            # Check if Skeleton exists
            if not hasattr(frame_obj, "Skeleton"):
                continue

            skel = frame_obj.Skeleton

            # Check if WorldPosition exists
            if not hasattr(skel, "WorldPosition"):
                continue

            wp = skel.WorldPosition

            # Extract X, Y, Z
            # Usually wp is a struct array of size 20, or an object with X,Y,Z arrays
            # We need to handle these variations.

            try:
                # Case A: wp is an array of objects/structs (one per joint)
                if isinstance(wp, (np.ndarray, list)) and len(wp) == 20:
                    for j in range(20):
                        joint = wp[j]
                        skeleton_data[t, j, 0] = float(joint.X)
                        skeleton_data[t, j, 1] = float(joint.Y)
                        skeleton_data[t, j, 2] = float(joint.Z)
                # Case B: wp is a single object containing arrays for X, Y, Z
                elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # Assuming X, Y, Z are arrays of length 20
                    # or single values if loop logic is different.
                    # Based on dataset description, it's likely Case A.
                    # But let's try to cast to array if possible.
                    xs = np.atleast_1d(wp.X)
                    ys = np.atleast_1d(wp.Y)
                    zs = np.atleast_1d(wp.Z)
                    if len(xs) == 20:
                        skeleton_data[t, :, 0] = xs
                        skeleton_data[t, :, 1] = ys
                        skeleton_data[t, :, 2] = zs
            except Exception:
                # If parsing fails for a frame, leave it as zeros (or interpolate later)
                continue

        # Convert millimeters to meters
        skeleton_data = skeleton_data / 1000.0

        return skeleton_data

    except Exception as e:
        # print(f"Error parsing {mat_path}: {e}")
        # Return a minimal valid array to prevent crash, but empty
        return np.zeros((10, 20, 3), dtype=np.float32)


def extract_audio_features(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCCs, and aligns them to the video frame count.
    Returns:
        numpy.ndarray: Shape (T, N_MFCC)
    """
    try:
        if not os.path.exists(audio_path):
            return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

        # Load audio
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SR)

        if len(y) == 0:
            return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

        # Extract MFCC
        # hop_length is set in config.
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
        )
        # MFCC shape is (n_mfcc, time_steps) -> Transpose to (time_steps, n_mfcc)
        mfcc = mfcc.T

        # Align to video frames using linear interpolation
        current_frames = mfcc.shape[0]
        if current_frames != target_num_frames:
            if current_frames > 0:
                # Resample along axis 0
                mfcc = scipy.signal.resample(mfcc, target_num_frames, axis=0)
            else:
                mfcc = np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

        return mfcc.astype(np.float32)

    except Exception as e:
        # print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def generate_label_sequence(num_frames, labels_data):
    """
    Generates a frame-wise label vector.
    """
    # Initialize with background class (0)
    label_seq = np.full(num_frames, Config.BACKGROUND_LABEL, dtype=np.int64)

    if not labels_data:
        return label_seq

    for label in labels_data:
        try:
            lid = int(label["id"])
            start = int(label["begin"]) - 1  # 1-based to 0-based
            end = int(label["end"])  # inclusive in Matlab, exclusive for python slice?
            # Matlab 1:10 is 10 frames. Python 0:10 is 10 frames.
            # If Matlab says 1 to 10. Python index 0 to 9.

            # Clamp to video bounds
            start = max(0, start)
            end = min(num_frames, end)

            if start < end:
                label_seq[start:end] = lid
        except:
            continue

    return label_seq


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, raw_data, stats, mode="train"):
        """
        Args:
            raw_data (dict): Dictionary containing 'audio', 'skeleton', 'labels', 'ids'.
            stats (dict): Normalization statistics.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.audio_list = raw_data["audio"]
        self.skeleton_list = raw_data["skeleton"]
        self.labels_list = raw_data["labels"]
        self.ids_list = raw_data["ids"]

        self.audio_mean = stats["audio_mean"]
        self.audio_std = stats["audio_std"]
        self.skel_pos_std = stats["skel_pos_std"]

        # Generate sliding windows
        self.windows = []
        self._prepare_windows()

    def _prepare_windows(self):
        """
        Pre-calculates valid window indices.
        """
        self.windows = []

        # Stride depends on mode
        stride = (
            Config.STRIDE
            if self.mode == "train"
            else int(Config.WINDOW_SIZE * Config.INFERENCE_OVERLAP)
        )
        # For test/val inference, we might want overlapping windows to cover everything.
        # For training, we use Config.STRIDE.

        for seq_idx, (skel, lbls) in enumerate(
            zip(self.skeleton_list, self.labels_list)
        ):
            num_frames = skel.shape[0]

            # If sequence is shorter than window, pad it later?
            # Or just take one window.
            if num_frames < Config.WINDOW_SIZE:
                self.windows.append((seq_idx, 0))
                continue

            # Generate start indices
            for start_idx in range(0, num_frames - Config.WINDOW_SIZE + 1, stride):
                self.windows.append((seq_idx, start_idx))

            # Ensure the last frame is covered (important for inference)
            if num_frames >= Config.WINDOW_SIZE:
                last_start = num_frames - Config.WINDOW_SIZE
                if self.windows[-1][1] != last_start:
                    self.windows.append((seq_idx, last_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]

        # Retrieve full sequence data
        # Explicitly cast to numeric types to handle object arrays from cache
        full_skel = self.skeleton_list[seq_idx].astype(np.float32)  # (T, 20, 3)
        full_audio = self.audio_list[seq_idx].astype(np.float32)  # (T, 13)
        full_labels = self.labels_list[seq_idx].astype(np.int64)  # (T,)

        seq_len = full_skel.shape[0]

        # Determine window bounds
        end_frame = start_frame + Config.WINDOW_SIZE

        # Handle padding if sequence is shorter than window
        if seq_len < Config.WINDOW_SIZE:
            # Pad with zeros
            pad_len = Config.WINDOW_SIZE - seq_len

            skel_window = full_skel
            audio_window = full_audio
            label_window = full_labels

            # Pad arrays
            skel_window = np.pad(
                skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="edge"
            )
            audio_window = np.pad(audio_window, ((0, pad_len), (0, 0)), mode="constant")
            label_window = np.pad(
                label_window, (0, pad_len), mode="constant", constant_values=0
            )

        else:
            skel_window = full_skel[start_frame:end_frame]
            audio_window = full_audio[start_frame:end_frame]
            label_window = full_labels[start_frame:end_frame]

        # ------------------------------------------
        # 1. Augmentation (Train Only)
        # ------------------------------------------
        if self.mode == "train":
            # Random Rotation around Y-axis
            theta = np.random.uniform(-0.3, 0.3)  # Radians, approx +/- 17 degrees
            rot_matrix = R.from_euler("y", theta).as_matrix()  # (3, 3)

            # Apply rotation: (T, J, 3) dot (3, 3) -> (T, J, 3)
            # Reshape to (T*J, 3) for matmul
            T, J, C = skel_window.shape
            flat_skel = skel_window.reshape(-1, 3)
            flat_skel = flat_skel @ rot_matrix.T
            skel_window = flat_skel.reshape(T, J, C)

            # Random Scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

        # ------------------------------------------
        # 2. Kinematic Feature Computation
        # ------------------------------------------
        # Position: (T, 20, 3)
        pos = skel_window

        # Velocity: (T, 20, 3) - Gradient along time
        vel = np.gradient(pos, axis=0)

        # Acceleration: (T, 20, 3)
        acc = np.gradient(vel, axis=0)

        # ------------------------------------------
        # 3. Hierarchical Normalization
        # ------------------------------------------
        # Audio Z-Score
        audio_norm = (audio_window - self.audio_mean) / (self.audio_std + 1e-6)

        # Skeleton Scaling (Divide all by sigma_pos)
        pos_norm = pos / (self.skel_pos_std + 1e-6)
        vel_norm = vel / (self.skel_pos_std + 1e-6)
        acc_norm = acc / (self.skel_pos_std + 1e-6)

        # Flatten skeleton features: (T, 20*3)
        pos_flat = pos_norm.reshape(Config.WINDOW_SIZE, -1)
        vel_flat = vel_norm.reshape(Config.WINDOW_SIZE, -1)
        acc_flat = acc_norm.reshape(Config.WINDOW_SIZE, -1)

        # Concatenate Kinematics: (T, 180)
        skel_features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)

        # ------------------------------------------
        # 4. Early Fusion
        # ------------------------------------------
        # Concatenate Audio + Skeleton: (T, 13 + 180) = (T, 193)
        features = np.concatenate([audio_norm, skel_features], axis=1)

        # Convert to Tensor
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(label_window, dtype=torch.long)

        return features, labels


# ==========================================
# Data Processing Pipeline
# ==========================================


def load_and_process_split(metadata_path, split_name, load_cached_data=True):
    """
    Loads raw data, processes it, and caches it.
    Returns dict with lists of arrays.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"dataset_{split_name}.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            return {
                "audio": list(data["audio"]),
                "skeleton": list(data["skeleton"]),
                "labels": list(data["labels"]),
                "ids": list(data["ids"]),
            }
        except Exception as e:
            print(f"Cache load failed for {split_name}: {e}. Recomputing...")

    # 2. Compute from Scratch
    if not os.path.exists(metadata_path):
        return {"audio": [], "skeleton": [], "labels": [], "ids": []}

    df = pd.read_csv(metadata_path)

    # Debug mode
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLES)

    audio_list = []
    skeleton_list = []
    labels_list = []
    ids_list = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]

        # Paths
        skel_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Parse Skeleton
        skeleton = polymorphic_skeleton_parser(skel_path)  # (T, 20, 3)
        num_frames = skeleton.shape[0]

        # Parse Audio (Align to num_frames)
        audio = extract_audio_features(audio_path, num_frames)  # (T, 13)

        # Generate Labels
        if "labels" in row and isinstance(row["labels"], str):
            labels_data = json.loads(row["labels"])
            labels = generate_label_sequence(num_frames, labels_data)
        else:
            labels = np.zeros(num_frames, dtype=np.int64)

        audio_list.append(audio)
        skeleton_list.append(skeleton)
        labels_list.append(labels)
        ids_list.append(sample_id)

    # 3. Save Cache
    np.savez_compressed(
        cache_file,
        audio=np.array(audio_list, dtype=object),
        skeleton=np.array(skeleton_list, dtype=object),
        labels=np.array(labels_list, dtype=object),
        ids=np.array(ids_list, dtype=object),
    )

    return {
        "audio": audio_list,
        "skeleton": skeleton_list,
        "labels": labels_list,
        "ids": ids_list,
    }


def compute_normalization_stats(train_data, load_cached_data=True):
    """
    Computes global mean/std for audio and global std for skeleton positions.
    """
    stats_file = os.path.join(Config.CACHE_DIR, "stats.npz")

    if load_cached_data and os.path.exists(stats_file):
        data = np.load(stats_file)
        return {
            "audio_mean": data["audio_mean"],
            "audio_std": data["audio_std"],
            "skel_pos_std": data["skel_pos_std"],
        }

    # Compute from train data
    print("Computing normalization stats from training data...")

    # Audio Stats
    all_audio = np.concatenate(train_data["audio"], axis=0)
    audio_mean = np.mean(all_audio, axis=0)
    audio_std = np.std(all_audio, axis=0)

    # Skeleton Stats (Position Only)
    all_skel = np.concatenate(train_data["skeleton"], axis=0)  # (TotalFrames, 20, 3)
    # Global std deviation of all position coordinates
    skel_pos_std = np.std(all_skel)

    np.savez(
        stats_file,
        audio_mean=audio_mean,
        audio_std=audio_std,
        skel_pos_std=skel_pos_std,
    )

    return {
        "audio_mean": audio_mean,
        "audio_std": audio_std,
        "skel_pos_std": skel_pos_std,
    }


# ==========================================
# Main Interface
# ==========================================


def get_dataloaders(config=Config, load_cached_data=True):
    """
    Factory function to create dataloaders.
    """
    # Load Raw Data
    train_raw = load_and_process_split(
        config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_raw = load_and_process_split(config.VAL_METADATA_PATH, "val", load_cached_data)
    test_raw = load_and_process_split(
        config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Compute Stats (from Train only)
    stats = compute_normalization_stats(train_raw, load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(train_raw, stats, mode="train")
    val_dataset = GestureDataset(val_raw, stats, mode="val")
    test_dataset = GestureDataset(test_raw, stats, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
