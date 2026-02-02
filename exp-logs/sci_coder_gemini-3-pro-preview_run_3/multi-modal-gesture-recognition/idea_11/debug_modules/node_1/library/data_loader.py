import os
import json
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

from library.config import Config
from library.utils import set_seed

# ==========================================
# Helper Classes
# ==========================================


class SkeletonAugmentor:
    """
    Applies random rotation and scaling to raw skeleton coordinates.
    Ensures kinematic consistency by transforming positions before derivatives.
    """

    def __init__(self, rotation_range=15.0, scale_range=0.1):
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def __call__(self, skeleton_data):
        """
        Args:
            skeleton_data (np.ndarray): Shape (T, Joints, 3)
        Returns:
            np.ndarray: Augmented skeleton data.
        """
        T, J, C = skeleton_data.shape

        # 1. Random Scaling
        scale_factor = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        augmented = skeleton_data * scale_factor

        # 2. Random Rotation around Y-axis (Up)
        # Angle in radians
        angle_deg = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle_deg)

        # Rotation matrix for Y-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = np.array(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
        )  # Shape (3, 3)

        # Apply rotation: (T*J, 3) @ (3, 3)
        flat_skel = augmented.reshape(-1, 3)
        rotated_skel = flat_skel @ rot_matrix.T

        return rotated_skel.reshape(T, J, C)


# ==========================================
# Data Processing Functions
# ==========================================


def load_mat_file(path):
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading mat file {path}: {e}")
        return None


def extract_raw_sequence(rgb_path, data_path, audio_path):
    """
    Extracts raw skeleton and audio energy aligned to video frames.
    Returns:
        skeleton (np.ndarray): (NumFrames, 20, 3)
        audio_energy (np.ndarray): (NumFrames, 1)
        num_frames (int)
    """
    # 1. Load Skeleton Data
    mat = load_mat_file(data_path)
    if mat is None or not hasattr(mat, "Video"):
        return None, None, 0

    video = mat.Video
    num_frames = int(video.NumFrames)

    # Extract WorldPosition for all 20 joints
    # Structure: video.Frames[i].Skeleton.WorldPosition.X/Y/Z
    # We assume standard Kinect 20 joints order
    skeleton_frames = []

    # Pre-allocate for speed if possible, but handling missing frames is safer with append
    # To optimize, we assume dense frames.

    # Check if Frames exists and is iterable
    if not hasattr(video, "Frames"):
        return None, None, 0

    frames_data = video.Frames
    if not isinstance(frames_data, (list, np.ndarray)):
        frames_data = [frames_data]  # Single frame case

    # We need to ensure we get exactly num_frames.
    # Sometimes metadata NumFrames != len(Frames). We trust len(Frames) or min.
    actual_frames = len(frames_data)
    limit = min(num_frames, actual_frames)

    # Initialize with zeros (T, 20, 3)
    skeleton_array = np.zeros((limit, 20, 3), dtype=np.float32)

    for i in range(limit):
        frame = frames_data[i]
        if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
            # Some files might have missing skeleton data for a frame
            # We try to extract, if fail, leave as 0 (or prev frame)
            try:
                wp = frame.Skeleton.WorldPosition
                # wp is a struct array of 20 elements (one per joint)
                # We expect 20 joints.
                if len(wp) == 20:
                    for j in range(20):
                        skeleton_array[i, j, 0] = wp[j].X
                        skeleton_array[i, j, 1] = wp[j].Y
                        skeleton_array[i, j, 2] = wp[j].Z
            except:
                pass

    # 2. Load Audio Data & Align
    audio_energy = np.zeros((limit, 1), dtype=np.float32)
    if os.path.exists(audio_path):
        try:
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)  # Mono

            # Calculate samples per video frame
            # Assuming ~20 FPS for Kinect, but we should use actual duration
            # If we don't know FPS, we map audio duration to video frames
            total_samples = len(y)
            samples_per_frame = int(total_samples / limit) if limit > 0 else 0

            if samples_per_frame > 0:
                # Compute RMS energy per frame
                # Reshape pad
                pad_len = (samples_per_frame * limit) - total_samples
                if pad_len > 0:
                    y = np.pad(y, (0, pad_len))
                elif pad_len < 0:
                    y = y[: samples_per_frame * limit]

                y_reshaped = y.reshape(limit, samples_per_frame)
                energy = np.sqrt(np.mean(y_reshaped**2, axis=1))
                audio_energy[:, 0] = energy
        except Exception as e:
            # print(f"Audio load error {audio_path}: {e}")
            pass

    return skeleton_array, audio_energy, limit


def generate_labels(num_frames, labels_list):
    """
    Generates frame-wise class labels and boundary labels.
    """
    class_labels = np.zeros(num_frames, dtype=np.int64)  # 0 is background
    boundary_labels = np.zeros(num_frames, dtype=np.float32)

    for label in labels_list:
        lid = label["id"]
        start = max(0, label["begin"] - 1)  # 1-based to 0-based
        end = min(num_frames, label["end"])

        # Set class
        class_labels[start:end] = lid

        # Set boundaries (radius 2)
        # Start boundary
        b_s = max(0, start - 2)
        b_e = min(num_frames, start + 3)
        boundary_labels[b_s:b_e] = 1.0

        # End boundary
        e_s = max(0, end - 2)
        e_e = min(num_frames, end + 3)
        boundary_labels[e_s:e_e] = 1.0

    return class_labels, boundary_labels


def process_and_cache_dataset(csv_path, cache_name, load_cached_data=True):
    """
    Loads raw data for a dataset split, processes it, and caches it.
    Returns lists of raw arrays.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            list(data["skeletons"]),
            list(data["audios"]),
            list(data["class_labels"]),
            list(data["boundary_labels"]),
            list(data["sample_ids"]),
        )

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    skeletons = []
    audios = []
    class_labels_list = []
    boundary_labels_list = []
    sample_ids = []

    for idx, row in df.iterrows():
        # Construct paths
        rgb_path = os.path.join(Config.INPUT_DIR, row["rgb_path"])
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Extract Raw
        skel, aud, n_frames = extract_raw_sequence(rgb_path, data_path, audio_path)

        if skel is None:
            # Fallback for missing/corrupt data: create zeros
            # We need at least Config.WINDOW_SIZE frames usually, but if video is short/empty?
            # We'll skip or pad. Let's pad to min window size if 0.
            n_frames = max(n_frames, Config.WINDOW_SIZE)
            skel = np.zeros((n_frames, 20, 3), dtype=np.float32)
            aud = np.zeros((n_frames, 1), dtype=np.float32)

        # Labels
        labels_json = (
            json.loads(row["labels"]) if isinstance(row["labels"], str) else []
        )
        c_lbl, b_lbl = generate_labels(n_frames, labels_json)

        skeletons.append(skel)
        audios.append(aud)
        class_labels_list.append(c_lbl)
        boundary_labels_list.append(b_lbl)
        sample_ids.append(row["sample_id"])

    # Cache
    np.savez_compressed(
        cache_path,
        skeletons=np.array(skeletons, dtype=object),
        audios=np.array(audios, dtype=object),
        class_labels=np.array(class_labels_list, dtype=object),
        boundary_labels=np.array(boundary_labels_list, dtype=object),
        sample_ids=np.array(sample_ids, dtype=object),
    )

    return skeletons, audios, class_labels_list, boundary_labels_list, sample_ids


def compute_stats(skeletons):
    """
    Computes global mean and std for derived features from raw skeletons.
    We simulate the feature extraction on a subset to get stats.
    """
    # We need to compute stats for: [Pos(60), Rel(60), Vel(60), Acc(60), Aud(1)]
    # We'll sample 10% of frames from training set

    all_feats = []

    # Hip Center index (SpineBase) is usually 0 in Kinect, but let's check.
    # Assuming index 0 is hip/spine base for relative calc.

    for skel in skeletons[::5]:  # Every 5th sequence
        if len(skel) < 3:
            continue

        # Pos
        pos = skel.reshape(len(skel), -1)

        # Rel (Relative to joint 0)
        root = skel[:, 0, :].reshape(len(skel), 1, 3)
        rel = (skel - root).reshape(len(skel), -1)

        # Vel
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # Acc
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Concat
        # Note: Audio stats handled separately or assumed unit?
        # Let's just compute stats for these 240 dims.
        feat = np.concatenate([pos, rel, vel, acc], axis=1)

        # Sample frames
        indices = np.linspace(0, len(feat) - 1, num=min(len(feat), 50), dtype=int)
        all_feats.append(feat[indices])

    all_feats = np.concatenate(all_feats, axis=0)
    mean = np.mean(all_feats, axis=0)
    std = np.std(all_feats, axis=0) + 1e-6  # Avoid div by zero

    return mean, std


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self,
        skeletons,
        audios,
        class_labels,
        boundary_labels,
        stats=None,
        augment=False,
        window_size=64,
        stride=32,
    ):
        self.skeletons = skeletons
        self.audios = audios
        self.class_labels = class_labels
        self.boundary_labels = boundary_labels
        self.augment = augment
        self.window_size = window_size
        self.stride = stride

        self.mean, self.std = stats if stats is not None else (0, 1)

        self.augmentor = SkeletonAugmentor() if augment else None

        # Pre-compute window indices
        self.windows = []
        for seq_idx in range(len(skeletons)):
            n_frames = len(skeletons[seq_idx])
            # We need valid windows.
            # If sequence is shorter than window, we pad it later?
            # Or we just take one window padded.
            if n_frames < window_size:
                self.windows.append((seq_idx, 0))  # Will handle short seq in getitem
            else:
                for start in range(0, n_frames - window_size + 1, stride):
                    self.windows.append((seq_idx, start))

                # Ensure last frames are covered
                if (n_frames - window_size) % stride != 0:
                    self.windows.append((seq_idx, n_frames - window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]

        # Data retrieval
        raw_skel = self.skeletons[seq_idx]  # (T, 20, 3)
        raw_audio = self.audios[seq_idx]  # (T, 1)
        cls_lbl = self.class_labels[seq_idx]
        bnd_lbl = self.boundary_labels[seq_idx]

        seq_len = len(raw_skel)

        # Handle short sequences (pad if needed)
        if seq_len < self.window_size:
            pad_amt = self.window_size - seq_len
            raw_skel = np.pad(raw_skel, ((0, pad_amt), (0, 0), (0, 0)), mode="edge")
            raw_audio = np.pad(raw_audio, ((0, pad_amt), (0, 0)), mode="constant")
            cls_lbl = np.pad(cls_lbl, (0, pad_amt), mode="constant", constant_values=0)
            bnd_lbl = np.pad(bnd_lbl, (0, pad_amt), mode="constant", constant_values=0)
            seq_len = self.window_size

        # Determine slice indices
        # We need padding for derivatives: -1 for Vel, -2 for Acc
        # To be safe, let's grab start-2 to end
        # But we only need output of length window_size.

        # Slice range: [start, start + window_size]
        # Context needed: [start - 2, start + window_size]

        ctx_start = start_frame - 2
        ctx_end = start_frame + self.window_size

        # Handle boundary conditions for context
        pad_pre = 0
        if ctx_start < 0:
            pad_pre = abs(ctx_start)
            ctx_start = 0

        skel_window = raw_skel[ctx_start:ctx_end]
        audio_window = raw_audio[
            ctx_start:ctx_end
        ]  # Audio doesn't need diffs, but align for simplicity

        if pad_pre > 0:
            # Pad beginning with first frame
            skel_window = np.pad(
                skel_window, ((pad_pre, 0), (0, 0), (0, 0)), mode="edge"
            )
            audio_window = np.pad(audio_window, ((pad_pre, 0), (0, 0)), mode="edge")

        # 1. Augmentation (Kinematically Consistent: Before Derivatives)
        if self.augment and self.augmentor:
            skel_window = self.augmentor(skel_window)

        # 2. Feature Computation
        # Pos: (W+2, 60)
        pos = skel_window.reshape(len(skel_window), -1)

        # Rel: (W+2, 60)
        root = skel_window[:, 0, :].reshape(len(skel_window), 1, 3)
        rel = (skel_window - root).reshape(len(skel_window), -1)

        # Vel: (W+2, 60) -> (W+1, 60)
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # Acc: (W+2, 60) -> (W+1, 60) -> (W, 60)
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # 3. Crop to Window Size (remove context frames)
        # We added 2 frames at start.
        # Indices: 0, 1 are context. 2 is start_frame.
        # Vel needs index-1. Acc needs index-1 (which is vel index-1).
        # Actually, simpler:
        # Pos at t.
        # Vel at t = Pos(t) - Pos(t-1).
        # Acc at t = Vel(t) - Vel(t-1).
        # So at index 2 (start_frame), we use 2, 1, 0.

        final_pos = pos[2:]
        final_rel = rel[2:]
        final_vel = vel[2:]
        final_acc = acc[2:]
        final_aud = audio_window[2:]

        # Concatenate
        # Dims: 60 + 60 + 60 + 60 + 1 = 241
        features = np.concatenate(
            [final_pos, final_rel, final_vel, final_acc, final_aud], axis=1
        )

        # 4. Normalize
        if isinstance(self.mean, np.ndarray):
            # Audio mean/std might be missing from stats if we calculated stats only on skel
            # Let's handle audio norm separately or just assume small values
            # Our stats function returned 240 dims.
            # We have 241.
            f_skel = features[:, :240]
            f_aud = features[:, 240:]

            f_skel = (f_skel - self.mean) / self.std
            features = np.concatenate([f_skel, f_aud], axis=1)

        # 5. Pad to INPUT_DIM (256)
        curr_dim = features.shape[1]
        if curr_dim < Config.INPUT_DIM:
            padding = np.zeros(
                (self.window_size, Config.INPUT_DIM - curr_dim), dtype=np.float32
            )
            features = np.concatenate([features, padding], axis=1)

        # Labels
        # Slice directly from original arrays (no context needed)
        c_window = cls_lbl[start_frame : start_frame + self.window_size]
        b_window = bnd_lbl[start_frame : start_frame + self.window_size]

        # Ensure numeric types (fix for np.object_ error from cached loads)
        features = features.astype(np.float32)
        b_window = b_window.astype(np.float32)

        return (
            torch.FloatTensor(features),
            torch.LongTensor(c_window),
            torch.FloatTensor(b_window),
        )


# ==========================================
# Main Interface
# ==========================================


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Prepares DataLoaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)

    # 1. Load and Cache Data
    train_data = process_and_cache_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        "dataset_train",
        load_cached_data,
    )
    val_data = process_and_cache_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), "dataset_val", load_cached_data
    )
    test_data = process_and_cache_dataset(
        os.path.join(Config.METADATA_DIR, "test.csv"), "dataset_test", load_cached_data
    )

    # Unpack
    train_skel, train_aud, train_cls, train_bnd, _ = train_data
    val_skel, val_aud, val_cls, val_bnd, _ = val_data
    test_skel, test_aud, test_cls, test_bnd, test_ids = test_data

    # Debug Mode: Subset
    if debug:
        subset = Config.DEBUG_SUBSET_SIZE
        train_skel = train_skel[:subset]
        train_aud = train_aud[:subset]
        train_cls = train_cls[:subset]
        train_bnd = train_bnd[:subset]

    # 2. Compute Stats (only on train)
    stats_path = os.path.join(Config.CACHE_DIR, "stats.npz")
    if load_cached_data and os.path.exists(stats_path):
        s = np.load(stats_path)
        stats = (s["mean"], s["std"])
    else:
        print("Computing normalization statistics...")
        mean, std = compute_stats(train_skel)
        np.savez(stats_path, mean=mean, std=std)
        stats = (mean, std)

    # 3. Create Datasets
    train_ds = GestureDataset(
        train_skel,
        train_aud,
        train_cls,
        train_bnd,
        stats=stats,
        augment=True,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
    )

    val_ds = GestureDataset(
        val_skel,
        val_aud,
        val_cls,
        val_bnd,
        stats=stats,
        augment=False,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
    )

    # Test Dataset: Stride should be smaller for dense prediction?
    # Or same stride and we average? Config says stride 32 (50% overlap).
    test_ds = GestureDataset(
        test_skel,
        test_aud,
        test_cls,
        test_bnd,
        stats=stats,
        augment=False,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders Ready. Train Windows: {len(train_ds)}, Val Windows: {len(val_ds)}"
    )

    # We return test_ids separately to map predictions back to files
    return train_loader, val_loader, test_loader, test_ids
