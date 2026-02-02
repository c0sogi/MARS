import os
import json
import glob
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from library.config import Config

# ==========================================
# Reproducibility
# ==========================================
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# Parsers & Processors
# ==========================================


class PolymorphicSkeletonParser:
    """
    Parses .mat files robustly handling variations in Matlab struct/cell array formats.
    Extracts 20 joints x 3 coordinates (WorldPosition).
    """

    def __init__(self, num_joints=Config.NUM_JOINTS):
        self.num_joints = num_joints

    def parse(self, mat_path):
        try:
            # Load mat file
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            if not hasattr(mat, "Video"):
                return None

            video = mat["Video"]

            # Determine NumFrames
            num_frames = 0
            if hasattr(video, "NumFrames"):
                num_frames = int(video.NumFrames)

            # Extract Frames
            frames_data = []
            if hasattr(video, "Frames"):
                frames_data = video.Frames

            # Handle different container types for Frames
            frames_list = []
            if isinstance(frames_data, (np.ndarray, list)):
                if len(frames_data) > 0:
                    frames_list = frames_data
            elif isinstance(frames_data, scipy.io.matlab.mat_struct):
                frames_list = [frames_data]

            # If NumFrames doesn't match list length, trust the list length if > 0
            real_len = len(frames_list)
            if real_len > 0:
                num_frames = real_len

            # Container for skeleton: (T, J, 3)
            skeleton_seq = np.zeros((num_frames, self.num_joints, 3), dtype=np.float32)

            for t, frame_obj in enumerate(frames_list):
                if t >= num_frames:
                    break

                # Check for Skeleton
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel = frame_obj.Skeleton

                # Skeleton might be a struct or array of structs.
                # The description says "Array of Skeleton structures".
                # Usually for 1 user, it's a single struct or array of length 1.
                # We take the first tracked user if multiple exist, or just the struct.

                target_skel = None
                if isinstance(skel, (np.ndarray, list)):
                    if len(skel) > 0:
                        target_skel = skel[0]  # Assume first skeleton is the target
                elif isinstance(skel, scipy.io.matlab.mat_struct):
                    target_skel = skel

                if target_skel is None:
                    continue

                # Extract WorldPosition
                if hasattr(target_skel, "WorldPosition"):
                    wp = target_skel.WorldPosition
                    # wp should be array of 20 structs or 20x3 matrix?
                    # Description: "WorldPosition... formed by 20x4 matrix"? No, that's Rotation.
                    # Description: "WorldPosition... The X value..."
                    # Usually in these datasets, wp is an array of Joint structures or a matrix.
                    # Let's handle common cases.

                    joints_xyz = np.zeros((self.num_joints, 3), dtype=np.float32)

                    # Case A: wp is an array of objects (one per joint)
                    if (
                        isinstance(wp, (np.ndarray, list))
                        and len(wp) == self.num_joints
                    ):
                        for j in range(self.num_joints):
                            joint = wp[j]
                            if (
                                hasattr(joint, "X")
                                and hasattr(joint, "Y")
                                and hasattr(joint, "Z")
                            ):
                                joints_xyz[j] = [joint.X, joint.Y, joint.Z]
                            elif (
                                isinstance(joint, (np.ndarray, list))
                                and len(joint) >= 3
                            ):
                                joints_xyz[j] = joint[:3]

                    # Case B: wp is a single object with arrays X, Y, Z inside? Unlikely.
                    # Case C: The skeleton struct itself has 'Joints' array?
                    # Let's stick to the prompt: "Skeleton... WorldPosition... X, Y, Z".
                    # It implies WorldPosition is a struct per joint.

                    # Fallback: If we couldn't parse, check if WorldPosition is a 20x3 matrix
                    elif isinstance(wp, (np.ndarray)) and wp.shape == (
                        self.num_joints,
                        3,
                    ):
                        joints_xyz = wp

                    skeleton_seq[t] = joints_xyz

            # Simple imputation for missing frames (zeros)
            # If a frame is all zeros, copy from previous if available
            for t in range(1, num_frames):
                if np.all(skeleton_seq[t] == 0):
                    skeleton_seq[t] = skeleton_seq[t - 1]

            return skeleton_seq

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return None


def process_audio(audio_path, target_num_frames):
    """
    Loads audio, computes MFCC, and interpolates to match video frame count.
    Returns: (target_num_frames, n_mfcc)
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        # We use standard settings. Window size/hop don't strictly matter
        # because we interpolate to video frames.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (time, n_mfcc)

        if mfcc.shape[0] < 2:
            return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

        # Interpolate to match video frames
        x_old = np.linspace(0, 1, mfcc.shape[0])
        x_new = np.linspace(0, 1, target_num_frames)

        f = interp1d(x_old, mfcc, axis=0, kind="linear", fill_value="extrapolate")
        mfcc_interp = f(x_new)

        return mfcc_interp.astype(np.float32)

    except Exception:
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from Positions.
    Input: (T, J, 3)
    Output: (T, J, 9) -> [Pos, Vel, Acc]
    """
    # Positions: (T, J, 3)
    # Velocity: (T, J, 3)
    # Acceleration: (T, J, 3)

    # Gradient uses central difference for interior, one-sided for boundaries
    # But for causality/simplicity in online settings, simple diff is often used.
    # Here we use np.gradient for smoother derivatives on the full sequence.

    vel = np.gradient(positions, axis=0)
    acc = np.gradient(vel, axis=0)

    # Concatenate: (T, J, 9)
    return np.concatenate([positions, vel, acc], axis=2)


# ==========================================
# Data Management
# ==========================================


def load_and_process_data(split_name, metadata_path, load_cached_data=True):
    """
    Main function to load data, process features, and manage cache.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"dataset_{split_name}.npz")
    stats_file = os.path.join(Config.CACHE_DIR, "stats.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading {split_name} from cache...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            # Reconstruct list of dicts
            samples = []
            # Keys are like 'sample_0_skel', 'sample_0_id'
            # We stored a list of IDs in 'sample_ids'
            sample_ids = data["sample_ids"]
            for i, sid in enumerate(sample_ids):
                samples.append(
                    {
                        "sample_id": str(sid),
                        "skeleton": data[f"skel_{i}"],  # (T, J, 9)
                        "audio": data[f"audio_{i}"],  # (T, 13)
                        "labels": data[f"label_{i}"],  # (T,)
                    }
                )

            # Load stats if available
            audio_stats = None
            if os.path.exists(stats_file):
                s = np.load(stats_file)
                audio_stats = {"mean": s["mean"], "std": s["std"]}

            return samples, audio_stats
        except Exception as e:
            # print(f"Cache load failed: {e}. Recomputing...")
            pass

    # 2. Process from Scratch
    df = pd.read_csv(metadata_path)
    parser = PolymorphicSkeletonParser()

    samples_data = []

    # Accumulators for Audio Stats (only for train split)
    audio_sum = np.zeros(Config.N_MFCC)
    audio_sq_sum = np.zeros(Config.N_MFCC)
    audio_count = 0

    for idx, row in df.iterrows():
        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Parse Skeleton
        skeleton_pos = parser.parse(mat_path)  # (T, J, 3)

        if skeleton_pos is None:
            # Skip corrupted samples
            continue

        num_frames = skeleton_pos.shape[0]

        # Compute Kinematics (Pos, Vel, Acc) -> (T, J, 9)
        skeleton_kin = compute_kinematics(skeleton_pos)

        # Process Audio
        audio_feat = process_audio(audio_path, num_frames)  # (T, 13)

        # Update stats if training
        if split_name == "train":
            audio_sum += np.sum(audio_feat, axis=0)
            audio_sq_sum += np.sum(audio_feat**2, axis=0)
            audio_count += num_frames

        # Process Labels
        labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)
        if "labels" in row and isinstance(row["labels"], str):
            label_list = json.loads(row["labels"])
            for l in label_list:
                # Matlab 1-based [Begin, End]
                # Python 0-based: start = Begin-1, end = End (exclusive slice? No, End is inclusive in Matlab)
                # If Matlab says 1..10, it means frames 1,2...10.
                # Python indices: 0..9.
                # So start = l['begin'] - 1
                # end = l['end'] (slice excludes end, so this covers up to l['end']-1)
                start = max(0, int(l["begin"]) - 1)
                end = min(num_frames, int(l["end"]))
                lid = int(l["id"])
                if start < end:
                    labels[start:end] = lid

        samples_data.append(
            {
                "sample_id": row["sample_id"],
                "skeleton": skeleton_kin.astype(np.float32),
                "audio": audio_feat.astype(np.float32),
                "labels": labels.astype(np.int64),
            }
        )

    # 3. Save Cache
    save_dict = {}
    sid_list = []
    for i, s in enumerate(samples_data):
        sid_list.append(s["sample_id"])
        save_dict[f"skel_{i}"] = s["skeleton"]
        save_dict[f"audio_{i}"] = s["audio"]
        save_dict[f"label_{i}"] = s["labels"]

    save_dict["sample_ids"] = np.array(sid_list)
    np.savez_compressed(cache_file, **save_dict)

    # 4. Save/Load Stats
    audio_stats = None
    if split_name == "train" and audio_count > 0:
        mean = audio_sum / audio_count
        std = np.sqrt((audio_sq_sum / audio_count) - mean**2) + 1e-6
        np.savez(stats_file, mean=mean, std=std)
        audio_stats = {"mean": mean, "std": std}
    elif os.path.exists(stats_file):
        s = np.load(stats_file)
        audio_stats = {"mean": s["mean"], "std": s["std"]}
    else:
        # Fallback if no stats (e.g. only running test)
        audio_stats = {"mean": np.zeros(Config.N_MFCC), "std": np.ones(Config.N_MFCC)}

    return samples_data, audio_stats


# ==========================================
# Datasets
# ==========================================


class GestureDataset(Dataset):
    """
    Dataset for Training/Validation using Sliding Windows.
    Applies Kinematic Augmentation and Physical Scaling.
    """

    def __init__(
        self,
        samples,
        audio_stats,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        augment=False,
    ):
        self.samples = samples
        self.audio_mean = audio_stats["mean"].astype(np.float32)
        self.audio_std = audio_stats["std"].astype(np.float32)
        self.window_size = window_size
        self.augment = augment

        # Pre-calculate windows
        self.windows = []
        for i, s in enumerate(samples):
            num_frames = s["skeleton"].shape[0]
            # Generate windows
            # If sequence is shorter than window, pad? Or skip?
            # Strategy: Pad if short, otherwise slide.

            if num_frames < window_size:
                self.windows.append(
                    (i, 0, num_frames)
                )  # Special case, handle in getitem
            else:
                for start in range(0, num_frames - window_size + 1, stride):
                    self.windows.append((i, start, start + window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start, end = self.windows[idx]
        sample = self.samples[sample_idx]

        # Extract Raw Data
        skel_data = sample["skeleton"][start:end]  # (T, J, 9)
        audio_data = sample["audio"][start:end]  # (T, 13)
        labels = sample["labels"][start:end]  # (T,)

        curr_len = skel_data.shape[0]

        # Padding if needed (for short sequences)
        if curr_len < self.window_size:
            pad_len = self.window_size - curr_len
            # Pad skeleton with zeros
            skel_pad = np.zeros((pad_len, *skel_data.shape[1:]), dtype=np.float32)
            skel_data = np.concatenate([skel_data, skel_pad], axis=0)

            # Pad audio with zeros (silence)
            audio_pad = np.zeros((pad_len, *audio_data.shape[1:]), dtype=np.float32)
            audio_data = np.concatenate([audio_data, audio_pad], axis=0)

            # Pad labels with 0 (background)
            label_pad = np.zeros((pad_len,), dtype=np.int64)
            labels = np.concatenate([labels, label_pad], axis=0)

        # 1. Kinematic Augmentation (Train Only)
        # Apply Rotation and Scaling to P, V, A simultaneously
        if self.augment:
            # Random Rotation around Y-axis
            theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
            c, s = np.cos(theta), np.sin(theta)
            R_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            # Random Scale
            scale = np.random.uniform(0.85, 1.15)

            # Reshape to (T*J, 3) for efficient matmul?
            # skel_data is (T, J, 9).
            # 0:3 is Pos, 3:6 is Vel, 6:9 is Acc.
            # All transform identically under Rotation and Scale.

            # Apply Scale
            skel_data = skel_data * scale

            # Apply Rotation
            # We can reshape to (T*J*3, 3) temporarily? No, structure is (T, J, 9)
            # Let's iterate components
            for k in range(3):  # Pos, Vel, Acc
                comp = skel_data[:, :, k * 3 : (k + 1) * 3]  # (T, J, 3)
                # Apply R: (N, 3) @ R.T
                shape_orig = comp.shape
                comp_flat = comp.reshape(-1, 3)
                comp_rot = comp_flat @ R_mat.T
                skel_data[:, :, k * 3 : (k + 1) * 3] = comp_rot.reshape(shape_orig)

        # 2. Physical Scaling (Deterministic)
        skel_data = skel_data * Config.PHYSICAL_SCALE

        # Flatten Skeleton: (T, J*9)
        skel_flat = skel_data.reshape(self.window_size, -1)

        # 3. Audio Normalization
        audio_norm = (audio_data - self.audio_mean) / self.audio_std

        # 4. Early Fusion
        features = np.concatenate([skel_flat, audio_norm], axis=1)  # (T, 193)

        return torch.from_numpy(features), torch.from_numpy(labels)


class SequenceDataset(Dataset):
    """
    Dataset for Validation/Inference.
    Returns full sequences. No augmentation.
    """

    def __init__(self, samples, audio_stats):
        self.samples = samples
        self.audio_mean = audio_stats["mean"].astype(np.float32)
        self.audio_std = audio_stats["std"].astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        skel_data = sample["skeleton"]  # (T, J, 9)
        audio_data = sample["audio"]  # (T, 13)
        labels = sample["labels"]  # (T,)

        T = skel_data.shape[0]

        # Physical Scaling
        skel_data = skel_data * Config.PHYSICAL_SCALE
        skel_flat = skel_data.reshape(T, -1)

        # Audio Normalization
        audio_norm = (audio_data - self.audio_mean) / self.audio_std

        # Fusion
        features = np.concatenate([skel_flat, audio_norm], axis=1)

        return torch.from_numpy(features), torch.from_numpy(labels), sample["sample_id"]


# ==========================================
# Main Interface
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    # Load Data
    train_samples, stats = load_and_process_data(
        "train", os.path.join(Config.METADATA_DIR, "train.csv"), load_cached_data
    )
    val_samples, _ = load_and_process_data(
        "val", os.path.join(Config.METADATA_DIR, "val.csv"), load_cached_data
    )
    test_samples, _ = load_and_process_data(
        "test", os.path.join(Config.METADATA_DIR, "test.csv"), load_cached_data
    )

    # Create Datasets
    # Train: Sliding Window + Augmentation
    train_ds = GestureDataset(train_samples, stats, augment=True)

    # Val: Full Sequence (for metric calculation) or Window?
    # For training monitoring, window is easier for batching.
    # But for accurate Levenshtein, we need full sequence.
    # We will provide SequenceDataset for Val to allow custom validation loop.
    val_ds = SequenceDataset(val_samples, stats)

    # Test: Full Sequence
    test_ds = SequenceDataset(test_samples, stats)

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test loaders use batch_size=1 to handle variable sequence lengths
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader
