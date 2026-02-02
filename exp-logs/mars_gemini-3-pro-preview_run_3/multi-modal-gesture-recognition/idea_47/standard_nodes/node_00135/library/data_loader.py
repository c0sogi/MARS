import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import safe_load_mat

# Ensure reproducible results
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the Gesture Recognition Task.
    Handles sliding windows for training and full sequences for validation/inference.
    """

    def __init__(self, data, mode="train", stats=None):
        self.data = data
        self.mode = mode
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE
        self.stats = stats

        self.indices = []

        # Pre-calculate indices
        if self.mode == "train":
            # Sliding window segmentation
            for i, sample in enumerate(self.data):
                num_frames = sample["features"].shape[0]
                # We need at least window_size frames
                if num_frames < self.window_size:
                    # Pad short sequences or skip?
                    # Strategy: Pad with zeros if shorter, or just take one window with padding
                    self.indices.append((i, 0))
                else:
                    # Generate sliding windows
                    for start in range(
                        0, num_frames - self.window_size + 1, self.stride
                    ):
                        self.indices.append((i, start))

                    # Ensure the last frames are covered if not exactly divisible
                    last_start = num_frames - self.window_size
                    if last_start > 0 and (last_start % self.stride != 0):
                        self.indices.append((i, last_start))
        else:
            # Full sequences for validation/test
            for i in range(len(self.data)):
                self.indices.append((i, 0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.indices[idx]
        sample = self.data[sample_idx]

        features = sample["features"]  # (T, InputDim)

        if self.mode == "train":
            # Extract Window
            end_frame = start_frame + self.window_size

            # Handle short sequences with padding
            if features.shape[0] < self.window_size:
                pad_len = self.window_size - features.shape[0]
                feat_window = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                label_window = np.pad(
                    sample["labels"], (0, pad_len), mode="constant", constant_values=0
                )
            else:
                feat_window = features[start_frame:end_frame]
                label_window = sample["labels"][start_frame:end_frame]

            # Convert to tensor
            feat_tensor = torch.FloatTensor(feat_window)
            label_tensor = torch.LongTensor(label_window)

            return feat_tensor, label_tensor

        else:
            # Return full sequence
            feat_tensor = torch.FloatTensor(features)
            # For test set, labels might be dummy, but we return them for consistency
            label_tensor = torch.LongTensor(sample["labels"])
            sample_id = sample["sample_id"]

            return feat_tensor, label_tensor, sample_id


def compute_kinematics(joints_3d):
    """
    Computes Central-Difference Kinematics.
    Input: (T, Joints, 3)
    Output: (T, Joints, 9) -> [Pos, Vel, Acc]
    """
    # 1. Position (T, J, 3)
    pos = joints_3d

    # 2. Velocity (Central Difference)
    # gradient returns list of arrays, one per axis. Axis 0 is time.
    vel = np.gradient(pos, axis=0)

    # 3. Acceleration (Central Difference of Velocity)
    acc = np.gradient(vel, axis=0)

    # Concatenate: (T, J, 9)
    return np.concatenate([pos, vel, acc], axis=2)


def augment_skeleton(joints_3d):
    """
    Applies random rotation around Y-axis and scaling.
    Input: (T, Joints, 3)
    """
    # Random Scale
    scale = np.random.uniform(0.9, 1.1)
    joints_3d = joints_3d * scale

    # Random Rotation around Y-axis
    theta = np.random.uniform(-np.pi / 18, np.pi / 18)  # +/- 10 degrees
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Apply rotation: (T, J, 3) dot (3, 3) -> (T, J, 3)
    # Reshape for matmul
    T, J, C = joints_3d.shape
    flat_joints = joints_3d.reshape(-1, 3)
    rotated_flat = flat_joints @ rotation_matrix.T
    rotated_joints = rotated_flat.reshape(T, J, C)

    # Add Gaussian Noise (Cite solution_lesson_node_00133)
    noise = np.random.normal(0, Config.AUG_NOISE_SIGMA, rotated_joints.shape)
    return rotated_joints + noise


def process_audio_mfcc(audio_path, target_frames):
    """
    Extracts MFCCs and aligns them to the video frame count.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.INPUT_DIM_AUDIO,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = mfcc.mean(dim=0)
        else:
            mfcc = mfcc.squeeze(0)

        # Shape is (n_mfcc, time). Transpose to (time, n_mfcc)
        mfcc = mfcc.transpose(0, 1)

        # Resize to match video frames
        # Input to interpolate must be (Batch, Channels, Length) -> (1, n_mfcc, time)
        mfcc_in = mfcc.transpose(0, 1).unsqueeze(0)
        mfcc_out = F.interpolate(
            mfcc_in, size=target_frames, mode="linear", align_corners=False
        )

        # Back to (target_frames, n_mfcc)
        return mfcc_out.squeeze(0).transpose(0, 1).numpy()

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_frames, Config.INPUT_DIM_AUDIO), dtype=np.float32)


def process_dataset(metadata_path, cache_path, mode="train", load_cached_data=True):
    """
    Main data processing pipeline with caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_list = loaded["data"].tolist()
            stats = loaded["stats"].item() if "stats" in loaded else None
            return data_list, stats
        except Exception as e:
            print(f"Cache load failed: {e}. Reprocessing...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_path)

    # Debug Subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        df = df.head(Config.DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Processing subset of {len(df)} samples.")

    processed_data = []

    # Statistics accumulators
    all_features_list = []

    print(f"Processing {len(df)} samples for {mode}...")

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # --- Load Skeleton ---
        mat = safe_load_mat(data_path)
        # Access top-level variables as dict keys (Cite debug_lesson_2)
        if mat is None or "Video" not in mat:
            continue

        video = mat["Video"]
        # Unwrap 0-d array if present (Cite debug_lesson_16)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        if not hasattr(video, "Frames") or not hasattr(video, "NumFrames"):
            continue

        frames = video.Frames
        num_frames = video.NumFrames

        # Extract WorldPosition
        # Handle cases where Frames might be a single object or array
        if isinstance(frames, (list, np.ndarray)):
            frame_list = frames
        else:
            frame_list = [frames]

        # Extract joints: (T, 20, 3)
        # Note: Depending on parsing, this might be slow loop.
        # Optimized extraction:
        skel_data = np.zeros((num_frames, Config.SKELETON_JOINTS, 3), dtype=np.float32)

        valid_frames = min(len(frame_list), num_frames)
        for f in range(valid_frames):
            try:
                if hasattr(frame_list[f], "Skeleton") and hasattr(
                    frame_list[f].Skeleton, "WorldPosition"
                ):
                    wp = frame_list[f].Skeleton.WorldPosition
                    # wp should be 20 objects or array
                    # Assuming standard Kinect structure where WorldPosition is (X,Y,Z) for each joint
                    # If WorldPosition is an array of structs or struct of arrays
                    # Based on description: "WorldPosition structure... X, Y, Z"
                    # And "Skeleton... JointsType... WorldPosition"
                    # Usually mat.Video.Frames[i].Skeleton.WorldPosition is a 20x1 struct array or similar

                    # Heuristic to handle structure variations
                    if (
                        hasattr(wp, "X")
                        and isinstance(wp.X, (np.ndarray, list))
                        and len(wp.X) == 20
                    ):
                        # Struct of arrays
                        skel_data[f, :, 0] = wp.X
                        skel_data[f, :, 1] = wp.Y
                        skel_data[f, :, 2] = wp.Z
                    elif isinstance(wp, (list, np.ndarray)) and len(wp) == 20:
                        # Array of structs
                        for j in range(20):
                            skel_data[f, j, 0] = wp[j].X
                            skel_data[f, j, 1] = wp[j].Y
                            skel_data[f, j, 2] = wp[j].Z
            except:
                pass  # Keep zeros

        # --- Feature Engineering ---

        # 1. Root-Relative Centering
        # Assuming HipCenter is index 0 (standard Kinect)
        hip_center = skel_data[:, 0:1, :]  # (T, 1, 3)
        centered_skel = skel_data - hip_center

        # 2. Augmentation (Only Train)
        if mode == "train":
            centered_skel = augment_skeleton(centered_skel)

        # 3. Kinematics (Pos, Vel, Acc)
        # (T, 20, 9) -> Flatten to (T, 180)
        kinematics = compute_kinematics(centered_skel)
        skel_features = kinematics.reshape(num_frames, -1)

        # 4. Audio MFCC
        audio_features = process_audio_mfcc(audio_path, num_frames)

        # 5. Fusion
        # (T, 180 + 13)
        final_features = np.concatenate([skel_features, audio_features], axis=1)

        # --- Labels ---
        # Create dense labels
        dense_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

        if mode != "test":
            label_list = json.loads(row["labels"])
            for l in label_list:
                lid = l["id"]
                start = max(0, l["begin"] - 1)  # 1-based to 0-based
                end = min(num_frames, l["end"])
                dense_labels[start:end] = lid

        processed_data.append(
            {
                "sample_id": sample_id,
                "features": final_features.astype(np.float32),
                "labels": dense_labels,
            }
        )

        if mode == "train":
            all_features_list.append(final_features)

    # Compute Stats for Normalization (Only on Train)
    stats = {}
    if mode == "train" and all_features_list:
        all_feats = np.concatenate(all_features_list, axis=0)
        mean = np.mean(all_feats, axis=0)
        std = np.std(all_feats, axis=0) + 1e-6  # Avoid div/0
        stats = {"mean": mean, "std": std}

    # Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path, data=np.array(processed_data, dtype=object), stats=stats
    )
    print(f"Saved processed data to {cache_path}")

    return processed_data, stats


def get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Factory function to get dataloaders.
    """
    if debug:
        Config.set_debug_mode()

    # 1. Process/Load Data
    train_data, train_stats = process_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, mode="train"
    )

    # Use training stats for validation and test
    val_data, _ = process_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, mode="val"
    )

    test_data, _ = process_dataset(
        Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, mode="test"
    )

    # 2. Apply Normalization
    # We apply it in-place here to avoid doing it in __getitem__ repeatedly
    if train_stats:
        mean = train_stats["mean"]
        std = train_stats["std"]

        for ds in [train_data, val_data, test_data]:
            for sample in ds:
                sample["features"] = (sample["features"] - mean) / std

    # 3. Create Datasets
    train_ds = GestureDataset(train_data, mode="train")
    val_ds = GestureDataset(val_data, mode="val")
    test_ds = GestureDataset(test_data, mode="test")

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # Val/Test use batch_size=1 for full sequence inference
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader
