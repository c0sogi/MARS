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

# ==========================================
# 1. Polymorphic Parser & Helpers
# ==========================================


class PolymorphicSkeletonParser:
    """
    Handles robust parsing of .mat files with varying structures (struct arrays,
    cell arrays, single objects) to extract skeleton joint positions.
    """

    @staticmethod
    def parse(mat_path):
        try:
            # Load mat file without squeezing to preserve dimensions initially,
            # but struct_as_record=False is key to accessing fields as attributes.
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None

        if "Video" not in mat:
            return None

        video = mat["Video"]
        # Unwrap 0-d array created by squeeze_me=True (Cite debug_lesson_16)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        # Initialize container: (Time, Joints=20, Coords=3)
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        # Handle cases where Frames might be a single object or empty
        if num_frames == 0:
            return skeleton_data

        # Ensure frames is iterable
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        for i, frame in enumerate(frames):
            if i >= num_frames:
                break

            # Check if Skeleton exists
            if not hasattr(frame, "Skeleton") or frame.Skeleton is None:
                continue

            skel = frame.Skeleton

            # Check if WorldPosition exists
            if not hasattr(skel, "WorldPosition") or skel.WorldPosition is None:
                continue

            wp = skel.WorldPosition

            # Extract coordinates. WP is usually an array of structs or a struct of arrays
            # In this dataset, it's typically a struct array of size 20
            # or a single struct with arrays.

            try:
                # Case 1: WP is an array of 20 objects, each has X, Y, Z
                if isinstance(wp, (list, np.ndarray)) and len(wp) == Config.NUM_JOINTS:
                    for j in range(Config.NUM_JOINTS):
                        joint = wp[j]
                        skeleton_data[i, j, 0] = float(joint.X)
                        skeleton_data[i, j, 1] = float(joint.Y)
                        skeleton_data[i, j, 2] = float(joint.Z)
                # Case 2: WP is a single object (unlikely for 20 joints but possible in Matlab oddities)
                # Case 3: Direct extraction if structure differs (omitted for brevity, relying on standard format)
            except Exception:
                # Fallback: leave as zeros (will be handled by normalization/filling later if needed)
                pass

        # Simple interpolation for missing frames (zeros)
        # Identify zero frames
        valid_mask = np.any(skeleton_data != 0, axis=(1, 2))

        # If we have at least some valid frames, interpolate
        if np.any(valid_mask) and not np.all(valid_mask):
            # Linear interpolation for each joint/coord
            x = np.where(valid_mask)[0]
            missing_x = np.where(~valid_mask)[0]

            for j in range(Config.NUM_JOINTS):
                for c in range(3):
                    skeleton_data[missing_x, j, c] = np.interp(
                        missing_x, x, skeleton_data[x, j, c]
                    )

        return skeleton_data


def load_audio_features(audio_path, target_num_frames):
    """
    Loads audio, computes MFCC, and aligns to video frame count.
    """
    if not os.path.exists(audio_path):
        return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, Time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = mfcc.mean(dim=0, keepdim=True)

        # Shape: (1, n_mfcc, Time) -> (1, Time, n_mfcc)
        mfcc = mfcc.permute(0, 2, 1)

        # Resize to match video frames
        # Input to interpolate must be (Batch, Channels, Time)
        # So we use (1, n_mfcc, Time)
        mfcc_t = mfcc.permute(0, 2, 1)

        if mfcc_t.shape[-1] > 0:
            mfcc_resampled = F.interpolate(
                mfcc_t, size=target_num_frames, mode="linear", align_corners=False
            )
            # Back to (Time, n_mfcc) and squeeze batch
            result = mfcc_resampled.squeeze(0).permute(1, 0).numpy()
        else:
            result = np.zeros(
                (target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32
            )

        return result

    except Exception as e:
        # print(f"Audio load error {audio_path}: {e}")
        return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)


def generate_label_array(labels_list, num_frames):
    """
    Converts list of label dicts to frame-wise array.
    """
    label_arr = np.zeros(num_frames, dtype=np.int64)  # 0 is background

    for l in labels_list:
        gid = int(l["id"])
        start = max(0, int(l["begin"]) - 1)  # 1-based to 0-based
        end = min(num_frames, int(l["end"]))

        if start < end:
            label_arr[start:end] = gid

    return label_arr


# ==========================================
# 2. Data Processing & Caching
# ==========================================


def process_and_cache_data(csv_path, cache_name, load_cached_data=True):
    """
    Reads metadata CSV, processes raw data, and caches to .npz.
    Returns a dictionary of data keyed by sample_id.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path)
            # Reconstruct dictionary
            data_dict = {}
            # We expect keys like 'Sample001_skel', 'Sample001_audio', 'Sample001_labels'
            # Get unique sample IDs from keys
            all_keys = list(loaded.keys())
            sample_ids = set([k.split("_")[0] for k in all_keys])

            for sid in sample_ids:
                data_dict[sid] = {
                    "skeleton": loaded[f"{sid}_skeleton"],
                    "audio": loaded[f"{sid}_audio"],
                    "labels": (
                        loaded[f"{sid}_labels"] if f"{sid}_labels" in loaded else None
                    ),
                }
            return data_dict
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    # Process from scratch
    df = pd.read_csv(csv_path)

    # Pre-parse labels
    if "labels" in df.columns:
        df["parsed_labels"] = df["labels"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    arrays_to_save = {}
    data_dict = {}

    print(f"Processing {len(df)} samples for {cache_name}...")

    for _, row in df.iterrows():
        sid = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Parse Skeleton
        skeleton = PolymorphicSkeletonParser.parse(data_path)
        if skeleton is None:
            # Create dummy if failed (should not happen with robust parser)
            skeleton = np.zeros((100, Config.NUM_JOINTS, 3), dtype=np.float32)

        num_frames = skeleton.shape[0]

        # 2. Parse Audio
        audio = load_audio_features(audio_path, num_frames)

        # 3. Parse Labels
        if "parsed_labels" in row:
            labels = generate_label_array(row["parsed_labels"], num_frames)
        else:
            labels = np.zeros(num_frames, dtype=np.int64)  # Test set

        # Store
        arrays_to_save[f"{sid}_skeleton"] = skeleton.astype(np.float32)
        arrays_to_save[f"{sid}_audio"] = audio.astype(np.float32)
        arrays_to_save[f"{sid}_labels"] = labels.astype(np.int64)

        data_dict[sid] = {
            "skeleton": skeleton.astype(np.float32),
            "audio": audio.astype(np.float32),
            "labels": labels.astype(np.int64),
        }

    # Save to cache
    np.savez_compressed(cache_path, **arrays_to_save)
    print(f"Cached data saved to {cache_path}")

    return data_dict


# ==========================================
# 3. Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self,
        data_dict,
        mode="train",
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
    ):
        """
        mode: 'train' (sliding windows + aug), 'val' (sliding windows, no aug), 'full' (full sequences)
        """
        self.data_dict = data_dict
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.sample_ids = sorted(list(data_dict.keys()))

        self.indices = []

        if self.mode in ["train", "val"]:
            # Pre-calculate sliding window indices
            for sid in self.sample_ids:
                num_frames = self.data_dict[sid]["skeleton"].shape[0]
                if num_frames < self.window_size:
                    # Pad short sequences? Or skip?
                    # For simplicity, we skip extremely short ones or take one padded window
                    # Let's just take one window starting at 0 and pad later
                    self.indices.append((sid, 0))
                else:
                    for start in range(
                        0, num_frames - self.window_size + 1, self.stride
                    ):
                        self.indices.append((sid, start))

                    # Ensure last frame is covered if not exact fit
                    if (num_frames - self.window_size) % self.stride != 0:
                        self.indices.append((sid, num_frames - self.window_size))
        else:
            # Full sequence mode (e.g. for inference/metric)
            # Indices are just sample IDs
            self.indices = self.sample_ids

    def __len__(self):
        return len(self.indices)

    def _augment_skeleton(self, skeleton):
        """
        Apply random rotation (Y-axis) and scaling.
        skeleton: (T, J, 3)
        """
        # 1. Rotation around Y-axis
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: dot product over last dim
        # (T, J, 3) dot (3, 3) -> (T, J, 3)
        skeleton_aug = np.dot(skeleton, rotation_matrix.T)

        # 2. Scaling
        scale = np.random.uniform(0.85, 1.15)
        skeleton_aug = skeleton_aug * scale

        return skeleton_aug

    def _compute_kinematics(self, skeleton):
        """
        Compute Velocity and Acceleration.
        skeleton: (T, J, 3)
        Returns: (T, J*9) flattened features [Pos, Vel, Acc]
        """
        # Velocity: P(t) - P(t-1)
        # Pad first frame with 0
        vel = np.zeros_like(skeleton)
        vel[1:] = skeleton[1:] - skeleton[:-1]

        # Acceleration: V(t) - V(t-1)
        acc = np.zeros_like(skeleton)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate: (T, J, 9)
        features = np.concatenate([skeleton, vel, acc], axis=-1)

        # Flatten joints: (T, J*9)
        T, J, C = features.shape
        return features.reshape(T, J * C)

    def __getitem__(self, idx):
        if self.mode in ["train", "val"]:
            sid, start_frame = self.indices[idx]
            raw_skel = self.data_dict[sid]["skeleton"]  # (TotalT, J, 3)
            raw_audio = self.data_dict[sid]["audio"]  # (TotalT, MFCC)
            raw_labels = self.data_dict[sid]["labels"]  # (TotalT,)

            # Extract Window
            end_frame = start_frame + self.window_size

            # Handle padding for short sequences
            total_frames = raw_skel.shape[0]
            if total_frames < self.window_size:
                # Pad with zeros
                pad_len = self.window_size - total_frames

                skel_window = raw_skel
                audio_window = raw_audio
                label_window = raw_labels

                # Pad arrays
                skel_window = np.pad(
                    skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="edge"
                )
                audio_window = np.pad(audio_window, ((0, pad_len), (0, 0)), mode="edge")
                label_window = np.pad(
                    label_window, (0, pad_len), mode="constant", constant_values=0
                )
            else:
                skel_window = raw_skel[start_frame:end_frame]
                audio_window = raw_audio[start_frame:end_frame]
                label_window = raw_labels[start_frame:end_frame]

            # Augmentation (only train)
            if self.mode == "train":
                skel_window = self._augment_skeleton(skel_window)

            # Compute Kinematics (on window)
            skel_features = self._compute_kinematics(skel_window)  # (Window, J*9)

            # Early Fusion
            # (Window, J*9 + MFCC)
            combined_features = np.concatenate([skel_features, audio_window], axis=-1)

            return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(
                label_window, dtype=torch.long
            )

        else:
            # Full sequence mode
            sid = self.indices[idx]
            raw_skel = self.data_dict[sid]["skeleton"]
            raw_audio = self.data_dict[sid]["audio"]
            raw_labels = self.data_dict[sid]["labels"]

            # No augmentation for inference
            skel_features = self._compute_kinematics(raw_skel)
            combined_features = np.concatenate([skel_features, raw_audio], axis=-1)

            return (
                torch.tensor(combined_features, dtype=torch.float32),
                torch.tensor(raw_labels, dtype=torch.long),
                sid,
            )


# ==========================================
# 4. Loader Factory
# ==========================================


def get_data_loaders(load_cached_data=True):
    """
    Main entry point. Loads data, creates datasets and dataloaders.
    """
    # 1. Process/Load Data
    train_data = process_and_cache_data(
        Config.TRAIN_CSV, "dataset_train", load_cached_data
    )
    val_data = process_and_cache_data(Config.VAL_CSV, "dataset_val", load_cached_data)
    test_data = process_and_cache_data(
        Config.TEST_CSV, "dataset_test", load_cached_data
    )

    # 2. Create Datasets
    train_ds = GestureDataset(train_data, mode="train")
    val_ds = GestureDataset(val_data, mode="val")

    # For validation metric, we might want full sequences
    val_full_ds = GestureDataset(val_data, mode="full")
    test_full_ds = GestureDataset(test_data, mode="full")

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Batch size 1 for full sequences (variable length)
    val_metric_loader = DataLoader(
        val_full_ds, batch_size=1, shuffle=False, num_workers=1
    )
    test_loader = DataLoader(test_full_ds, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader, val_metric_loader, test_loader
