import os
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import scipy.io
import torchaudio
import warnings

# Suppress warnings from torchaudio or numpy
warnings.filterwarnings("ignore")

# ==========================================
# Constants
# ==========================================
JOINTS_COUNT = 20
AUDIO_MFCC_N = 13
CACHE_DIR_DEFAULT = "./working/idea_18"

# ==========================================
# Helper Functions
# ==========================================


def parse_mat_polymorphic(mat_path):
    """
    Parses the .mat file to extract skeleton data safely, handling polymorphic structures.
    Returns: numpy array of shape (T, 20, 3) or None if failed.
    """
    try:
        # Load mat file: struct_as_record=False converts structs to objects
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        if "Video" not in mat:
            return None

        video = mat["Video"]

        # Get number of frames
        num_frames = 0
        if hasattr(video, "NumFrames"):
            num_frames = int(video.NumFrames)

        if num_frames == 0:
            return None

        # Initialize skeleton array (T, 20, 3)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        if not hasattr(video, "Frames"):
            return skeleton_data

        frames = video.Frames

        # Handle case where Frames is a single object vs list/array
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        # Iterate over frames safely
        loop_len = min(len(frames), num_frames)

        for i in range(loop_len):
            frame = frames[i]

            # Check if Skeleton exists
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Check if Skeleton is valid (sometimes it's 0, empty, or None)
            if isinstance(skel, (int, float)) or skel is None:
                continue

            # Check for WorldPosition
            if not hasattr(skel, "WorldPosition"):
                continue

            wp = skel.WorldPosition

            # Extract coordinates safely
            try:
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    x = np.array(wp.X, dtype=np.float32).flatten()
                    y = np.array(wp.Y, dtype=np.float32).flatten()
                    z = np.array(wp.Z, dtype=np.float32).flatten()

                    # Ensure we have data for 20 joints
                    if (
                        x.size == JOINTS_COUNT
                        and y.size == JOINTS_COUNT
                        and z.size == JOINTS_COUNT
                    ):
                        skeleton_data[i, :, 0] = x
                        skeleton_data[i, :, 1] = y
                        skeleton_data[i, :, 2] = z
            except Exception:
                continue

        return skeleton_data

    except Exception:
        return None


def process_audio(audio_path, target_num_frames, n_mfcc=AUDIO_MFCC_N):
    """
    Loads audio, extracts MFCCs, and aligns them to video frames via interpolation.
    Returns: numpy array (T, n_mfcc)
    """
    try:
        if not os.path.exists(audio_path):
            return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

        waveform, sample_rate = torchaudio.load(audio_path)

        # MFCC Configuration
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (channels, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # Average over channels -> (n_mfcc, time)
        mfcc = mfcc.permute(1, 0)  # (time, n_mfcc)

        mfcc_np = mfcc.detach().numpy()

        # Interpolate to match target_num_frames
        curr_frames = mfcc_np.shape[0]
        if curr_frames == 0:
            return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

        if curr_frames != target_num_frames:
            # Linear interpolation for each coefficient
            x_old = np.linspace(0, 1, curr_frames)
            x_new = np.linspace(0, 1, target_num_frames)

            new_mfcc = np.zeros((target_num_frames, n_mfcc), dtype=np.float32)
            for c in range(n_mfcc):
                new_mfcc[:, c] = np.interp(x_new, x_old, mfcc_np[:, c])
            return new_mfcc

        return mfcc_np

    except Exception:
        return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from positions.
    Input: (T, J, 3)
    Output: (T, J, 9) -> [Pos, Vel, Acc]
    """
    velocity = np.zeros_like(positions)
    acceleration = np.zeros_like(positions)

    # Velocity: P(t) - P(t-1)
    if positions.shape[0] > 1:
        velocity[1:] = positions[1:] - positions[:-1]

    # Acceleration: V(t) - V(t-1)
    if velocity.shape[0] > 1:
        acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([positions, velocity, acceleration], axis=2)


def augment_skeleton(positions):
    """
    Applies random rotation (Y-axis) and scaling.
    Input: (T, J, 3)
    """
    # Random Scale (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)
    positions = positions * scale

    # Random Rotation around Y-axis (-30 to +30 degrees)
    theta = np.random.uniform(-np.pi / 6, np.pi / 6)
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix for Y-axis
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Apply rotation: P_rot = P @ R.T
    T, J, _ = positions.shape
    flat_pos = positions.reshape(-1, 3)
    rot_pos = np.dot(flat_pos, R.T)

    return rot_pos.reshape(T, J, 3)


def load_dataset_data(metadata_file, cache_dir, load_cached, split_name):
    """
    Loads data from cache or processes from scratch.
    Uses flattened arrays to avoid pickle issues.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"dataset_{split_name}.npz")

    if load_cached and os.path.exists(cache_path):
        try:
            data_dict = np.load(cache_path)

            # Unpack flattened arrays
            flat_skeletons = data_dict["flat_skeletons"]
            flat_audios = data_dict["flat_audios"]
            flat_labels = data_dict["flat_labels"]
            indices = data_dict["indices"]  # (N, 2) -> [start, len]
            sample_ids_arr = data_dict["sample_ids"]

            samples = []
            for i in range(len(indices)):
                start, length = indices[i]
                end = start + length

                samples.append(
                    {
                        "sample_id": str(sample_ids_arr[i]),
                        "skeleton": flat_skeletons[start:end].reshape(length, 20, 3),
                        "audio": flat_audios[start:end],
                        "labels": flat_labels[start:end],
                    }
                )
            return samples
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing {split_name}...")

    # Process from scratch
    df = pd.read_csv(metadata_file)

    samples = []

    for idx, row in df.iterrows():
        base_dir = "./input"
        mat_path = os.path.join(base_dir, row["data_path"])
        audio_path = os.path.join(base_dir, row["audio_path"])

        # Parse Skeleton
        skeleton = parse_mat_polymorphic(mat_path)
        if skeleton is None:
            continue

        num_frames = skeleton.shape[0]

        # Parse Audio
        audio = process_audio(audio_path, num_frames)

        # Parse Labels
        label_seq = np.zeros(num_frames, dtype=np.int64)
        if "labels" in row and isinstance(row["labels"], str):
            try:
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # Labels are 1-20. 0 is background.
                    # Metadata 'begin'/'end' are 1-based usually in MATLAB, but let's check.
                    # Assuming 1-based from MATLAB export description.
                    start = max(0, int(l["begin"]) - 1)
                    end = min(num_frames, int(l["end"]))
                    lid = int(l["id"])
                    label_seq[start:end] = lid
            except:
                pass

        samples.append(
            {
                "sample_id": row["sample_id"],
                "skeleton": skeleton,
                "audio": audio,
                "labels": label_seq,
            }
        )

    # Flatten for caching
    all_skeletons = []
    all_audios = []
    all_labels = []
    indices = []
    sample_ids = []

    current_idx = 0
    for s in samples:
        length = s["skeleton"].shape[0]
        all_skeletons.append(
            s["skeleton"].reshape(length, -1)
        )  # Flatten spatial dims temporarily
        all_audios.append(s["audio"])
        all_labels.append(s["labels"])
        indices.append([current_idx, length])
        sample_ids.append(s["sample_id"])
        current_idx += length

    if len(samples) > 0:
        flat_skeletons = np.concatenate(all_skeletons, axis=0)
        flat_audios = np.concatenate(all_audios, axis=0)
        flat_labels = np.concatenate(all_labels, axis=0)
        indices_arr = np.array(indices, dtype=np.int64)
        sample_ids_arr = np.array(sample_ids)

        np.savez_compressed(
            cache_path,
            flat_skeletons=flat_skeletons,
            flat_audios=flat_audios,
            flat_labels=flat_labels,
            indices=indices_arr,
            sample_ids=sample_ids_arr,
        )

    return samples


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(data.Dataset):
    def __init__(
        self,
        metadata_file,
        split="train",
        window_size=64,
        stride=32,
        cache_dir=CACHE_DIR_DEFAULT,
        load_cached=True,
        augment=False,
    ):
        """
        Dataset for multimodal gesture recognition.
        Args:
            metadata_file: Path to CSV metadata.
            split: 'train', 'val', or 'test'.
            window_size: Size of sliding window.
            stride: Stride for sliding window.
            cache_dir: Directory to store/load processed numpy files.
            load_cached: Whether to try loading from cache.
            augment: Whether to apply kinematic augmentation.
        """
        self.window_size = window_size
        self.augment = augment
        self.split = split

        # Load processed samples
        self.samples = load_dataset_data(metadata_file, cache_dir, load_cached, split)

        # Generate sliding windows
        self.windows = []
        for i, sample in enumerate(self.samples):
            num_frames = sample["skeleton"].shape[0]

            if num_frames < window_size:
                # If sequence is shorter than window, take one window starting at 0 (will be padded)
                self.windows.append((i, 0))
            else:
                # Sliding window
                step = stride if stride > 0 else window_size

                # For training, strict sliding
                # For val/test, we ensure we cover the end
                for start in range(0, num_frames - window_size + 1, step):
                    self.windows.append((i, start))

                # Ensure coverage of the last frame for evaluation splits
                if split in ["val", "test"]:
                    last_start = max(0, num_frames - window_size)
                    if not self.windows or self.windows[-1][1] != last_start:
                        self.windows.append((i, last_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample_idx, start_frame = self.windows[index]
        sample = self.samples[sample_idx]

        skel_raw = sample["skeleton"]  # (T, 20, 3)
        audio_raw = sample["audio"]  # (T, 13)
        labels_raw = sample["labels"]  # (T,)

        total_frames = skel_raw.shape[0]

        # Extract window
        if total_frames < self.window_size:
            # Pad with edge values for skeleton, zeros for audio/labels
            pad_len = self.window_size - total_frames

            # Pad Skeleton (edge padding)
            skel_win = np.pad(skel_raw, ((0, pad_len), (0, 0), (0, 0)), mode="edge")

            # Pad Audio (constant 0)
            audio_win = np.pad(audio_raw, ((0, pad_len), (0, 0)), mode="constant")

            # Pad Labels (constant 0 - background)
            labels_win = np.pad(
                labels_raw, (0, pad_len), mode="constant", constant_values=0
            )
        else:
            end_frame = start_frame + self.window_size
            skel_win = skel_raw[start_frame:end_frame]
            audio_win = audio_raw[start_frame:end_frame]
            labels_win = labels_raw[start_frame:end_frame]

        # Apply Augmentation (on raw positions before kinematics)
        if self.augment:
            skel_win = augment_skeleton(skel_win)

        # Compute Kinematics: (W, 20, 9)
        skel_kin = compute_kinematics(skel_win)

        # Flatten Skeleton: (W, 180)
        skel_flat = skel_kin.reshape(self.window_size, -1)

        # Early Fusion: Concatenate Audio (W, 180 + 13) = (W, 193)
        features = np.concatenate([skel_flat, audio_win], axis=1)

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_win, dtype=torch.long)

        return features_tensor, labels_tensor
