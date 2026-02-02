import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library import config

# Set fixed seeds for reproducibility
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# Joint mapping based on the provided dataset description
JOINTS_ORDER = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]


def load_skeleton_polymorphic(mat_path):
    """
    Robustly parses .mat files to extract skeleton data, handling various
    MATLAB struct/cell array inconsistencies.
    Returns: Numpy array (NumFrames, 20, 3)
    """
    try:
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if "Video" not in mat:
            return None

        video = mat["Video"]
        # Unwrap 0-d array if necessary (scipy.io behavior with squeeze_me=True)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        # Initialize skeleton array: (T, Joints, 3)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Handle cases where Frames might be a single object or an array
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        last_valid_frame = np.zeros((20, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            if t >= num_frames:
                break

            try:
                # Check if Skeleton exists and is not empty/zero
                skel = getattr(frame, "Skeleton", None)

                # Handle case where Skeleton might be a list/array (multi-user) or single obj
                if isinstance(skel, (list, np.ndarray)) and len(skel) > 0:
                    skel = skel[0]  # Take first user

                if skel is None or (isinstance(skel, (int, float)) and skel == 0):
                    # Missing skeleton, use last valid
                    skeleton_data[t] = last_valid_frame
                    continue

                # Extract WorldPosition
                # It might be an object with X,Y,Z or a struct array
                wp = getattr(skel, "WorldPosition", None)

                if wp is None:
                    skeleton_data[t] = last_valid_frame
                    continue

                # Check if Joints are accessible by name or index
                # The dataset description implies a specific structure.
                # However, scipy.io parsing of structs can be tricky.
                # We try to extract 20 joints.

                # Strategy A: wp is an array of structs (one per joint)
                # Strategy B: wp is a single struct with arrays?
                # Usually in this dataset, Skeleton contains JointsType and WorldPosition arrays.

                # Let's try to infer structure.
                # If wp is an array of length 20
                if isinstance(wp, (list, np.ndarray)) and len(wp) == 20:
                    for j in range(20):
                        joint_pos = wp[j]
                        # joint_pos might be an object with X,Y,Z or array
                        if hasattr(joint_pos, "X"):
                            skeleton_data[t, j, 0] = joint_pos.X
                            skeleton_data[t, j, 1] = joint_pos.Y
                            skeleton_data[t, j, 2] = joint_pos.Z
                        elif (
                            isinstance(joint_pos, (list, np.ndarray))
                            and len(joint_pos) >= 3
                        ):
                            skeleton_data[t, j] = joint_pos[:3]
                else:
                    # Fallback or specific object structure
                    # Sometimes wp itself has X, Y, Z arrays? Unlikely for this dataset format.
                    # Assume failure if not array of 20
                    skeleton_data[t] = last_valid_frame
                    continue

                last_valid_frame = skeleton_data[t].copy()

            except Exception:
                # On any parsing error for a frame, copy previous
                skeleton_data[t] = last_valid_frame

        return skeleton_data

    except Exception as e:
        # print(f"Error parsing {mat_path}: {e}")
        return None


def load_audio_aligned(audio_path, target_num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    Returns: Numpy array (NumFrames, 13)
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=config.AUDIO_MFCC_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # Average over channels if stereo -> (n_mfcc, time)

        # Align to video frames using interpolation
        # Input to interpolate must be (Batch, Channels, Time)
        mfcc = mfcc.unsqueeze(0)

        aligned_mfcc = F.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Remove batch dim and transpose to (Time, Features)
        aligned_mfcc = aligned_mfcc.squeeze(0).transpose(0, 1)

        return aligned_mfcc.numpy()

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_num_frames, config.AUDIO_MFCC_N_MFCC), dtype=np.float32)


def process_metadata(metadata_path, subset_name, load_cached_data=True):
    """
    Loads metadata and processes/caches the raw aligned data.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"dataset_{subset_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached {subset_name} data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "sample_ids": data["sample_ids"],
                "skeletons": data["skeletons"],
                "audios": data["audios"],
                "labels": data["labels"],
            }
        except Exception:
            pass  # Fallback to reprocessing

    # Load metadata CSV
    df = pd.read_csv(metadata_path)

    # Debugging subset
    if config.DEBUG:
        df = df.head(config.DEBUG_SUBSET_SIZE)

    sample_ids = []
    skeletons = []
    audios = []
    labels_list = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton
        skel_data = load_skeleton_polymorphic(data_path)
        if skel_data is None:
            # Skip corrupted samples or handle?
            # For this challenge, we should try to keep all samples.
            # Create dummy if failed?
            # Let's assume most load correctly. If not, we skip to avoid crashing training.
            continue

        num_frames = skel_data.shape[0]

        # 2. Load Audio
        audio_data = load_audio_aligned(audio_path, num_frames)

        # 3. Create Labels
        label_seq = np.zeros(num_frames, dtype=np.int64)  # 0 = Background

        if "labels" in row and isinstance(row["labels"], str):
            try:
                anns = json.loads(row["labels"])
                for ann in anns:
                    start = max(0, ann["begin"] - 1)  # 1-based to 0-based
                    end = min(num_frames, ann["end"])
                    lid = ann["id"]
                    if 1 <= lid <= 20:
                        label_seq[start:end] = lid
            except:
                pass

        sample_ids.append(sample_id)
        skeletons.append(skel_data)
        audios.append(audio_data)
        labels_list.append(label_seq)

    # Save to cache
    # Use object arrays for variable length sequences
    skeletons_arr = np.array(skeletons, dtype=object)
    audios_arr = np.array(audios, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)
    sample_ids_arr = np.array(sample_ids)

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_ids=sample_ids_arr,
        skeletons=skeletons_arr,
        audios=audios_arr,
        labels=labels_arr,
    )

    return {
        "sample_ids": sample_ids_arr,
        "skeletons": skeletons_arr,
        "audios": audios_arr,
        "labels": labels_arr,
    }


class GestureDataset(Dataset):
    def __init__(
        self,
        data_dict,
        augment=False,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
    ):
        self.sample_ids = data_dict["sample_ids"]
        self.skeletons = data_dict["skeletons"]
        self.audios = data_dict["audios"]
        self.labels = data_dict["labels"]
        self.augment = augment
        self.window_size = window_size

        # Pre-calculate windows
        self.indices = []
        for i, skel in enumerate(self.skeletons):
            n_frames = skel.shape[0]
            # Generate sliding windows
            # If sequence is shorter than window, pad?
            # We skip short sequences or pad. Given dataset stats (min ~40 frames),
            # most are close to window size. We'll pad in __getitem__ if needed.
            if n_frames < window_size:
                self.indices.append((i, 0))  # Single window, will be padded
            else:
                for start in range(0, n_frames - window_size + 1, stride):
                    self.indices.append((i, start))
                # Ensure last frames are covered
                if (n_frames - window_size) % stride != 0:
                    self.indices.append((i, n_frames - window_size))

    def __len__(self):
        return len(self.indices)

    def augment_skeleton(self, skeleton):
        """
        Applies random Y-axis rotation and uniform scaling.
        Skeleton: (T, 20, 3)
        """
        # Random Rotation around Y-axis
        theta = np.random.uniform(-0.3, 0.3)  # Radians (~ +/- 17 degrees)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: (T*J, 3) @ R.T -> (T*J, 3)
        shape = skeleton.shape
        flat_skel = skeleton.reshape(-1, 3)
        rotated_skel = np.dot(flat_skel, R.T)

        # Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        scaled_skel = rotated_skel * scale

        return scaled_skel.reshape(shape)

    def compute_kinematics(self, position):
        """
        Computes Velocity and Acceleration.
        Input: (T, 20, 3)
        Output: (T, 20, 9) [Pos, Vel, Acc]
        """
        # Velocity: P[t] - P[t-1]
        # Pad first frame with 0 velocity (replicate position or 0 diff)
        # Using 0 diff (pad with 0 at dim 0)
        vel = np.diff(position, axis=0, prepend=position[0:1])

        # Acceleration: V[t] - V[t-1]
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        return np.concatenate([position, vel, acc], axis=2)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.indices[idx]

        raw_skel = self.skeletons[seq_idx]  # (T_seq, 20, 3)
        raw_audio = self.audios[seq_idx]  # (T_seq, 13)
        raw_label = self.labels[seq_idx]  # (T_seq,)

        seq_len = raw_skel.shape[0]

        # Handle padding for short sequences
        if seq_len < self.window_size:
            # Pad end with zeros/last value
            pad_len = self.window_size - seq_len

            # Skeleton: Repeat last frame
            last_skel = raw_skel[-1:]
            skel_pad = np.repeat(last_skel, pad_len, axis=0)
            win_skel = np.concatenate([raw_skel, skel_pad], axis=0)

            # Audio: Zero pad
            audio_pad = np.zeros((pad_len, raw_audio.shape[1]), dtype=raw_audio.dtype)
            win_audio = np.concatenate([raw_audio, audio_pad], axis=0)

            # Label: Pad with background (0)
            label_pad = np.zeros(pad_len, dtype=raw_label.dtype)
            win_label = np.concatenate([raw_label, label_pad], axis=0)

        else:
            end_frame = start_frame + self.window_size
            win_skel = raw_skel[start_frame:end_frame]
            win_audio = raw_audio[start_frame:end_frame]
            win_label = raw_label[start_frame:end_frame]

        # 1. Augmentation (Training Only)
        if self.augment:
            win_skel = self.augment_skeleton(win_skel)

        # 2. Compute Kinematics (on potentially augmented data)
        # (T, 20, 3) -> (T, 20, 9)
        kinematics = self.compute_kinematics(win_skel)

        # 3. Global Linear Scaling (mm -> m)
        # Apply to all components (Pos, Vel, Acc) to maintain hierarchy
        kinematics = kinematics * config.SKELETON_SCALE_FACTOR

        # 4. Flatten Skeleton: (T, 20, 9) -> (T, 180)
        feat_skel = kinematics.reshape(self.window_size, -1)

        # 5. Early Fusion: Concat Audio
        # (T, 180) + (T, 13) -> (T, 193)
        features = np.concatenate([feat_skel, win_audio], axis=1)

        # Convert to Float Tensor
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(win_label, dtype=torch.long)

        return features, targets


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get dataloaders.
    """
    # Load raw data dictionaries
    train_data = process_metadata(config.TRAIN_METADATA_PATH, "train", load_cached_data)
    val_data = process_metadata(config.VAL_METADATA_PATH, "val", load_cached_data)
    test_data = process_metadata(config.TEST_METADATA_PATH, "test", load_cached_data)

    # Create Datasets
    train_ds = GestureDataset(train_data, augment=True)
    val_ds = GestureDataset(val_data, augment=False)
    test_ds = GestureDataset(
        test_data, augment=False, stride=config.WINDOW_SIZE // 2
    )  # 50% overlap for test

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ds.sample_ids, test_ds.indices
