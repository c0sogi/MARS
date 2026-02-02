import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from library.config import Config

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class PolymorphicSkeletonParser:
    """
    Robustly parses .mat files to extract skeleton data, handling
    inconsistencies between struct arrays, cell arrays, and objects.
    """

    @staticmethod
    def parse(mat_path, num_frames_expected):
        try:
            # Load mat file with squeeze_me=True to simplify structures
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            skeleton_data = np.zeros(
                (num_frames_expected, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            if "Video" not in mat.__dict__:
                return skeleton_data

            video = mat.Video
            if not hasattr(video, "Frames"):
                return skeleton_data

            frames = video.Frames

            # Handle case where Frames is a single object or scalar
            if not isinstance(frames, (np.ndarray, list)):
                frames = [frames]

            # Determine actual number of frames to process
            num_frames_to_read = min(len(frames), num_frames_expected)

            for i in range(num_frames_to_read):
                frame_obj = frames[i]

                # Check if Skeleton exists
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel = frame_obj.Skeleton

                # Check if WorldPosition exists
                if not hasattr(skel, "WorldPosition"):
                    continue

                wp = skel.WorldPosition

                # Extract coordinates based on structure type
                # Case A: WorldPosition is a struct with X, Y, Z fields
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # X, Y, Z can be scalars or arrays (though usually scalars for joints?
                    # Actually usually WorldPosition is 20x1 struct array or similar)
                    # In this dataset, WorldPosition is often a struct of arrays or array of structs
                    pass

                # The dataset description says WorldPosition has X, Y, Z.
                # However, usually for 20 joints, it's an array of structs or struct of arrays.
                # Let's try to interpret `wp` as a 20x3 matrix or similar.

                # Robust extraction strategy:
                # 1. Try to get direct 20x3 array
                if isinstance(wp, np.ndarray) and wp.shape == (Config.NUM_JOINTS, 3):
                    skeleton_data[i] = wp
                    continue

                # 2. If wp is a struct/object, it might be a single joint?
                # Actually, the Skeleton is usually an array of 20 joints.
                # Let's look at the `skel` object again.
                # Usually `skel` is the Skeleton Frame, containing `JointsType` etc.
                # If `skel` is a single object, maybe it has arrays.

                # Let's assume `skel` might be an array of 20 objects (one per joint)
                # OR `skel` has `WorldPosition` which is an array of 20 objects.

                # Re-reading description: "Skeleton Frame... contains joint positions... Skeleton structure... WorldPosition"
                # It implies Skeleton is a struct containing WorldPosition.

                # Let's try to extract from the `skel` object directly if it is iterable (20 joints)
                joints_found = False
                if isinstance(skel, np.ndarray) and len(skel) == Config.NUM_JOINTS:
                    for j in range(Config.NUM_JOINTS):
                        joint = skel[j]
                        if hasattr(joint, "WorldPosition"):
                            p = joint.WorldPosition
                            if hasattr(p, "X") and hasattr(p, "Y") and hasattr(p, "Z"):
                                skeleton_data[i, j, 0] = p.X
                                skeleton_data[i, j, 1] = p.Y
                                skeleton_data[i, j, 2] = p.Z
                    joints_found = True

                if joints_found:
                    continue

                # Fallback: Maybe WorldPosition is the array inside Skeleton
                if hasattr(wp, "shape") and wp.shape == (Config.NUM_JOINTS, 3):
                    skeleton_data[i] = wp
                    continue

            # Fill missing frames (zeros) with nearest valid frame or interpolation
            # Simple forward fill then backward fill
            valid_mask = np.any(skeleton_data != 0, axis=(1, 2))
            if np.any(valid_mask):
                # Indices of valid frames
                valid_indices = np.where(valid_mask)[0]

                # Forward fill
                for idx in range(num_frames_expected):
                    if not valid_mask[idx]:
                        # Find nearest valid
                        nearest = valid_indices[np.abs(valid_indices - idx).argmin()]
                        skeleton_data[idx] = skeleton_data[nearest]

            return skeleton_data

        except Exception as e:
            # Return zeros on catastrophic failure
            return np.zeros(
                (num_frames_expected, Config.NUM_JOINTS, 3), dtype=np.float32
            )


class AudioProcessor:
    """
    Handles audio loading and MFCC extraction, ensuring alignment with video frames.
    """

    @staticmethod
    def process(audio_path, target_num_frames):
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            # We use standard settings, can be tuned.
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.NUM_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, time)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = torch.mean(mfcc, dim=0, keepdim=True)

            # Shape is now (1, n_mfcc, time_steps)
            # We need (target_num_frames, n_mfcc)

            # Interpolate to match video frames
            # Input to interpolate needs to be (Batch, Channels, Time)
            # Here Batch=1, Channels=n_mfcc, Time=original_time
            mfcc = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )

            # Permute to (Time, Features) -> (target_num_frames, n_mfcc)
            mfcc = mfcc.squeeze(0).permute(1, 0)

            return mfcc.numpy()

        except Exception:
            # Return zeros if audio fails
            return np.zeros((target_num_frames, Config.NUM_MFCC), dtype=np.float32)


class KinematicAugmentor:
    """
    Handles 3D augmentation and kinematic derivative computation.
    """

    @staticmethod
    def augment(skeleton_pos):
        """
        Args:
            skeleton_pos: (T, J, 3) numpy array
        Returns:
            augmented_pos: (T, J, 3) numpy array
        """
        # Random Rotation around Y-axis
        theta = np.random.uniform(-30, 30)  # degrees
        r = R.from_euler("y", theta, degrees=True)
        rot_matrix = r.as_matrix()  # (3, 3)

        # Reshape for matmul: (T*J, 3)
        T, J, C = skeleton_pos.shape
        flat_pos = skeleton_pos.reshape(-1, 3)

        # Apply rotation
        aug_pos = np.dot(flat_pos, rot_matrix.T)

        # Random Scaling (0.85 to 1.15)
        scale = np.random.uniform(0.85, 1.15)
        aug_pos = aug_pos * scale

        return aug_pos.reshape(T, J, C).astype(np.float32)

    @staticmethod
    def compute_kinematics(skeleton_pos):
        """
        Computes Velocity and Acceleration.
        Args:
            skeleton_pos: (T, J, 3)
        Returns:
            features: (T, J*3*3) -> (Pos, Vel, Acc) flattened
        """
        # Velocity: P(t) - P(t-1)
        # Pad first frame
        vel = np.diff(skeleton_pos, axis=0, prepend=skeleton_pos[0:1])

        # Acceleration: V(t) - V(t-1)
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        # Flatten joints and coordinates
        # Input: (T, 20, 3) -> Output (T, 60)
        T = skeleton_pos.shape[0]
        pos_flat = skeleton_pos.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)
        acc_flat = acc.reshape(T, -1)

        # Concatenate: (T, 180)
        return np.concatenate([pos_flat, vel_flat, acc_flat], axis=1).astype(np.float32)


class GestureDataset(Dataset):
    def __init__(self, data_list, augment=False, stride=Config.STRIDE):
        """
        Args:
            data_list: List of dicts {'skeleton': (T, J, 3), 'audio': (T, 13), 'labels': (T,), 'id': str}
            augment: Boolean, whether to apply kinematic augmentation
            stride: Stride for sliding window
        """
        self.data_list = data_list
        self.augment = augment
        self.window_size = Config.WINDOW_SIZE
        self.stride = stride

        # Pre-calculate window indices
        self.window_indices = []
        for seq_idx, sample in enumerate(self.data_list):
            num_frames = sample["skeleton"].shape[0]
            if num_frames < self.window_size:
                # Pad short sequences later or skip?
                # Better to include at least one window padded
                self.window_indices.append((seq_idx, 0))
            else:
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.window_indices.append((seq_idx, start))

                # Ensure last frame is covered if not exact fit
                last_start = num_frames - self.window_size
                if last_start > 0 and (last_start % self.stride != 0):
                    self.window_indices.append((seq_idx, last_start))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.window_indices[idx]
        sample = self.data_list[seq_idx]

        # Extract Raw Data
        skel_full = sample["skeleton"]  # (T, 20, 3)
        audio_full = sample["audio"]  # (T, 13)
        labels_full = sample["labels"]  # (T,)

        # Handle Padding for short sequences
        total_frames = skel_full.shape[0]
        if total_frames < self.window_size:
            pad_len = self.window_size - total_frames
            # Pad skeleton with last frame
            skel_window = np.pad(skel_full, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            # Pad audio with zeros
            audio_window = np.pad(audio_full, ((0, pad_len), (0, 0)), mode="constant")
            # Pad labels with background (0)
            labels_window = np.pad(
                labels_full, (0, pad_len), mode="constant", constant_values=0
            )
        else:
            end_frame = start_frame + self.window_size
            skel_window = skel_full[start_frame:end_frame]
            audio_window = audio_full[start_frame:end_frame]
            labels_window = labels_full[start_frame:end_frame]

        # Augmentation (On Raw Skeleton)
        if self.augment:
            skel_window = KinematicAugmentor.augment(skel_window)

        # Compute Kinematics (Pos, Vel, Acc)
        # Shape: (Window, 180)
        kinematics = KinematicAugmentor.compute_kinematics(skel_window)

        # Early Fusion: Concat Kinematics + Audio
        # Shape: (Window, 180 + 13) = (64, 193)
        features = np.concatenate([kinematics, audio_window], axis=1)

        # Convert to Torch Tensors
        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels_window).long()

        return features_tensor, labels_tensor


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for train, val, or test splits.
    Handles caching to .npz files to avoid re-processing.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{mode}.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct list of dicts
            sample_ids = loaded["sample_ids"]
            data_list = []
            for i, sid in enumerate(sample_ids):
                data_list.append(
                    {
                        "id": str(sid),
                        "skeleton": loaded[f"skel_{i}"],
                        "audio": loaded[f"audio_{i}"],
                        "labels": loaded[f"label_{i}"],
                    }
                )
            return data_list
        except Exception as e:
            print(f"Cache load failed: {e}. Re-processing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    if mode == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df = pd.read_csv(meta_path)

    processed_data = []

    # Pre-allocate dicts for saving to npz
    save_dict = {}
    sample_ids_list = []

    for idx, row in df.iterrows():
        sid = row["sample_id"]

        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Parse Skeleton to get frame count and raw positions
        # We need to peek at the mat file to get NumFrames usually,
        # but Parser handles it if we pass a large enough number or check metadata.
        # Let's trust the parser to find frames.
        # However, to initialize the array in parser, we need a guess.
        # We can use a large buffer or read 'NumFrames' from mat first.
        try:
            mat_info = scipy.io.loadmat(
                mat_path, squeeze_me=True, struct_as_record=False
            )
            num_frames = int(mat_info["Video"].NumFrames)
        except:
            num_frames = 3000  # Fallback buffer

        skeleton = PolymorphicSkeletonParser.parse(mat_path, num_frames)

        # 2. Process Audio
        audio = AudioProcessor.process(audio_path, num_frames)

        # 3. Create Dense Labels
        labels = np.zeros(num_frames, dtype=np.int64)
        if mode != "test":
            label_list = json.loads(row["labels"])
            for l in label_list:
                lid = int(l["id"])
                start = int(l["begin"]) - 1  # 1-based to 0-based
                end = int(l["end"])  # inclusive in matlab usually
                # Clip to valid range
                start = max(0, start)
                end = min(num_frames, end)
                if start < end:
                    labels[start:end] = lid

        # Store
        entry = {"id": sid, "skeleton": skeleton, "audio": audio, "labels": labels}
        processed_data.append(entry)

        # Prepare for cache saving
        save_dict[f"skel_{idx}"] = skeleton
        save_dict[f"audio_{idx}"] = audio
        save_dict[f"label_{idx}"] = labels
        sample_ids_list.append(sid)

    # Save to Cache
    save_dict["sample_ids"] = np.array(sample_ids_list)
    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return processed_data


def get_data_loaders(batch_size=Config.BATCH_SIZE):
    """
    Factory function to create DataLoaders for training and validation.
    """
    # Load Data
    train_data = load_data("train")
    val_data = load_data("val")

    # Create Datasets
    # Train: Augmentation ON, Stride = Config.STRIDE
    train_dataset = GestureDataset(train_data, augment=True, stride=Config.STRIDE)

    # Val: Augmentation OFF, Stride = Config.STRIDE (or larger/smaller depending on eval strategy)
    # For monitoring loss, we use sliding windows.
    val_dataset = GestureDataset(val_data, augment=False, stride=Config.STRIDE)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
