import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger(name="DataProcessing")


class DataProcessor:
    """
    Handles loading, processing, alignment, and caching of multimodal data.
    """

    # Kinect v2 Joint Indices
    JOINTS = {
        "HipCenter": 0,
        "Spine": 1,
        "ShoulderCenter": 2,
        "Head": 3,
        "ShoulderLeft": 4,
        "ElbowLeft": 5,
        "WristLeft": 6,
        "HandLeft": 7,
        "ShoulderRight": 8,
        "ElbowRight": 9,
        "WristRight": 10,
        "HandRight": 11,
        "HipLeft": 12,
        "KneeLeft": 13,
        "AnkleLeft": 14,
        "FootLeft": 15,
        "HipRight": 16,
        "KneeRight": 17,
        "AnkleRight": 18,
        "FootRight": 19,
    }

    def __init__(self):
        self.num_joints = 20
        self.joint_dim = 3

    def load_mat_skeleton(self, mat_path):
        """Parses .mat file to extract skeleton frames (T, 20, 3)."""
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None

            video = mat["Video"]
            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames
            num_frames = len(frames)
            skeleton_data = np.zeros(
                (num_frames, self.num_joints, self.joint_dim), dtype=np.float32
            )

            for i, frame in enumerate(frames):
                if hasattr(frame, "Skeleton") and hasattr(
                    frame.Skeleton, "WorldPosition"
                ):
                    # WorldPosition is usually 20x1 struct array or similar
                    # We need to robustly extract X, Y, Z
                    wp = frame.Skeleton.WorldPosition

                    # Handle different mat structures
                    if isinstance(wp, (list, np.ndarray)):
                        # Case 1: Numeric Matrix (20, 3) or (20, 4)
                        if (
                            isinstance(wp, np.ndarray)
                            and wp.ndim == 2
                            and wp.shape[0] == self.num_joints
                            and wp.shape[1] >= 3
                        ):
                            skeleton_data[i] = wp[:, :3]

                        # Case 2: Array of objects or arrays (length 20)
                        elif len(wp) == self.num_joints:
                            for j in range(self.num_joints):
                                joint = wp[j]

                                # Sub-case 2a: Joint is an object with attributes (Struct)
                                if hasattr(joint, "X"):
                                    skeleton_data[i, j, 0] = joint.X
                                    skeleton_data[i, j, 1] = joint.Y
                                    skeleton_data[i, j, 2] = joint.Z

                                # Sub-case 2b: Joint is a numeric array/vector (e.g. [x, y, z])
                                elif isinstance(joint, np.ndarray) and joint.size >= 3:
                                    # Flatten to handle shape (3,) or (1, 3) or (3, 1)
                                    flat_j = joint.flatten()
                                    skeleton_data[i, j, 0] = flat_j[0]
                                    skeleton_data[i, j, 1] = flat_j[1]
                                    skeleton_data[i, j, 2] = flat_j[2]
                    else:
                        # Fallback or specific handling if structure differs
                        pass

            return skeleton_data
        except Exception as e:
            logger.warning(f"Failed to load skeleton from {mat_path}: {e}")
            return None

    def align_skeleton(self, skeleton_frames):
        """
        Performs Canonical View Alignment.
        Rotates skeleton so HipLeft -> HipRight vector aligns with global X-axis.
        Centers skeleton on HipCenter.
        """
        # Indices
        idx_hc = self.JOINTS["HipCenter"]
        idx_hl = self.JOINTS["HipLeft"]
        idx_hr = self.JOINTS["HipRight"]

        aligned_frames = np.zeros_like(skeleton_frames)

        for t in range(skeleton_frames.shape[0]):
            frame = skeleton_frames[t]  # (20, 3)

            # 1. Center at HipCenter
            center = frame[idx_hc].copy()
            centered_frame = frame - center

            # 2. Calculate Rotation Angle
            # Vector from Left Hip to Right Hip
            hip_vector = centered_frame[idx_hr] - centered_frame[idx_hl]
            # We want to align this vector to the X-axis in the XZ plane
            # angle = atan2(z, x)
            theta = np.arctan2(hip_vector[2], hip_vector[0])

            # Rotation Matrix around Y-axis (vertical) to undo theta
            # We rotate by -theta to align to X-axis
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

            # Apply rotation
            # (20, 3) @ (3, 3).T -> (20, 3)
            aligned_frames[t] = centered_frame @ R.T

        return aligned_frames

    def compute_kinematics(self, positions):
        """
        Computes Velocity and Acceleration.
        Input: (T, 20, 3)
        Output: (T, 20, 9) -> [Pos, Vel, Acc]
        """
        # Velocity: P(t+1) - P(t-1) / 2 (Central difference)
        # For simplicity and edge cases, we use gradient
        vel = np.gradient(positions, axis=0)
        acc = np.gradient(vel, axis=0)

        return np.concatenate([positions, vel, acc], axis=-1)

    def augment_skeleton(self, positions):
        """
        Applies random scaling and rotation for data augmentation.
        Input: (T, 20, 3)
        """
        # Random Scaling (+/- 10%)
        scale = np.random.uniform(0.9, 1.1)

        # Random Rotation around Y-axis (+/- 10 degrees)
        angle_deg = np.random.uniform(-10, 10)
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

        augmented = positions @ R.T
        augmented = augmented * scale

        return augmented

    def process_audio(self, audio_path, target_frames):
        """
        Loads audio, computes MFCC, and interpolates to match video frame count.
        Returns: (T, n_mfcc)
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            # Using standard params
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

            # Interpolate to match target_frames
            # Input to interpolate needs to be (Batch, Channels, Time)
            # mfcc is (1, 13, Time)
            mfcc_interpolated = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )

            # Transpose to (T, n_mfcc)
            return mfcc_interpolated.squeeze(0).transpose(0, 1).numpy()

        except Exception as e:
            # Return zeros if audio fails
            return np.zeros((target_frames, 13), dtype=np.float32)

    def generate_labels(self, num_frames, label_list):
        """Generates frame-wise dense labels (T,)."""
        labels = np.zeros((num_frames,), dtype=np.int64)  # 0 is background

        for l in label_list:
            gid = l["id"]
            start = max(0, l["begin"] - 1)  # 1-based to 0-based
            end = min(num_frames, l["end"])
            labels[start:end] = gid

        return labels

    def process_dataset(
        self, csv_path, cache_path, is_train=False, debug_size=None, load_cached=True
    ):
        """
        Main driver to process a dataset split.
        Returns: features (N_total_frames, D), labels (N_total_frames,), boundaries (N_seq+1,)
        """
        # 1. Try Load Cache
        if load_cached and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            try:
                data = np.load(cache_path)
                return data["features"], data["labels"], data["boundaries"]
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing.")

        # 2. Process from Scratch
        df = pd.read_csv(csv_path)
        if debug_size:
            df = df.head(debug_size)

        all_features = []
        all_labels = []
        boundaries = [0]

        logger.info(f"Processing {len(df)} sequences from {csv_path}...")

        for idx, row in df.iterrows():
            # Paths
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Load Skeleton
            skel = self.load_mat_skeleton(mat_path)
            if skel is None:
                continue

            num_frames = skel.shape[0]
            if num_frames < 10:
                continue  # Skip tiny sequences

            # Align
            aligned_skel = self.align_skeleton(skel)

            # Augment (only if training)
            if is_train:
                # Apply augmentation before kinematics
                aligned_skel = self.augment_skeleton(aligned_skel)

            # Kinematics
            kinematics = self.compute_kinematics(aligned_skel)  # (T, 20, 9)
            flat_kinematics = kinematics.reshape(num_frames, -1)  # (T, 180)

            # Audio
            audio_feat = self.process_audio(audio_path, num_frames)  # (T, 13)

            # Early Fusion
            combined_feat = np.concatenate(
                [flat_kinematics, audio_feat], axis=1
            )  # (T, 193)

            # Labels
            if "labels" in row and isinstance(row["labels"], str):
                import json

                l_list = json.loads(row["labels"])
                seq_labels = self.generate_labels(num_frames, l_list)
            else:
                seq_labels = np.zeros((num_frames,), dtype=np.int64)

            all_features.append(combined_feat)
            all_labels.append(seq_labels)
            boundaries.append(boundaries[-1] + num_frames)

        # Concatenate
        if not all_features:
            logger.error("No valid data processed!")
            return np.array([]), np.array([]), np.array([])

        features_concat = np.concatenate(all_features, axis=0).astype(np.float32)
        labels_concat = np.concatenate(all_labels, axis=0).astype(np.int64)
        boundaries_arr = np.array(boundaries, dtype=np.int32)

        # Cache
        logger.info(f"Saving cache to {cache_path}")
        np.savez_compressed(
            cache_path,
            features=features_concat,
            labels=labels_concat,
            boundaries=boundaries_arr,
        )

        return features_concat, labels_concat, boundaries_arr


class GestureDataset(Dataset):
    """
    PyTorch Dataset that generates sliding windows from the processed continuous data.
    """

    def __init__(
        self, features, labels, boundaries, window_size=64, stride=32, augment=False
    ):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.augment = augment  # Note: geometric augmentation handled in preprocessing, this flag reserved for feature noise if needed

        self.windows = []

        # Pre-calculate window indices
        num_sequences = len(boundaries) - 1
        for i in range(num_sequences):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            seq_len = end_idx - start_idx

            if seq_len < window_size:
                # Pad short sequences? Or skip.
                # For now, if shorter than window, take one window with padding
                self.windows.append(
                    (start_idx, end_idx, True)
                )  # True indicates padding needed
            else:
                # Sliding window
                curr = 0
                while curr + window_size <= seq_len:
                    self.windows.append(
                        (start_idx + curr, start_idx + curr + window_size, False)
                    )
                    curr += stride

                # Handle remainder if significant?
                # Usually strict sliding window is fine.

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        abs_start, abs_end, need_pad = self.windows[idx]

        if need_pad:
            # Extract what we have
            data = self.features[abs_start:abs_end]
            lbl = self.labels[abs_start:abs_end]

            # Pad with zeros or edge
            pad_len = self.window_size - len(data)
            data = np.pad(data, ((0, pad_len), (0, 0)), mode="edge")
            lbl = np.pad(lbl, (0, pad_len), mode="constant", constant_values=0)
        else:
            data = self.features[abs_start:abs_end]
            lbl = self.labels[abs_start:abs_end]

        # Convert to tensor
        data_t = torch.from_numpy(data).float()
        lbl_t = torch.from_numpy(lbl).long()

        return data_t, lbl_t


def get_dataloaders(debug_subset=None):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    processor = DataProcessor()

    # Train
    train_feats, train_lbls, train_bounds = processor.process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_PATH,
        is_train=True,
        debug_size=debug_subset,
    )

    # Val
    val_feats, val_lbls, val_bounds = processor.process_dataset(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_PATH,
        is_train=False,
        debug_size=debug_subset,
    )

    # Test
    test_feats, test_lbls, test_bounds = processor.process_dataset(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_PATH,
        is_train=False,
        debug_size=debug_subset,
    )

    # Datasets
    train_ds = GestureDataset(
        train_feats,
        train_lbls,
        train_bounds,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        augment=True,
    )
    val_ds = GestureDataset(
        val_feats,
        val_lbls,
        val_bounds,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        augment=False,
    )

    # For test, we might want different stride or just standard
    test_ds = GestureDataset(
        test_feats,
        test_lbls,
        test_bounds,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        augment=False,
    )

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_bounds
