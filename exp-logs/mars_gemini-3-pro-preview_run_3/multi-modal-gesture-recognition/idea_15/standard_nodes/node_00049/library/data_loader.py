import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union

from library.config import Config


class PolymorphicMatParser:
    """
    Robust parser for MATLAB files to handle inconsistent Skeleton data structures.
    Designed to prevent failures when 'Skeleton' or 'WorldPosition' fields vary in format.
    """

    @staticmethod
    def load_skeleton(mat_path: str) -> Optional[np.ndarray]:
        """
        Parses the .mat file and extracts skeleton joint positions.
        Returns:
            np.ndarray: Shape (NumFrames, NumJoints, 3) or None if parsing fails.
        """
        try:
            # Load with squeeze_me=True and struct_as_record=False to get objects
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            if "Video" not in mat:
                return None

            video = mat["Video"]

            # Check for Frames
            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames

            # Handle case where frames is a single object, scalar, or None
            if frames is None:
                return None

            # Normalize frames to a list/array structure
            if isinstance(frames, np.ndarray):
                if frames.size == 0:
                    return None
                frame_list = frames
                num_frames = frames.size
            else:
                # Single frame object
                frame_list = np.array([frames])
                num_frames = 1

            # Pre-allocate skeleton array: (T, J, 3)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            # Iterate and extract
            # If frame_list is 0-d array (scalar object wrapped in array), flatten behavior is consistent
            if np.ndim(frame_list) == 0:
                frame_list = np.array([frame_list])

            for i, frame in enumerate(frame_list):
                if not hasattr(frame, "Skeleton"):
                    continue

                skel = frame.Skeleton

                # Polymorphic check: Skeleton might be None, struct, or array
                if skel is None:
                    continue

                target_skel = None

                # If Skeleton is an array (multiple users), take the first one
                if isinstance(skel, np.ndarray):
                    if skel.size > 0:
                        target_skel = skel[0]
                elif hasattr(skel, "WorldPosition"):
                    target_skel = skel

                if target_skel is None or not hasattr(target_skel, "WorldPosition"):
                    continue

                wp = target_skel.WorldPosition

                # Parse WorldPosition
                # Case A: Numpy array (20x3 or 3x20)
                if isinstance(wp, np.ndarray):
                    if wp.shape == (Config.NUM_JOINTS, 3):
                        skeleton_data[i] = wp
                    elif wp.shape == (3, Config.NUM_JOINTS):
                        skeleton_data[i] = wp.T
                    continue

                # Case B: Struct with X, Y, Z fields (common in Kinect data)
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    try:
                        # Extract and flatten to ensure they are 1D arrays
                        x = np.array(wp.X, dtype=np.float32).flatten()
                        y = np.array(wp.Y, dtype=np.float32).flatten()
                        z = np.array(wp.Z, dtype=np.float32).flatten()

                        if (
                            len(x) == Config.NUM_JOINTS
                            and len(y) == Config.NUM_JOINTS
                            and len(z) == Config.NUM_JOINTS
                        ):
                            skeleton_data[i, :, 0] = x
                            skeleton_data[i, :, 1] = y
                            skeleton_data[i, :, 2] = z
                    except Exception:
                        pass

            return skeleton_data

        except Exception as e:
            # In production, we might log this. For now, return None to indicate failure.
            return None


class FeatureEngineer:
    """
    Handles augmentation and feature extraction (Kinematics + Audio).
    """

    def __init__(self):
        # Audio transforms
        self.mfcc_transform = T.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 40, "center": False},
        )

    def augment_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Applies random 3D rotation (Y-axis) and scaling.
        skeleton: (T, J, 3)
        """
        # Random rotation around Y-axis
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)

        # Apply rotation and scale
        T_frames, J, C = skeleton.shape
        flat = skeleton.reshape(-1, 3)
        augmented = np.dot(flat, R.T) * scale
        return augmented.reshape(T_frames, J, C)

    def compute_kinematics(
        self, skeleton: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Relative Pos, Velocity, Acceleration.
        skeleton: (T, J, 3)
        """
        # 1. Root Relative (HipCenter at index 0 is root)
        root = skeleton[:, 0:1, :]  # (T, 1, 3)
        rel_pos = skeleton - root

        # 2. Velocity (First difference)
        vel = np.zeros_like(rel_pos)
        vel[1:] = rel_pos[1:] - rel_pos[:-1]

        # 3. Acceleration (Second difference)
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        return rel_pos, vel, acc

    def process_audio(self, audio_path: str, target_len_frames: int) -> np.ndarray:
        """
        Loads audio, computes MFCC, and aligns to video frames via interpolation.
        """
        try:
            if not os.path.exists(audio_path):
                return np.zeros((target_len_frames, Config.N_MFCC), dtype=np.float32)

            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = T.Resample(sample_rate, Config.AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC -> (1, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (Time_audio, n_mfcc)

            if mfcc.shape[0] == 0:
                return np.zeros((target_len_frames, Config.N_MFCC), dtype=np.float32)

            # Align to video frames using linear interpolation
            x_old = np.linspace(0, 1, mfcc.shape[0])
            x_new = np.linspace(0, 1, target_len_frames)

            aligned_mfcc = np.zeros(
                (target_len_frames, Config.N_MFCC), dtype=np.float32
            )
            for i in range(Config.N_MFCC):
                aligned_mfcc[:, i] = np.interp(x_new, x_old, mfcc[:, i])

            return aligned_mfcc

        except Exception:
            return np.zeros((target_len_frames, Config.N_MFCC), dtype=np.float32)


def get_labels_array(num_frames: int, labels_meta: List[Dict]) -> np.ndarray:
    """
    Constructs frame-wise label array (0=Background, 1-20=Gestures).
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    for l in labels_meta:
        # Convert 1-based indexing to 0-based
        start = max(0, l["begin"] - 1)
        end = min(num_frames, l["end"])
        gid = l["id"]
        if start < end:
            labels[start:end] = gid
    return labels


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        mode: str = "train",
        load_cached_data: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            metadata_path: Path to csv.
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to try loading from cache.
            debug: If True, use subset.
        """
        self.mode = mode
        self.debug = debug
        self.df = pd.read_csv(metadata_path)

        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SIZE]

        self.parser = PolymorphicMatParser()
        self.engineer = FeatureEngineer()

        # Cache setup
        cache_name = f"dataset_{mode}{'_debug' if debug else ''}.npz"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_name)
        Config.setup_directories()

        # Data containers
        self.X = None
        self.Y = None
        self.window_metadata = []  # List of {'sample_id': str, 'start_frame': int}

        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached data from {self.cache_path}...")
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self.X = data["X"]
                self.Y = data["Y"]
                # Load metadata if available
                if "metadata" in data:
                    self.window_metadata = data["metadata"].tolist()
                print(f"Loaded {len(self.X)} windows.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
                self._process_and_cache()
        else:
            self._process_and_cache()

    def _process_and_cache(self):
        print(f"Processing {len(self.df)} samples for {self.mode}...")

        X_list = []
        Y_list = []
        meta_list = []

        for idx, row in self.df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # 1. Load Skeleton
            skeleton = self.parser.load_skeleton(data_path)

            # Handle missing skeleton
            if skeleton is None:
                if self.mode == "test":
                    # Fallback for test: create dummy skeleton of length 100 (approx avg duration)
                    # or try to infer length from audio?
                    # Let's use a safe default to ensure pipeline continuity.
                    skeleton = np.zeros((100, Config.NUM_JOINTS, 3), dtype=np.float32)
                else:
                    # Skip bad training samples
                    continue

            num_frames = skeleton.shape[0]

            # 2. Augment (only in train)
            if self.mode == "train":
                skeleton = self.engineer.augment_skeleton(skeleton)

            # 3. Compute Kinematics
            rel_pos, vel, acc = self.engineer.compute_kinematics(skeleton)

            # Flatten spatial dims: (T, Features)
            f_rel = rel_pos.reshape(num_frames, -1)
            f_vel = vel.reshape(num_frames, -1)
            f_acc = acc.reshape(num_frames, -1)

            # 4. Audio
            f_audio = self.engineer.process_audio(audio_path, num_frames)

            # 5. Early Fusion
            # (T, 193)
            features = np.concatenate([f_rel, f_vel, f_acc, f_audio], axis=1)

            # 6. Labels
            if self.mode != "test":
                labels_meta = json.loads(row["labels"])
                labels = get_labels_array(num_frames, labels_meta)
            else:
                labels = np.zeros(num_frames, dtype=np.int64)

            # 7. Sliding Window Generation
            # Pad if shorter than window
            if num_frames < Config.WINDOW_SIZE:
                pad_len = Config.WINDOW_SIZE - num_frames
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                labels = np.pad(labels, (0, pad_len), mode="constant")
                num_frames = Config.WINDOW_SIZE

            # Determine stride
            # Train: Overlap to augment data. Test: Dense overlap for ensemble or standard stride.
            # Using standard stride for test to cover sequence.
            stride = (
                Config.WINDOW_STRIDE
                if self.mode == "train"
                else int(Config.WINDOW_SIZE * 0.5)
            )

            for start in range(0, num_frames - Config.WINDOW_SIZE + 1, stride):
                end = start + Config.WINDOW_SIZE
                win_x = features[start:end]
                win_y = labels[start:end]

                X_list.append(win_x.astype(np.float32))
                Y_list.append(win_y.astype(np.int64))
                meta_list.append({"sample_id": sample_id, "start_frame": start})

            # Ensure the last frame is covered in Test mode if not perfectly divisible
            if self.mode == "test" and (num_frames - Config.WINDOW_SIZE) % stride != 0:
                start = num_frames - Config.WINDOW_SIZE
                win_x = features[start:]
                win_y = labels[start:]
                X_list.append(win_x.astype(np.float32))
                Y_list.append(win_y.astype(np.int64))
                meta_list.append({"sample_id": sample_id, "start_frame": start})

        if len(X_list) == 0:
            print("Warning: No valid windows generated.")
            self.X = np.zeros(
                (0, Config.WINDOW_SIZE, Config.INPUT_DIM), dtype=np.float32
            )
            self.Y = np.zeros((0, Config.WINDOW_SIZE), dtype=np.int64)
        else:
            self.X = np.stack(X_list)
            self.Y = np.stack(Y_list)
            self.window_metadata = meta_list

        print(f"Generated {len(self.X)} windows. Saving to cache...")
        np.savez(self.cache_path, X=self.X, Y=self.Y, metadata=self.window_metadata)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return tensors and index (for metadata lookup)
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        return x, y, idx


def get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Returns train and validation DataLoaders.
    """
    train_ds = GestureDataset(Config.TRAIN_CSV, mode="train", debug=debug)
    val_ds = GestureDataset(Config.VAL_CSV, mode="val", debug=debug)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Returns test DataLoader.
    """
    test_ds = GestureDataset(Config.TEST_CSV, mode="test", debug=debug)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return test_loader
