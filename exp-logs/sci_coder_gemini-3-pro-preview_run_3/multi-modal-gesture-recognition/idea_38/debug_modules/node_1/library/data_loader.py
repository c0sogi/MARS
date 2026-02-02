import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import scipy.io
import json
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import safe_load_mat, parse_skeleton_structure

# Set random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class GestureDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and behavior.
            load_cached_data (bool): Whether to load pre-processed data from cache.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.mode = mode
        self.debug = debug
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE_TRAIN if mode == "train" else Config.STRIDE_TEST

        # Cache file path
        cache_name = f"dataset_{mode}{'_debug' if debug else ''}.npz"
        self.cache_path = os.path.join(Config.WORKING_DIR, cache_name)

        # Data containers
        self.all_skeleton = None  # (TotalFrames, NumJoints, 3)
        self.all_audio = None  # (TotalFrames, AudioDim)
        self.all_labels = None  # (TotalFrames,)
        self.sample_indices = None  # (NumSamples, 2) -> [start, end)

        # Load data
        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache(metadata_path)

        # Pre-calculate sliding windows
        self.windows = self._make_windows()

    def _load_cache(self):
        print(f"Loading cached data from {self.cache_path}...")
        data = np.load(self.cache_path)
        self.all_skeleton = data["skeleton"]
        self.all_audio = data["audio"]
        self.all_labels = data["labels"]
        self.sample_indices = data["indices"]

    def _process_and_cache(self, metadata_path):
        print(f"Processing data from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        if self.debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        skeleton_list = []
        audio_list = []
        labels_list = []
        indices_list = []

        current_idx = 0

        # MFCC Transform setup
        # Assuming 16kHz audio based on analysis, but we read SR from file
        # We'll instantiate transform per file if SR varies, or assume fixed.
        # Analysis showed 16000Hz.

        for _, row in df.iterrows():
            # 1. Load Skeleton Data
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            mat = safe_load_mat(mat_path)

            if mat is None:
                continue

            # Cite debug_lesson_2: Access top-level variables from loadmat as dictionary keys
            if "Video" not in mat:
                continue
            video_struct = mat["Video"]

            # Cite debug_lesson_16: Unwrap 0-d NumPy arrays from squeeze_me=True
            if isinstance(video_struct, np.ndarray) and video_struct.ndim == 0:
                video_struct = video_struct.item()

            frames_data = parse_skeleton_structure(video_struct)

            if frames_data is None:
                # Fallback or skip
                continue

            num_frames = len(frames_data)
            if num_frames == 0:
                continue

            # Extract WorldPosition (T, 20, 3)
            # Initialize with zeros
            skel_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

            for t, frame in enumerate(frames_data):
                if hasattr(frame, "Skeleton") and hasattr(
                    frame.Skeleton, "WorldPosition"
                ):
                    wp = frame.Skeleton.WorldPosition
                    # wp might be a single object or array, handle robustly
                    # Assuming standard structure based on prompt description
                    # WorldPosition X, Y, Z in mm
                    try:
                        # Iterate over joints
                        # The structure is often an array of joints
                        # Check if WorldPosition is an array of structs or a single struct with arrays
                        # Based on description: "Skeleton ... JointsType ... WorldPosition"
                        # Usually Skeleton is an array of 20 joints
                        joints = frame.Skeleton
                        if (
                            isinstance(joints, (list, np.ndarray))
                            and len(joints) == Config.NUM_JOINTS
                        ):
                            for j_idx, joint in enumerate(joints):
                                if hasattr(joint, "WorldPosition"):
                                    pos = joint.WorldPosition
                                    skel_data[t, j_idx, 0] = pos.X
                                    skel_data[t, j_idx, 1] = pos.Y
                                    skel_data[t, j_idx, 2] = pos.Z
                    except Exception:
                        pass  # Keep zeros

            # 2. Load Audio Data and Align
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)

                # Compute MFCC
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=Config.AUDIO_MFCC_DIM,
                    melkwargs={
                        "n_fft": 400,
                        "hop_length": 160,
                        "n_mels": 23,
                        "center": False,
                    },
                )
                mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

                # Interpolate to match number of video frames
                # Input to interpolate: (Batch, Channels, Time)
                mfcc = mfcc.unsqueeze(0)
                mfcc = torch.nn.functional.interpolate(
                    mfcc, size=num_frames, mode="linear", align_corners=False
                )
                mfcc = mfcc.squeeze(0).permute(2, 1, 0).squeeze(-1)  # (Time, n_mfcc)
                audio_feat = mfcc.numpy()
            else:
                audio_feat = np.zeros(
                    (num_frames, Config.AUDIO_MFCC_DIM), dtype=np.float32
                )

            # 3. Process Labels
            # Initialize background (0)
            label_seq = (
                np.zeros(num_frames, dtype=np.int64) + Config.BACKGROUND_CLASS_ID
            )

            if self.mode != "test":
                try:
                    labels_meta = json.loads(row["labels"])
                    for l in labels_meta:
                        start = max(0, int(l["begin"]) - 1)  # 1-based to 0-based
                        end = min(num_frames, int(l["end"]))
                        gid = int(l["id"])
                        if start < end:
                            label_seq[start:end] = gid
                except:
                    pass

            # 4. Append
            skeleton_list.append(skel_data)
            audio_list.append(audio_feat)
            labels_list.append(label_seq)

            indices_list.append([current_idx, current_idx + num_frames])
            current_idx += num_frames

        # Concatenate
        if len(skeleton_list) > 0:
            self.all_skeleton = np.concatenate(skeleton_list, axis=0)
            self.all_audio = np.concatenate(audio_list, axis=0)
            self.all_labels = np.concatenate(labels_list, axis=0)
            self.sample_indices = np.array(indices_list, dtype=np.int64)
        else:
            # Empty dataset handling
            self.all_skeleton = np.zeros((0, Config.NUM_JOINTS, 3), dtype=np.float32)
            self.all_audio = np.zeros((0, Config.AUDIO_MFCC_DIM), dtype=np.float32)
            self.all_labels = np.zeros((0,), dtype=np.int64)
            self.sample_indices = np.zeros((0, 2), dtype=np.int64)

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(
            self.cache_path,
            skeleton=self.all_skeleton,
            audio=self.all_audio,
            labels=self.all_labels,
            indices=self.sample_indices,
        )
        print(f"Cached data saved to {self.cache_path}")

    def _make_windows(self):
        """
        Create a list of (start_index, end_index) tuples for sliding windows.
        Indices refer to the global concatenated arrays.
        """
        windows = []
        for start_sample, end_sample in self.sample_indices:
            length = end_sample - start_sample

            # If sequence is shorter than window, pad it (handled in getitem via slicing logic)
            # or just take one window.
            if length <= self.window_size:
                windows.append((start_sample, end_sample))
                continue

            # Sliding window
            curr = start_sample
            while curr < end_sample:
                # Define window end
                w_end = curr + self.window_size

                # If we go past the end of the sample
                if w_end > end_sample:
                    # If strictly training, we might skip the partial last window or pad.
                    # For testing, we usually want to cover everything.
                    # Strategy: Shift back to fit the last window exactly at the end
                    curr = max(start_sample, end_sample - self.window_size)
                    w_end = end_sample
                    windows.append((curr, w_end))
                    break

                windows.append((curr, w_end))
                curr += self.stride

        return windows

    def _augment_skeleton(self, positions):
        """
        Apply random rotation (Y-axis) and scaling.
        positions: (T, 20, 3)
        """
        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        positions = positions * scale

        # Random rotation around Y-axis
        theta = np.random.uniform(-15, 15) * np.pi / 180.0
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: dot product over last dimension
        # positions shape: (T, J, 3)
        # Reshape to (T*J, 3) for matmul
        T, J, _ = positions.shape
        flat_pos = positions.reshape(-1, 3)
        rotated = np.dot(flat_pos, rotation_matrix.T)
        return rotated.reshape(T, J, 3)

    def _compute_kinematics(self, positions):
        """
        Compute Velocity and Acceleration.
        positions: (T, 20, 3)
        Returns: (T, 20*3*3) -> [Pos, Vel, Acc] flattened
        """
        T, J, C = positions.shape

        # Velocity
        vel = np.zeros_like(positions)
        vel[1:] = positions[1:] - positions[:-1]

        # Acceleration
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate and Flatten
        # (T, J, 3) -> (T, J*3)
        pos_flat = positions.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)
        acc_flat = acc.reshape(T, -1)

        # (T, J*9)
        features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)
        return features

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx, end_idx = self.windows[idx]

        # Extract raw data
        # Copy to avoid modifying cached array during augmentation
        skel_window = self.all_skeleton[start_idx:end_idx].copy()
        audio_window = self.all_audio[start_idx:end_idx].copy()
        labels_window = self.all_labels[start_idx:end_idx].copy()

        current_len = skel_window.shape[0]

        # Padding if window is smaller than target size (rare, but possible for short clips)
        if current_len < self.window_size:
            pad_len = self.window_size - current_len
            # Pad skeleton with zeros
            skel_pad = np.zeros((pad_len, Config.NUM_JOINTS, 3), dtype=np.float32)
            skel_window = np.concatenate([skel_window, skel_pad], axis=0)

            # Pad audio with zeros
            audio_pad = np.zeros((pad_len, Config.AUDIO_MFCC_DIM), dtype=np.float32)
            audio_window = np.concatenate([audio_window, audio_pad], axis=0)

            # Pad labels with background (0)
            label_pad = np.zeros(pad_len, dtype=np.int64) + Config.BACKGROUND_CLASS_ID
            labels_window = np.concatenate([labels_window, label_pad], axis=0)

        # Augmentation (Training only)
        if self.mode == "train":
            skel_window = self._augment_skeleton(skel_window)

        # Compute Kinematics (Pos, Vel, Acc)
        # Shape: (WindowSize, 180)
        skel_features = self._compute_kinematics(skel_window)

        # Concatenate Audio
        # Shape: (WindowSize, 193)
        features = np.concatenate([skel_features, audio_window], axis=1)

        # Convert to tensors
        features_t = torch.tensor(features, dtype=torch.float32)
        labels_t = torch.tensor(labels_window, dtype=torch.long)

        return features_t, labels_t


def get_data_loaders(config=Config, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Train
    train_dataset = GestureDataset(
        config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached_data=load_cached_data,
        debug=config.DEBUG,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_dataset = GestureDataset(
        config.VAL_METADATA_PATH,
        mode="val",
        load_cached_data=load_cached_data,
        debug=config.DEBUG,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Test
    test_dataset = GestureDataset(
        config.TEST_METADATA_PATH,
        mode="test",
        load_cached_data=load_cached_data,
        debug=config.DEBUG,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
