import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_mat_file_polymorphic


class GestureDataset(Dataset):
    """
    Dataset class for the Root-Centric Moderate-Capacity Network.
    Handles multimodal data ingestion, caching, windowing, and kinematic augmentation.
    """

    def __init__(self, split="train", load_cached_data=True, transform=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.
            transform (bool): Whether to apply augmentation (only for 'train').
        """
        self.split = split
        self.transform = transform and (split == "train")
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

        # Determine metadata file
        if split == "train":
            self.metadata_file = Config.TRAIN_CSV
        elif split == "val":
            self.metadata_file = Config.VAL_CSV
        else:
            self.metadata_file = Config.TEST_CSV

        # Cache path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split}.npz")

        # Load data (cached or fresh)
        self.data = self._load_and_cache_data(load_cached_data)

        # Build indices
        self.indices = []
        if self.split == "train":
            self._build_window_indices()
        else:
            # For val/test, we index by sequence
            self.indices = list(range(len(self.data)))

    def _load_and_cache_data(self, load_cached_data):
        """
        Loads data from cache or processes raw files and saves to cache.
        Returns a list of dictionaries: [{'sample_id', 'skeleton', 'audio', 'label'}, ...]
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading {self.split} data from cache: {self.cache_path}")
                loaded = np.load(self.cache_path, allow_pickle=True)
                data_list = []
                # Reconstruct list of dicts from npz arrays
                # Keys are stored as 'sample_id_0', 'skeleton_0', etc.
                # To be robust, we store a metadata array with keys
                keys = loaded["keys"]
                for i, k in enumerate(keys):
                    item = {
                        "sample_id": str(k),
                        "skeleton": loaded[f"skeleton_{i}"],
                        "audio": loaded[f"audio_{i}"],
                        "label": loaded[f"label_{i}"],
                    }
                    data_list.append(item)
                return data_list
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from Scratch
        print(f"Processing {self.split} data from raw files...")
        df = pd.read_csv(self.metadata_file)

        # Parse labels column if it exists
        if "labels" in df.columns:
            df["parsed_labels"] = df["labels"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else []
            )

        data_list = []

        # Audio transform initialization
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,  # Will resample if needed
            n_mfcc=Config.AUDIO_MFCC_DIM,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        for idx, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # --- Skeleton Processing ---
            # Load raw skeleton: (T, 20, 3)
            skeleton = load_mat_file_polymorphic(data_path)

            if skeleton is None:
                # Fallback for missing/corrupt data: create dummy zero data based on audio or default
                # Try to get length from audio or default to small number
                T = 100
                skeleton = np.zeros((T, Config.NUM_JOINTS, 3), dtype=np.float32)

            T = skeleton.shape[0]

            # Root-Relative Centering
            # HipCenter is assumed to be index 0.
            # Subtract HipCenter position from all joints for each frame.
            hip_center = skeleton[:, 0:1, :]  # (T, 1, 3)
            skeleton_centered = skeleton - hip_center

            # --- Audio Processing ---
            audio_features = np.zeros((T, Config.AUDIO_MFCC_DIM), dtype=np.float32)
            if os.path.exists(audio_path):
                try:
                    # Load audio
                    waveform, sample_rate = sf.read(audio_path)
                    # Convert to torch
                    if len(waveform.shape) > 1:
                        waveform = waveform[:, 0]  # Take first channel
                    waveform = torch.from_numpy(waveform).float()

                    # Resample if necessary (assuming 16k target for MFCC config)
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        )
                        waveform = resampler(waveform)

                    # Compute MFCC
                    mfcc = mfcc_transform(waveform)  # (n_mfcc, time)
                    mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

                    # Align with Video Frames
                    # Interpolate MFCCs to match video frame count T
                    if mfcc.shape[0] > 0:
                        mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
                        mfcc_aligned = torch.nn.functional.interpolate(
                            mfcc, size=T, mode="linear", align_corners=False
                        )
                        audio_features = mfcc_aligned.squeeze(0).transpose(0, 1).numpy()

                except Exception as e:
                    # Keep zeros if audio fails
                    pass

            # --- Label Generation ---
            labels = np.zeros(T, dtype=np.int64)  # Default 0 (background)
            if "parsed_labels" in row:
                for gesture in row["parsed_labels"]:
                    gid = gesture["id"]
                    start = max(0, gesture["begin"] - 1)  # 1-based to 0-based
                    end = min(T, gesture["end"])
                    if start < end:
                        labels[start:end] = gid

            data_list.append(
                {
                    "sample_id": sample_id,
                    "skeleton": skeleton_centered.astype(np.float32),
                    "audio": audio_features.astype(np.float32),
                    "label": labels.astype(np.int64),
                }
            )

        # 3. Save to Cache
        save_dict = {"keys": [d["sample_id"] for d in data_list]}
        for i, item in enumerate(data_list):
            save_dict[f"skeleton_{i}"] = item["skeleton"]
            save_dict[f"audio_{i}"] = item["audio"]
            save_dict[f"label_{i}"] = item["label"]

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Saved processed data to {self.cache_path}")

        return data_list

    def _build_window_indices(self):
        """
        Creates a list of (sample_index, start_frame) tuples for sliding windows.
        Used only for training.
        """
        self.indices = []
        for i, item in enumerate(self.data):
            num_frames = item["skeleton"].shape[0]
            # If sequence is shorter than window, pad it later or skip?
            # We will pad in __getitem__ if needed, but here we just add index 0
            if num_frames <= self.window_size:
                self.indices.append((i, 0))
            else:
                # Sliding window
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.indices.append((i, start))

                # Ensure the last frame is covered
                if (num_frames - self.window_size) % self.stride != 0:
                    self.indices.append((i, num_frames - self.window_size))

    def _augment_skeleton(self, skeleton):
        """
        Applies random rotation (Y-axis) and scaling to the skeleton window.
        Args:
            skeleton: (T, Joints, 3)
        Returns:
            augmented_skeleton: (T, Joints, 3)
        """
        T, J, C = skeleton.shape

        # 1. Random Rotation around Y-axis
        # Angle in radians, e.g., +/- 20 degrees (~0.35 rad)
        theta = np.random.uniform(-0.35, 0.35)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Reshape for matmul: (T*J, 3)
        flat_skel = skeleton.reshape(-1, 3)
        rotated_skel = np.dot(flat_skel, rotation_matrix.T)
        skeleton = rotated_skel.reshape(T, J, 3)

        # 2. Random Scaling
        # Scale factor between 0.9 and 1.1
        scale = np.random.uniform(0.9, 1.1)
        skeleton = skeleton * scale

        return skeleton

    def _compute_kinematics(self, skeleton):
        """
        Computes Velocity and Acceleration.
        Args:
            skeleton: (T, Joints, 3) - Position
        Returns:
            features: (T, Joints * 9) - Flattened [Pos, Vel, Acc]
        """
        # Velocity: P[t] - P[t-1]
        # Pad first frame with 0
        vel = np.zeros_like(skeleton)
        vel[1:] = skeleton[1:] - skeleton[:-1]

        # Acceleration: V[t] - V[t-1]
        acc = np.zeros_like(skeleton)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate: (T, Joints, 9)
        # Structure: Joint1_Pos(xyz), Joint1_Vel(xyz), Joint1_Acc(xyz), Joint2...
        # Or: All_Pos, All_Vel, All_Acc.
        # Config says SKELETON_INPUT_DIM = 20 * 3 * 3 = 180.
        # We'll flatten joints last to keep spatial structure if needed,
        # but for MLP/RNN input usually we flatten everything per frame.

        combined = np.concatenate([skeleton, vel, acc], axis=2)  # (T, J, 9)
        flattened = combined.reshape(combined.shape[0], -1)  # (T, J*9)

        return flattened

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            features: (T, InputDim) Tensor
            labels: (T,) LongTensor
        """
        if self.split == "train":
            # Window-based retrieval
            sample_idx, start_frame = self.indices[idx]
            item = self.data[sample_idx]

            end_frame = start_frame + self.window_size

            # Extract raw data
            skel_window = item["skeleton"][
                start_frame:end_frame
            ].copy()  # (T_win, 20, 3)
            audio_window = item["audio"][start_frame:end_frame].copy()  # (T_win, 13)
            label_window = item["label"][start_frame:end_frame].copy()  # (T_win,)

            # Pad if shorter than window_size (only happens if sequence < window_size)
            curr_len = skel_window.shape[0]
            if curr_len < self.window_size:
                pad_len = self.window_size - curr_len
                # Pad with zeros (or replicate last frame)
                skel_window = np.pad(
                    skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="edge"
                )
                audio_window = np.pad(
                    audio_window, ((0, pad_len), (0, 0)), mode="constant"
                )
                label_window = np.pad(
                    label_window, (0, pad_len), mode="constant", constant_values=0
                )

            # Augmentation (Position only)
            if self.transform:
                skel_window = self._augment_skeleton(skel_window)

            # Compute Kinematics (on potentially augmented data)
            skel_features = self._compute_kinematics(skel_window)  # (T, 180)

            # Fusion
            features = np.concatenate([skel_features, audio_window], axis=1)  # (T, 193)

            return (
                torch.from_numpy(features).float(),
                torch.from_numpy(label_window).long(),
            )

        else:
            # Full sequence retrieval (Val/Test)
            # idx is directly the sample index
            item = self.data[idx]

            skel_seq = item["skeleton"].copy()
            audio_seq = item["audio"].copy()
            label_seq = item["label"].copy()

            # Compute Kinematics (No augmentation)
            skel_features = self._compute_kinematics(skel_seq)

            # Fusion
            features = np.concatenate([skel_features, audio_seq], axis=1)

            return (
                torch.from_numpy(features).float(),
                torch.from_numpy(label_seq).long(),
            )
