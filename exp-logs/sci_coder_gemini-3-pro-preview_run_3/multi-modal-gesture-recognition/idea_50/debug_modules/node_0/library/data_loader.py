import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# Ensure reproducible results
utils.set_seed(config.SEED)


class GestureDataset(Dataset):
    def __init__(
        self,
        split,
        mode="train",
        augment=False,
        max_samples=None,
        load_cached_data=True,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            mode (str): 'train' (windowed) or 'inference' (full sequence).
            augment (bool): Whether to apply kinematic augmentation.
            max_samples (int): Limit dataset size for debugging.
            load_cached_data (bool): Use cached .npz files.
        """
        self.split = split
        self.mode = mode
        self.augment = augment
        self.max_samples = max_samples

        # Metadata loading
        metadata_file = os.path.join(config.METADATA_DIR, f"{split}.csv")
        self.metadata = pd.read_csv(metadata_file)

        if self.max_samples is not None:
            self.metadata = self.metadata.iloc[: self.max_samples]

        # Cache paths
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"dataset_{split}.npz")

        # Load or Process Data
        self.data = self._load_data(load_cached_data)

        # Prepare Windows if in training mode
        self.windows = []
        if self.mode == "train":
            self._prepare_windows()

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes from raw files.
        Returns a list of dictionaries containing 'positions', 'audio', 'labels', 'sample_id'.
        """
        if load_cached and os.path.exists(self.cache_file):
            try:
                print(f"Loading cached {self.split} data from {self.cache_file}...")
                loaded = np.load(self.cache_file, allow_pickle=True)
                # Reconstruct list of dicts
                data = []
                for i in range(len(loaded["sample_ids"])):
                    item = {
                        "sample_id": str(loaded["sample_ids"][i]),
                        "positions": loaded["positions"][i],
                        "audio": loaded["audio"][i],
                        "labels": loaded["labels"][i],
                    }
                    data.append(item)
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Processing {self.split} data from scratch...")
        data = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

            # 1. Parse Skeleton
            positions = self._load_mat_robust(mat_path)
            if positions is None:
                continue  # Skip corrupt samples

            num_frames = positions.shape[0]

            # 2. Parse Audio
            audio_features = self._process_audio(audio_path, num_frames)

            # 3. Parse Labels
            labels_json = row["labels"]
            labels_list = (
                json.loads(labels_json) if isinstance(labels_json, str) else []
            )

            # Convert labels to frame-wise annotations for training/eval
            # 0 = Background
            frame_labels = np.zeros(num_frames, dtype=int)
            for l in labels_list:
                start = max(0, l["begin"] - 1)  # 1-based to 0-based
                end = min(num_frames, l["end"])
                gid = l["id"]
                frame_labels[start:end] = gid

            data.append(
                {
                    "sample_id": sample_id,
                    "positions": positions,  # (T, J, 3)
                    "audio": audio_features,  # (T, AudioDim)
                    "labels": frame_labels,  # (T,)
                }
            )

        # Save to cache
        try:
            sample_ids = [d["sample_id"] for d in data]
            positions_arr = np.array([d["positions"] for d in data], dtype=object)
            audio_arr = np.array([d["audio"] for d in data], dtype=object)
            labels_arr = np.array([d["labels"] for d in data], dtype=object)

            np.savez_compressed(
                self.cache_file,
                sample_ids=sample_ids,
                positions=positions_arr,
                audio=audio_arr,
                labels=labels_arr,
            )
            print(f"Saved {self.split} data to cache.")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

        return data

    def _load_mat_robust(self, path):
        """
        Parses .mat file robustly to extract skeleton world positions.
        Returns (T, 20, 3) numpy array or None if failed.
        """
        try:
            mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat.__dict__:
                return None
            video = mat.Video

            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames

            # Handle polymorphic Frames structure
            # It could be a list, an array, or a single object
            if isinstance(frames, (list, np.ndarray)):
                num_frames = len(frames)
                frame_list = frames
            elif hasattr(frames, "Skeleton"):  # Single frame object
                num_frames = 1
                frame_list = [frames]
            else:
                return None

            # Pre-allocate
            # 20 joints, 3 coords
            skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

            for i, f in enumerate(frame_list):
                if hasattr(f, "Skeleton"):
                    skel = f.Skeleton
                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        # wp should be 20 objects or an array of structs
                        # In these datasets, usually wp is an array or list of structs
                        # We need to iterate joints
                        # Check if wp is iterable
                        if isinstance(wp, (list, np.ndarray)) and len(wp) == 20:
                            for j in range(20):
                                joint = wp[j]
                                # Check for X, Y, Z
                                if (
                                    hasattr(joint, "X")
                                    and hasattr(joint, "Y")
                                    and hasattr(joint, "Z")
                                ):
                                    skeleton_data[i, j, 0] = joint.X
                                    skeleton_data[i, j, 1] = joint.Y
                                    skeleton_data[i, j, 2] = joint.Z

            return skeleton_data

        except Exception as e:
            return None

    def _process_audio(self, path, target_frames):
        """
        Loads audio, extracts MFCC, and aligns to video frames.
        Returns (T, MFCC_Dim)
        """
        if not os.path.exists(path):
            return np.zeros((target_frames, config.AUDIO_N_MFCC), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(path)

            # Resample if needed
            if sample_rate != config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Extract MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=config.AUDIO_SAMPLE_RATE,
                n_mfcc=config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

            mfcc_np = mfcc.numpy()

            # Align to video frames using linear interpolation
            curr_len = mfcc_np.shape[0]
            if curr_len != target_frames:
                # Create time indices
                x_old = np.linspace(0, 1, curr_len)
                x_new = np.linspace(0, 1, target_frames)

                aligned_mfcc = np.zeros(
                    (target_frames, config.AUDIO_N_MFCC), dtype=np.float32
                )
                for c in range(config.AUDIO_N_MFCC):
                    aligned_mfcc[:, c] = np.interp(x_new, x_old, mfcc_np[:, c])
                return aligned_mfcc

            return mfcc_np

        except Exception:
            return np.zeros((target_frames, config.AUDIO_N_MFCC), dtype=np.float32)

    def _prepare_windows(self):
        """
        Creates a list of (sample_idx, start_frame) for sliding windows.
        """
        self.windows = []
        for idx, sample in enumerate(self.data):
            num_frames = sample["positions"].shape[0]
            if num_frames < config.WINDOW_SIZE:
                self.windows.append((idx, 0))
            else:
                for start in range(
                    0, num_frames - config.WINDOW_SIZE + 1, config.STRIDE
                ):
                    self.windows.append((idx, start))
                # Ensure we cover the end
                if (num_frames - config.WINDOW_SIZE) % config.STRIDE != 0:
                    self.windows.append((idx, num_frames - config.WINDOW_SIZE))

    def _compute_kinematics(self, positions):
        """
        Derives kinematic features with optional augmentation.
        positions: (T, 20, 3)
        Returns: (T, InputDim)
        """
        T, J, D = positions.shape

        # 1. Root-Relative Centering
        # HipCenter is index 0
        root = positions[:, 0:1, :]  # (T, 1, 3)
        centered = positions - root

        # 2. Augmentation (Injection before derivation)
        if self.augment:
            # Gaussian Noise
            noise = np.random.normal(0, config.NOISE_SIGMA, centered.shape)
            centered = centered + noise

            # Random Rotation (Y-axis)
            if config.RANDOM_ROTATION:
                theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                # Apply rotation: (T*J, 3) @ R.T
                flat = centered.reshape(-1, 3)
                centered = np.dot(flat, R.T).reshape(T, J, 3)

            # Random Scaling
            if config.RANDOM_SCALE:
                scale = np.random.uniform(0.9, 1.1)
                centered = centered * scale

        # 3. Feature Derivation
        features_list = []

        # A. Positions (Raw/Centered)
        features_list.append(centered)  # (T, 20, 3)

        # B. Explicit Bone Vectors
        if config.USE_BONE_VECTORS:
            bones = np.zeros_like(centered)
            # Map pairs
            for child_idx, parent_idx in config.SKELETON_PAIRS:
                bones[:, child_idx, :] = (
                    centered[:, child_idx, :] - centered[:, parent_idx, :]
                )
            features_list.append(bones)

        # C. Velocity
        if config.USE_VELOCITY:
            # Gradient along time axis (axis 0)
            vel = np.gradient(centered, axis=0)
            features_list.append(vel)

        # D. Acceleration
        if config.USE_ACCELERATION:
            if not config.USE_VELOCITY:
                vel = np.gradient(centered, axis=0)
            # Gradient of velocity
            acc = np.gradient(vel, axis=0)
            features_list.append(acc)

        # Concatenate features: (T, 20, 3 * num_types)
        combined = np.concatenate(features_list, axis=2)

        # Flatten joints: (T, 20 * FeaturesPerJoint)
        flat_features = combined.reshape(T, -1)

        return flat_features.astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return len(self.windows)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            sample_idx, start_frame = self.windows[idx]
            sample = self.data[sample_idx]

            # Extract Window
            end_frame = start_frame + config.WINDOW_SIZE
            pos_window = sample["positions"][start_frame:end_frame]
            audio_window = sample["audio"][start_frame:end_frame]
            label_window = sample["labels"][start_frame:end_frame]

            # Padding if shorter (should handle edge cases)
            curr_len = pos_window.shape[0]
            if curr_len < config.WINDOW_SIZE:
                pad_len = config.WINDOW_SIZE - curr_len
                # Pad positions with last frame
                pos_pad = np.tile(pos_window[-1:], (pad_len, 1, 1))
                pos_window = np.concatenate([pos_window, pos_pad], axis=0)

                # Pad audio with zeros
                audio_pad = np.zeros(
                    (pad_len, audio_window.shape[1]), dtype=audio_window.dtype
                )
                audio_window = np.concatenate([audio_window, audio_pad], axis=0)

                # Pad labels with background (0)
                label_pad = np.zeros(pad_len, dtype=label_window.dtype)
                label_window = np.concatenate([label_window, label_pad], axis=0)

            # Compute Kinematics (includes augmentation)
            skel_features = self._compute_kinematics(pos_window)

            return {
                "skeleton": torch.from_numpy(skel_features),
                "audio": torch.from_numpy(audio_window),
                "labels": torch.from_numpy(label_window).long(),
            }

        else:
            # Inference Mode: Return full sequence
            sample = self.data[idx]
            pos_seq = sample["positions"]
            audio_seq = sample["audio"]
            label_seq = sample["labels"]

            skel_features = self._compute_kinematics(pos_seq)

            return {
                "skeleton": torch.from_numpy(skel_features),
                "audio": torch.from_numpy(audio_seq),
                "labels": torch.from_numpy(label_seq).long(),
                "sample_id": sample["sample_id"],
            }


def collate_fn(batch):
    """
    Custom collate function to handle variable lengths in inference mode.
    For training, lengths are fixed by windowing.
    """
    # Check if we are in inference mode (variable lengths)
    if "sample_id" in batch[0]:
        max_len = max([b["skeleton"].shape[0] for b in batch])
        skel_dim = batch[0]["skeleton"].shape[1]
        audio_dim = batch[0]["audio"].shape[1]

        skeletons = torch.zeros(len(batch), max_len, skel_dim)
        audios = torch.zeros(len(batch), max_len, audio_dim)
        labels = torch.zeros(len(batch), max_len, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        sample_ids = []

        for i, b in enumerate(batch):
            l = b["skeleton"].shape[0]
            skeletons[i, :l] = b["skeleton"]
            audios[i, :l] = b["audio"]
            labels[i, :l] = b["labels"]
            lengths[i] = l
            sample_ids.append(b["sample_id"])

        return {
            "skeleton": skeletons,
            "audio": audios,
            "labels": labels,
            "lengths": lengths,
            "sample_ids": sample_ids,
        }
    else:
        # Training mode: Fixed window size
        return torch.utils.data.default_collate(batch)


def get_loaders(batch_size=config.BATCH_SIZE, num_workers=2, max_samples=None):
    """
    Returns train, val, and test dataloaders.
    """
    train_ds = GestureDataset(
        "train", mode="train", augment=True, max_samples=max_samples
    )
    val_ds = GestureDataset(
        "val", mode="inference", augment=False, max_samples=max_samples
    )
    test_ds = GestureDataset(
        "test", mode="inference", augment=False, max_samples=max_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Full sequence inference
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
