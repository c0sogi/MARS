import os
import numpy as np
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from library.config import Config
from library.utils import robust_load_mat, load_metadata


# Set fixed seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class GestureDataset(Dataset):
    """
    Dataset class for the PAK-RN model.
    Handles multi-modal data loading, alignment, caching, and windowing.
    """

    def __init__(self, split, load_cached_data=True):
        self.split = split
        self.is_train = split == "train"
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE_TRAIN

        # Ensure working directories exist
        Config.ensure_dirs()

        # Cache path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split}.npz")

        # Data containers
        self.samples_skel = []  # List of (T, 20, 3) arrays
        self.samples_audio = []  # List of (T, N_MFCC) arrays
        self.samples_labels = []  # List of (T,) arrays
        self.sample_ids = []  # List of strings

        # Load data
        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache()

        # Build indices for access
        self.indices = []
        self._build_indices()

    def _process_and_cache(self):
        """Loads raw data from disk, aligns streams, and saves to cache."""
        print(f"Processing {self.split} data from scratch...")
        df = load_metadata(self.split)

        data_dict = {}

        valid_count = 0
        for idx, row in df.iterrows():
            sample_id = row["sample_id"]

            # 1. Load Skeleton (already scaled to meters by robust_load_mat)
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            skeleton = robust_load_mat(mat_path)  # (T, 20, 3)

            num_frames = skeleton.shape[0]
            if num_frames == 0:
                continue

            # 2. Load and Align Audio
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            audio_features = self._extract_audio(audio_path, num_frames)  # (T, N_MFCC)

            # 3. Create Labels
            labels = np.zeros(num_frames, dtype=np.int64)
            if "labels" in row and isinstance(row["labels"], list):
                for label_info in row["labels"]:
                    lid = label_info["id"]
                    start = max(0, label_info["begin"] - 1)  # 1-based to 0-based
                    end = min(num_frames, label_info["end"])
                    if start < end:
                        labels[start:end] = lid

            # Store in memory
            self.samples_skel.append(skeleton)
            self.samples_audio.append(audio_features)
            self.samples_labels.append(labels)
            self.sample_ids.append(sample_id)

            # Store for caching
            data_dict[f"skel_{valid_count}"] = skeleton
            data_dict[f"audio_{valid_count}"] = audio_features
            data_dict[f"label_{valid_count}"] = labels
            data_dict[f"id_{valid_count}"] = sample_id

            valid_count += 1

        # Save to cache
        np.savez_compressed(self.cache_path, count=valid_count, **data_dict)
        print(f"Cached {valid_count} samples to {self.cache_path}")

    def _load_cache(self):
        """Loads pre-processed data from .npz cache."""
        print(f"Loading cached {self.split} data from {self.cache_path}...")
        try:
            with np.load(self.cache_path, allow_pickle=True) as data:
                count = data["count"]
                for i in range(count):
                    self.samples_skel.append(data[f"skel_{i}"])
                    self.samples_audio.append(data[f"audio_{i}"])
                    self.samples_labels.append(data[f"label_{i}"])
                    # Handle string loading from numpy
                    sid = data[f"id_{i}"]
                    self.sample_ids.append(str(sid))
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing...")
            self._process_and_cache()

    def _extract_audio(self, path, target_frames):
        """Loads wav, computes MFCC, and aligns to target_frames."""
        if not os.path.exists(path):
            return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(path)

            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Compute MFCC
            # n_fft=400 (25ms), hop_length=160 (10ms) at 16kHz
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=Config.AUDIO_SAMPLE_RATE,
                n_mfcc=Config.N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

            # Interpolate to match video frames
            # Input to interpolate must be (Batch, Channels, Time)
            if mfcc.shape[-1] != target_frames:
                mfcc = F.interpolate(
                    mfcc, size=target_frames, mode="linear", align_corners=False
                )

            # (Time, n_mfcc)
            return mfcc.squeeze(0).transpose(0, 1).numpy()

        except Exception:
            return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)

    def _build_indices(self):
        """Builds the list of items (windows or sequences) to retrieve."""
        if self.is_train:
            # Sliding window strategy
            for idx, skel in enumerate(self.samples_skel):
                num_frames = skel.shape[0]
                if num_frames < self.window_size:
                    # Pad short sequences
                    self.indices.append((idx, 0, True))  # True = needs padding
                else:
                    # Slide
                    for start in range(
                        0, num_frames - self.window_size + 1, self.stride
                    ):
                        self.indices.append((idx, start, False))

                    # Ensure last frame is covered if stride leaves a gap
                    last_start = num_frames - self.window_size
                    if last_start > 0 and (last_start % self.stride != 0):
                        self.indices.append((idx, last_start, False))
        else:
            # Full sequence strategy
            for idx in range(len(self.samples_skel)):
                self.indices.append((idx, 0, False))

    def _augment_skeleton(self, skeleton):
        """Applies random Y-axis rotation to skeleton (T, 20, 3)."""
        theta = np.random.uniform(-0.3, 0.3)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        shape = skeleton.shape
        flat = skeleton.reshape(-1, 3)
        rotated = flat @ R.T
        return rotated.reshape(shape)

    def _compute_kinematics(self, skeleton):
        """
        Computes velocity and acceleration.
        Input: (T, 20, 3)
        Output: (T, 20, 9) -> [Pos, Vel, Acc]
        """
        # Velocity
        vel = np.zeros_like(skeleton)
        vel[1:] = skeleton[1:] - skeleton[:-1]
        vel[0] = vel[1]  # Replicate first frame

        # Acceleration
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
        acc[0] = acc[1]

        # Concatenate: (T, 20, 3+3+3) = (T, 20, 9)
        return np.concatenate([skeleton, vel, acc], axis=-1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start_frame, needs_padding = self.indices[idx]

        skel = self.samples_skel[sample_idx]
        audio = self.samples_audio[sample_idx]
        labels = self.samples_labels[sample_idx]

        # Slicing / Windowing
        if self.is_train:
            if needs_padding:
                # Pad to window size
                pad_len = self.window_size - skel.shape[0]
                # Pad end with zeros or edge? Zero padding is safer for variable length
                skel = np.pad(skel, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
                audio = np.pad(audio, ((0, pad_len), (0, 0)), mode="edge")
                labels = np.pad(
                    labels, (0, pad_len), mode="constant", constant_values=0
                )
            else:
                end_frame = start_frame + self.window_size
                skel = skel[start_frame:end_frame]
                audio = audio[start_frame:end_frame]
                labels = labels[start_frame:end_frame]

        # Augmentation (Train only)
        if self.is_train:
            skel = self._augment_skeleton(skel)

        # Feature Engineering (Kinematics)
        # Flatten joints: (T, 20, 9) -> (T, 180)
        kinematics = self._compute_kinematics(skel)
        kinematics_flat = kinematics.reshape(kinematics.shape[0], -1)

        # Fusion
        # (T, 180) + (T, 13) -> (T, 193)
        features = np.concatenate([kinematics_flat, audio], axis=-1)

        # Convert to tensors
        features_t = torch.from_numpy(features).float()
        labels_t = torch.from_numpy(labels).long()

        if self.is_train:
            return features_t, labels_t
        else:
            # For validation/test, return sample ID as well for submission generation
            return features_t, labels_t, self.sample_ids[sample_idx]


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """Factory function to create dataloaders."""

    train_ds = GestureDataset("train")
    val_ds = GestureDataset("val")
    test_ds = GestureDataset("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test use batch_size=1 to handle variable sequence lengths
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
