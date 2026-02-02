import os
import json
import numpy as np
import pandas as pd
import scipy.io
import scipy.interpolate
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed

# ==========================================
# 1. Robust Data Parsing
# ==========================================


class SkeletonParser:
    """
    Parses .mat files containing Kinect skeleton data with robust handling
    for polymorphic structures (struct arrays vs cell arrays).
    """

    @staticmethod
    def parse(mat_path):
        try:
            # Load mat file, don't squeeze me too much to keep structure checkable
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

            if "Video" not in mat:
                return None

            video = mat["Video"]
            # Unwrap 0-d array if present (Cite debug_lesson_16)
            if isinstance(video, np.ndarray) and video.ndim == 0:
                video = video.item()

            num_frames = getattr(video, "NumFrames", 0)
            frames = getattr(video, "Frames", [])

            if num_frames == 0 or len(frames) == 0:
                return None

            # Pre-allocate: (NumFrames, 20 Joints, 3 Coords)
            skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

            # Iterate frames
            for i in range(min(num_frames, len(frames))):
                frame_obj = frames[i]

                # Robust extraction of Skeleton field
                skel_obj = None
                if hasattr(frame_obj, "Skeleton"):
                    skel_obj = frame_obj.Skeleton

                # Check if skeleton object is valid and has WorldPosition
                # It might be a struct, or a numpy array containing a struct, or empty
                valid_skel = False
                world_pos = None

                if isinstance(skel_obj, scipy.io.matlab.mat_struct):
                    if hasattr(skel_obj, "WorldPosition"):
                        world_pos = skel_obj.WorldPosition
                        valid_skel = True
                elif isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                    # Sometimes it's wrapped in an array
                    item = skel_obj.item()
                    if isinstance(item, scipy.io.matlab.mat_struct) and hasattr(
                        item, "WorldPosition"
                    ):
                        world_pos = item.WorldPosition
                        valid_skel = True

                if valid_skel and world_pos is not None:
                    # WorldPosition should have X, Y, Z fields (arrays of 20)
                    # Sometimes they are scalars if only 1 joint? But dataset says 20.
                    # Usually X, Y, Z are (20,) arrays or similar
                    try:
                        # Ensure we cast to float array
                        xs = np.array(world_pos.X, dtype=np.float32).flatten()
                        ys = np.array(world_pos.Y, dtype=np.float32).flatten()
                        zs = np.array(world_pos.Z, dtype=np.float32).flatten()

                        if len(xs) == 20 and len(ys) == 20 and len(zs) == 20:
                            skeleton_data[i, :, 0] = xs
                            skeleton_data[i, :, 1] = ys
                            skeleton_data[i, :, 2] = zs
                        else:
                            # Fallback or partial? Treat as invalid for safety
                            pass
                    except Exception:
                        pass

            # Simple imputation: forward fill then backward fill
            # Check for zero-frames (where no skeleton was found)
            # We assume (0,0,0) is invalid for a skeleton in this context
            mask = np.any(skeleton_data != 0, axis=(1, 2))  # True if valid frame

            if not np.any(mask):
                return None  # No valid frames at all

            # Forward fill
            last_valid = skeleton_data[mask][0]
            for i in range(num_frames):
                if mask[i]:
                    last_valid = skeleton_data[i]
                else:
                    skeleton_data[i] = last_valid

            return skeleton_data

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return None


class AudioProcessor:
    """
    Handles audio loading and MFCC extraction aligned to video frames.
    """

    def __init__(self, target_sr=16000, n_mfcc=13):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

    def process(self, audio_path, num_video_frames):
        if not os.path.exists(audio_path):
            return np.zeros((num_video_frames, self.n_mfcc), dtype=np.float32)

        try:
            # Load audio
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # Mono

            # Resample if needed (simple approximation if close enough, else torchaudio)
            if sr != self.target_sr:
                # We assume 16k as per config, but let's use torchaudio for safety if needed
                # For this challenge, dataset is mostly 16k.
                pass

            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

            # Compute MFCC
            # Shape: (1, n_mfcc, time)
            mfcc = self.mfcc_transform(y_tensor)
            mfcc = mfcc.squeeze(0).numpy().T  # (time, n_mfcc)

            # Interpolate to match video frames
            if mfcc.shape[0] != num_video_frames:
                x_old = np.linspace(0, 1, mfcc.shape[0])
                x_new = np.linspace(0, 1, num_video_frames)
                f = scipy.interpolate.interp1d(
                    x_old, mfcc, axis=0, kind="linear", fill_value="extrapolate"
                )
                mfcc_aligned = f(x_new)
            else:
                mfcc_aligned = mfcc

            return mfcc_aligned.astype(np.float32)

        except Exception:
            return np.zeros((num_video_frames, self.n_mfcc), dtype=np.float32)


# ==========================================
# 2. Feature Extraction
# ==========================================


class FeatureExtractor:
    """
    Derives explicit spatial-kinematic features from raw skeleton data.
    """

    # Parent indices for 20 joints (Kinect V1/V2 mapping based on description list)
    # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head, 4:ShoulderLeft...
    # We define parent for each node. Root (0) has parent -1 (itself or 0 vector).
    PARENTS = [
        -1,  # 0 HipCenter -> Root
        0,  # 1 Spine -> HipCenter
        1,  # 2 ShoulderCenter -> Spine
        2,  # 3 Head -> ShoulderCenter
        2,  # 4 ShoulderLeft -> ShoulderCenter
        4,  # 5 ElbowLeft -> ShoulderLeft
        5,  # 6 WristLeft -> ElbowLeft
        6,  # 7 HandLeft -> WristLeft
        2,  # 8 ShoulderRight -> ShoulderCenter
        8,  # 9 ElbowRight -> ShoulderRight
        9,  # 10 WristRight -> ElbowRight
        10,  # 11 HandRight -> WristRight
        0,  # 12 HipLeft -> HipCenter
        12,  # 13 KneeLeft -> HipLeft
        13,  # 14 AnkleLeft -> KneeLeft
        14,  # 15 FootLeft -> AnkleLeft
        0,  # 16 HipRight -> HipCenter
        16,  # 17 KneeRight -> HipRight
        17,  # 18 AnkleRight -> KneeRight
        18,  # 19 FootRight -> AnkleRight
    ]

    SPINE_IDX = 1

    @staticmethod
    def augment(skeleton, rotation_range=30, scale_range=0.2):
        """
        Applies random rotation around Y-axis and scaling.
        skeleton: (T, 20, 3)
        """
        T, J, C = skeleton.shape

        # Random Rotation (Y-axis)
        theta_deg = np.random.uniform(-rotation_range, rotation_range)
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation
        # Reshape to (T*J, 3) for matmul
        flat_skel = skeleton.reshape(-1, 3)
        rotated_skel = np.dot(flat_skel, R.T)

        # Random Scaling
        scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
        augmented = rotated_skel * scale

        return augmented.reshape(T, J, C)

    @classmethod
    def compute_features(cls, skeleton, audio, augment=False):
        """
        skeleton: (T, 20, 3)
        audio: (T, 13)
        Returns: (T, 253)
        """
        if augment:
            skeleton = cls.augment(skeleton)

        T, J, C = skeleton.shape

        # 1. Root-Relative Positions
        # Subtract Spine position from all joints
        spine_pos = skeleton[:, cls.SPINE_IDX : cls.SPINE_IDX + 1, :]  # (T, 1, 3)
        rel_pos = skeleton - spine_pos  # (T, 20, 3)

        # 2. Bone Vectors
        # Vector from Parent to Child
        bone_vecs = np.zeros_like(skeleton)
        for child_idx, parent_idx in enumerate(cls.PARENTS):
            if parent_idx == -1:
                # Root has no parent, use 0 or global pos? Use 0 vector for consistency
                bone_vecs[:, child_idx, :] = 0
            else:
                bone_vecs[:, child_idx, :] = (
                    skeleton[:, child_idx, :] - skeleton[:, parent_idx, :]
                )

        # 3. Velocity (First Derivative)
        # Pad first frame with 0
        vel = np.zeros_like(rel_pos)
        vel[1:] = rel_pos[1:] - rel_pos[:-1]

        # 4. Acceleration (Second Derivative)
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Flatten Joint Features
        # (T, 20, 3) -> (T, 60)
        f_rel = rel_pos.reshape(T, -1)
        f_bone = bone_vecs.reshape(T, -1)
        f_vel = vel.reshape(T, -1)
        f_acc = acc.reshape(T, -1)

        # Concatenate all
        # Shape: (T, 20*3*4 + 13) = (T, 240 + 13) = (T, 253)
        features = np.concatenate([f_rel, f_bone, f_vel, f_acc, audio], axis=1)

        return features.astype(np.float32)


# ==========================================
# 3. Dataset & Caching
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        cache_path,
        mode="train",
        load_cached_data=True,
        debug_max=None,
    ):
        """
        mode: 'train', 'val', 'test'
        """
        self.mode = mode
        self.config = Config
        self.window_size = self.config.WINDOW_SIZE
        self.stride = self.config.STRIDE

        # Load Metadata
        self.meta_df = pd.read_csv(metadata_csv)
        if debug_max is not None:
            self.meta_df = self.meta_df.iloc[:debug_max]

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 1. Load or Process Data
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            self.samples = data["samples"]  # List of dicts or object array
        else:
            print(f"Processing data for {mode} (Cache miss)...")
            self.samples = self._process_and_cache(cache_path)

        # 2. Build Window Index
        # We need to map linear index to (sample_idx, start_frame)
        self.windows = []
        for s_idx, sample in enumerate(self.samples):
            num_frames = sample["skeleton"].shape[0]
            if num_frames < self.window_size:
                # Pad short sequences? Or skip?
                # For this challenge, most are long enough. If short, we pad in __getitem__
                self.windows.append((s_idx, 0))
            else:
                # Sliding window
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.windows.append((s_idx, start))

                # Ensure last frames are covered if not exact fit
                last_start = num_frames - self.window_size
                if (
                    last_start > 0
                    and (num_frames - self.window_size) % self.stride != 0
                ):
                    self.windows.append((s_idx, last_start))

        # 3. Load Stats for Normalization
        self.stats = None
        if os.path.exists(self.config.STATS_CACHE):
            self.stats = np.load(self.config.STATS_CACHE)
            self.mean = torch.from_numpy(self.stats["mean"]).float()
            self.std = torch.from_numpy(self.stats["std"]).float()
        else:
            # If training, compute stats; if val/test, warn or wait
            if self.mode == "train":
                self._compute_stats()
            else:
                # Default identity
                self.mean = torch.zeros(self.config.INPUT_DIM)
                self.std = torch.ones(self.config.INPUT_DIM)

    def _process_and_cache(self, cache_path):
        samples = []
        audio_proc = AudioProcessor(
            target_sr=self.config.AUDIO_SAMPLE_RATE, n_mfcc=self.config.AUDIO_N_MFCC
        )

        for idx, row in self.meta_df.iterrows():
            # Paths
            mat_path = os.path.join(self.config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(self.config.INPUT_DIR, row["audio_path"])

            # Parse Skeleton
            skeleton = SkeletonParser.parse(mat_path)
            if skeleton is None:
                continue  # Skip corrupt files

            num_frames = skeleton.shape[0]

            # Parse Audio
            audio = audio_proc.process(audio_path, num_frames)

            # Create Labels
            labels = np.zeros(num_frames, dtype=np.int64)
            if self.mode != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # 1-based indices in mat to 0-based
                    start = max(0, l["begin"] - 1)
                    end = min(num_frames, l["end"])
                    labels[start:end] = l["id"]

            samples.append(
                {
                    "sample_id": row["sample_id"],
                    "skeleton": skeleton,  # (T, 20, 3)
                    "audio": audio,  # (T, 13)
                    "labels": labels,  # (T,)
                }
            )

        # Save
        np.savez_compressed(cache_path, samples=np.array(samples, dtype=object))
        return samples

    def _compute_stats(self):
        print("Computing normalization statistics...")
        # Accumulate sum and sq_sum
        # We'll use a subset or online algorithm.
        # Given dataset size, we can iterate all windows without augmentation.

        all_features = []
        # Sample 10% of windows to save time/memory
        indices = np.random.choice(
            len(self.windows), size=min(2000, len(self.windows)), replace=False
        )

        for idx in indices:
            s_idx, start = self.windows[idx]
            sample = self.samples[s_idx]

            skel_chunk = sample["skeleton"][start : start + self.window_size]
            audio_chunk = sample["audio"][start : start + self.window_size]

            # No augmentation for stats
            feats = FeatureExtractor.compute_features(
                skel_chunk, audio_chunk, augment=False
            )
            all_features.append(feats)

        all_features = np.concatenate(all_features, axis=0)  # (N*T, D)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0) + 1e-6  # Avoid div/0

        np.savez(self.config.STATS_CACHE, mean=mean, std=std)
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        print("Stats computed and saved.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_idx, start = self.windows[idx]
        sample = self.samples[s_idx]

        # Get raw data
        skel_full = sample["skeleton"]
        audio_full = sample["audio"]
        labels_full = sample["labels"]

        num_frames = skel_full.shape[0]

        # Handle short sequences (Padding)
        if num_frames < self.window_size:
            pad_len = self.window_size - num_frames
            # Pad skeleton with last frame
            skel_chunk = np.pad(skel_full, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            audio_chunk = np.pad(audio_full, ((0, pad_len), (0, 0)), mode="constant")
            labels_chunk = np.pad(
                labels_full, (0, pad_len), mode="constant", constant_values=0
            )
        else:
            end = start + self.window_size
            skel_chunk = skel_full[start:end]
            audio_chunk = audio_full[start:end]
            labels_chunk = labels_full[start:end]

        # Augmentation (only for training)
        do_augment = self.mode == "train"

        # Feature Extraction
        features = FeatureExtractor.compute_features(
            skel_chunk, audio_chunk, augment=do_augment
        )

        # Convert to Tensor
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels_chunk).long()

        # Normalize
        features = (features - self.mean) / self.std

        # For test, we might need metadata to reconstruct
        if self.mode == "test":
            return features, labels, sample["sample_id"], start

        return features, labels


# ==========================================
# 4. Data Loader Factory
# ==========================================


def get_dataloaders(debug_max=None):
    set_seed(Config.SEED)

    # Train
    train_ds = GestureDataset(
        metadata_csv=Config.TRAIN_CSV,
        cache_path=Config.TRAIN_CACHE,
        mode="train",
        load_cached_data=True,
        debug_max=debug_max,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_ds = GestureDataset(
        metadata_csv=Config.VAL_CSV,
        cache_path=Config.VAL_CACHE,
        mode="val",
        load_cached_data=True,
        debug_max=debug_max,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Test
    test_ds = GestureDataset(
        metadata_csv=Config.TEST_CSV,
        cache_path=Config.TEST_CACHE,
        mode="test",
        load_cached_data=True,
        debug_max=debug_max,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
