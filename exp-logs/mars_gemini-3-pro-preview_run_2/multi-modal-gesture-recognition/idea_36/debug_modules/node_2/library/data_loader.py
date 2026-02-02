import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    GESTURE_MAP,
    JOINTS_LIST,
    AUDIO_PARAMS,
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.utils import set_seed, generate_padding_mask

# Set seed for reproducibility
set_seed(SEED)


class GestureDataset(Dataset):
    def __init__(
        self,
        split="train",
        load_cached_data=True,
        max_samples=None,
        augment=False,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache.
            max_samples (int, optional): Limit dataset size for debugging.
            augment (bool): Apply data augmentation (only for training).
        """
        self.split = split
        self.augment = augment and (split == "train")

        # Determine metadata path
        if split == "train":
            self.metadata_path = TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache_path = os.path.join(CACHE_DIR, f"{split}_data.npz")

        # Load data
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading {split} data from cache: {self.cache_path}")
            self._load_cache()
        else:
            print(f"Processing {split} data from scratch...")
            self._process_and_cache(max_samples)

    def _load_cache(self):
        try:
            data = np.load(self.cache_path)
            self.all_features = data["features"]
            self.all_labels = data["labels"]
            self.all_boundaries = data["boundaries"]
            self.offsets = data["offsets"]
            self.num_samples = len(self.offsets)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            self._process_and_cache(None)

    def _process_and_cache(self, max_samples):
        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        if max_samples is not None:
            df = df.iloc[:max_samples]

        features_list = []
        labels_list = []
        boundaries_list = []
        offsets = []

        current_offset = 0

        for idx, row in df.iterrows():
            # Paths
            data_path = os.path.join(INPUT_DIR, row["data_path"])
            audio_path = os.path.join(INPUT_DIR, row["audio_path"])

            # Process
            try:
                feats, labs, bnds = self._process_single_sample(data_path, audio_path)

                # Append
                n_frames = feats.shape[0]
                if n_frames == 0:
                    continue

                features_list.append(feats)
                labels_list.append(labs)
                boundaries_list.append(bnds)

                offsets.append([current_offset, current_offset + n_frames])
                current_offset += n_frames

            except Exception as e:
                # print(f"Error processing {row['sample_id']}: {e}")
                continue

        # Concatenate
        if not features_list:
            # Create empty arrays if no data found (e.g. very small max_samples with errors)
            self.all_features = np.zeros(
                (0, 36 + AUDIO_PARAMS["n_mfcc"]), dtype=np.float32
            )
            self.all_labels = np.zeros((0,), dtype=np.int64)
            self.all_boundaries = np.zeros((0,), dtype=np.float32)
            self.offsets = np.zeros((0, 2), dtype=np.int64)
            self.num_samples = 0
            print("Warning: No valid samples processed.")
        else:
            self.all_features = np.concatenate(features_list, axis=0).astype(np.float32)
            self.all_labels = np.concatenate(labels_list, axis=0).astype(np.int64)
            self.all_boundaries = np.concatenate(boundaries_list, axis=0).astype(
                np.float32
            )
            self.offsets = np.array(offsets, dtype=np.int64)
            self.num_samples = len(self.offsets)

        # Save to cache
        np.savez(
            self.cache_path,
            features=self.all_features,
            labels=self.all_labels,
            boundaries=self.all_boundaries,
            offsets=self.offsets,
        )
        print(f"Saved {self.split} data to cache: {self.cache_path}")

    def _process_single_sample(self, mat_path, audio_path):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"MAT file missing: {mat_path}")

        # 1. Load Skeleton Data
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception:
            raise ValueError(f"Corrupt MAT file: {mat_path}")

        if "Video" not in mat:
            raise ValueError("Missing 'Video' struct")

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)

        # Extract Frames
        frames = getattr(video, "Frames", [])
        if isinstance(frames, np.ndarray) and len(frames) != num_frames:
            num_frames = len(frames)

        if num_frames == 0:
            raise ValueError("Zero frames")

        # Initialize Skeleton Array: (T, Joints, 3)
        skeleton_data = np.zeros((num_frames, len(JOINTS_LIST), 3), dtype=np.float32)

        # Parse Skeleton
        frames_iter = frames if isinstance(frames, np.ndarray) else [frames]

        for t, frame_obj in enumerate(frames_iter):
            if t >= num_frames:
                break

            if not hasattr(frame_obj, "Skeleton"):
                continue

            skel_obj = frame_obj.Skeleton

            # Handle array of skeletons (pick first valid)
            if isinstance(skel_obj, np.ndarray):
                if skel_obj.size == 0:
                    continue
                skel_obj = skel_obj[0]

            if skel_obj is None:
                continue

            # Extract joints
            for j_idx, joint_name in enumerate(JOINTS_LIST):
                if hasattr(skel_obj, joint_name):
                    joint_node = getattr(skel_obj, joint_name)
                    if hasattr(joint_node, "WorldPosition"):
                        pos = joint_node.WorldPosition
                        if hasattr(pos, "X"):
                            skeleton_data[t, j_idx, 0] = pos.X
                            skeleton_data[t, j_idx, 1] = pos.Y
                            skeleton_data[t, j_idx, 2] = pos.Z
                        elif isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                            skeleton_data[t, j_idx, :] = pos[:3]

        # 2. Normalize Skeleton
        # Convert mm to meters
        skeleton_data = skeleton_data / 1000.0

        # Center to HipCenter (Index 0)
        hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_center

        # 3. Audio Features
        audio_features = self._extract_audio(audio_path, num_frames)

        # 4. Construct Labels and Boundaries
        labels = np.zeros(num_frames, dtype=np.int64)
        boundaries = np.zeros(num_frames, dtype=np.float32)

        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            if not isinstance(raw_labels, (list, np.ndarray)):
                raw_labels = [raw_labels]

            for lbl in raw_labels:
                try:
                    name = lbl.Name
                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        start = int(lbl.Begin) - 1  # Matlab 1-based
                        end = int(lbl.End) - 1

                        start = max(0, start)
                        end = min(num_frames - 1, end)

                        if start <= end:
                            labels[start : end + 1] = gid
                            boundaries[start] = 1.0
                            boundaries[end] = 1.0
                except AttributeError:
                    continue

        # 5. Flatten Skeleton: (T, J*3)
        skeleton_flat = skeleton_data.reshape(num_frames, -1)

        # Concatenate: (T, J*3 + AudioDim)
        features = np.concatenate([skeleton_flat, audio_features], axis=1)

        return features, labels, boundaries

    def _extract_audio(self, audio_path, target_frames):
        dim = AUDIO_PARAMS["n_mfcc"]
        if not os.path.exists(audio_path):
            return np.zeros((target_frames, dim), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample
            if sample_rate != AUDIO_PARAMS["sample_rate"]:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, AUDIO_PARAMS["sample_rate"]
                )
                waveform = resampler(waveform)

            # Extract MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=AUDIO_PARAMS["sample_rate"],
                n_mfcc=AUDIO_PARAMS["n_mfcc"],
                melkwargs={
                    "n_fft": AUDIO_PARAMS["n_fft"],
                    "hop_length": AUDIO_PARAMS["hop_length"],
                    "n_mels": AUDIO_PARAMS["n_mels"],
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
            if mfcc.dim() == 3:
                mfcc = mfcc.mean(dim=0)  # (n_mfcc, time)

            # Interpolate
            mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
            mfcc_resampled = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )

            return mfcc_resampled.squeeze(0).transpose(0, 1).numpy()

        except Exception:
            return np.zeros((target_frames, dim), dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start, end = self.offsets[idx]

        # Retrieve raw data from cache
        feats = self.all_features[start:end]
        labels = self.all_labels[start:end]
        boundaries = self.all_boundaries[start:end]

        # Separate Skeleton (first 36 cols) and Audio
        num_skel_features = len(JOINTS_LIST) * 3
        skeleton_pos = feats[:, :num_skel_features]
        audio_feats = feats[:, num_skel_features:]

        # Augmentation (Train only)
        if self.augment:
            # Physically Consistent Augmentation: Add noise to position, then derive velocity
            noise = np.random.normal(0, 0.005, skeleton_pos.shape).astype(
                np.float32
            )  # 5mm std dev
            skeleton_pos = skeleton_pos + noise

        # Compute Explicit Velocity
        # V_t = P_t - P_{t-1}
        velocity = np.zeros_like(skeleton_pos)
        if skeleton_pos.shape[0] > 1:
            velocity[1:] = skeleton_pos[1:] - skeleton_pos[:-1]

        # Combine: Position (36) + Velocity (36) + Audio (13) = 85
        combined_features = np.concatenate(
            [skeleton_pos, velocity, audio_feats], axis=1
        )

        return {
            "features": torch.from_numpy(combined_features).float(),
            "targets": torch.from_numpy(labels).long(),
            "boundaries": torch.from_numpy(boundaries).float(),
        }


def collate_fn(batch):
    """
    Collates a list of samples into a batch.
    Pads sequences to the maximum length in the batch.
    """
    features = [b["features"] for b in batch]
    targets = [b["targets"] for b in batch]
    boundaries = [b["boundaries"] for b in batch]

    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    # Pad sequences
    # features: (B, T_max, D)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    # targets: (B, T_max), padding with 0 (background)
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )
    # boundaries: (B, T_max), padding with 0
    boundaries_padded = torch.nn.utils.rnn.pad_sequence(
        boundaries, batch_first=True, padding_value=0
    )

    # Generate mask
    mask = generate_padding_mask(lengths)

    return {
        "features": features_padded,
        "targets": targets_padded,
        "boundaries": boundaries_padded,
        "mask": mask,
        "lengths": lengths,
    }
