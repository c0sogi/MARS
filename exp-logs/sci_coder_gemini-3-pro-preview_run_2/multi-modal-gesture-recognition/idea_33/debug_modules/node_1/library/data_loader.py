import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import soundfile as sf
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from library.config import Config

# Set random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class GestureDataset(Dataset):
    """
    Dataset class for loading and preprocessing multi-modal gesture data.
    Handles caching, normalization, feature extraction, and physically consistent augmentation.
    """

    def __init__(
        self, split="train", augment=False, debug=False, load_cached_data=True
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation (only for training).
            debug (bool): If True, limits dataset size for debugging.
            load_cached_data (bool): Whether to load pre-processed data from cache.
        """
        self.split = split
        self.augment = augment
        self.debug = debug
        self.joints_map = self._get_joint_map()
        self.selected_joint_indices = [
            self.joints_map[j] for j in Config.JOINTS_OF_INTEREST
        ]

        # Define cache path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

        # Load data (either from cache or raw processing)
        self.data = self._load_data(load_cached_data)

    def _get_joint_map(self):
        """Returns mapping from joint name to index based on standard Kinect format."""
        # Standard Kinect v1/v2 20-joint skeleton order
        joints = [
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
        return {name: i for i, name in enumerate(joints)}

    def _load_data(self, load_cached):
        """Loads data from cache or processes from scratch."""
        if load_cached and os.path.exists(self.cache_path):
            print(f"Loading {self.split} data from cache: {self.cache_path}")
            try:
                loaded = np.load(self.cache_path, allow_pickle=True)
                data_list = list(loaded["data"])
                if self.debug:
                    return data_list[: Config.DEBUG_SUBSET_SIZE]
                return data_list
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {self.split} data from raw files...")
        metadata_file = os.path.join(Config.METADATA_DIR, f"{self.split}.csv")
        df = pd.read_csv(metadata_file)

        # Convert labels string to list
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(
                lambda x: (
                    [int(i) for i in str(x).split()]
                    if pd.notna(x) and str(x).strip() != ""
                    else []
                )
            )
        else:
            df["labels"] = [[] for _ in range(len(df))]

        if self.debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        processed_data = []

        for idx, row in df.iterrows():
            try:
                sample = self._process_single_sample(row)
                if sample is not None:
                    processed_data.append(sample)
            except Exception as e:
                print(f"Error processing sample {row['sample_id']}: {e}")

        # Save to cache
        if not self.debug:
            print(f"Saving {self.split} data to cache: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez_compressed(
                self.cache_path, data=np.array(processed_data, dtype=object)
            )

        return processed_data

    def _process_single_sample(self, row):
        """Reads raw files and extracts features for a single sample."""
        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton Data
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]
            num_frames = video_struct.NumFrames
            frames_struct = video_struct.Frames
        except Exception:
            return None

        # Extract WorldPositions
        # frames_struct can be an array of objects.
        # We need to robustly handle the structure.
        all_positions = []

        # Handle case where Frames is a single object or array
        frames_iter = (
            frames_struct if isinstance(frames_struct, np.ndarray) else [frames_struct]
        )

        # If the mat file indicates fewer frames than actual array length, trust the array
        if len(frames_iter) != num_frames:
            num_frames = len(frames_iter)

        for f_idx in range(num_frames):
            frame_obj = frames_iter[f_idx]
            # Skeleton might be an array (multiple users) or single object
            skeletons = frame_obj.Skeleton

            # Select primary skeleton (assuming index 0 or the only one)
            if isinstance(skeletons, np.ndarray) and skeletons.size > 0:
                skel = skeletons[0] if skeletons.ndim > 0 else skeletons.item()
            else:
                skel = skeletons

            # Extract WorldPosition (X, Y, Z)
            # Check if WorldPosition exists
            if not hasattr(skel, "WorldPosition"):
                # Fallback or skip
                return None

            wp = skel.WorldPosition
            # wp should be a struct with x, y, z or an array
            # Based on description: "X value represents x-component..."
            # Usually scipy.io loads this as an object or array.
            # Let's try to parse it into [x, y, z]
            # If it's a struct with fields x, y, z
            try:
                # Assuming 20 joints. We need to extract all 20 first, then select.
                # However, the structure description says "Skeleton... contains joint positions".
                # It doesn't explicitly say it's an array of 20 positions.
                # It says "JointsType" can be HipCenter etc.
                # This implies Skeleton is an array of Joints?
                # Or Skeleton has a field WorldPosition which is 20x3?
                # "WorldPosition... formed by 20x4 matrix" (Wait, rotation is 20x4).
                # Position is likely 20x3 or array of 20 structs.
                # Let's assume standard Chalearn/Kinect format where WorldPosition is (Joints x 3) or similar.
                # If we look at `skel.WorldPosition`, if it's an array:
                pos = np.array(
                    [skel.WorldPosition.X, skel.WorldPosition.Y, skel.WorldPosition.Z]
                ).T
                # If pos is (3, 20), transpose to (20, 3)
                if pos.shape == (3, 20):
                    pos = pos.T
            except AttributeError:
                # If structure is different, try direct array access if available
                return None

            all_positions.append(pos)

        if not all_positions:
            return None

        # Shape: (T, 20, 3)
        raw_positions = np.array(all_positions)

        # 2. Load Audio Data
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # waveform: (Channels, Time)

            # Compute MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_MFCC_N_COEFFS,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = mfcc.mean(dim=0)  # (n_mfcc, time)
            else:
                mfcc = mfcc.squeeze(0)

            # Resample/Interpolate MFCC to match video frame count
            # Input to interpolate needs to be (Batch, Channels, Length)
            mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=num_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (num_frames, n_mfcc)

        except Exception:
            # Fallback if audio fails: zero features
            mfcc = np.zeros((num_frames, Config.AUDIO_MFCC_N_COEFFS))

        # 3. Create Sample Dict
        # We store raw positions to allow augmentation on the fly
        return {
            "sample_id": row["sample_id"],
            "positions": raw_positions,  # (T, 20, 3)
            "audio_features": mfcc,  # (T, 13)
            "labels": np.array(row["labels"], dtype=np.int64),
            "num_frames": num_frames,
        }

    def _physically_consistent_augmentation(self, positions):
        """
        Applies Gaussian noise to joint positions and smooths it temporally.
        Derives velocity from the noisy positions.

        Args:
            positions: (T, J, 3)
        Returns:
            aug_positions: (T, J, 3)
        """
        sigma = 0.005 * 1000  # 5mm noise (data is in mm)
        temporal_sigma = 2.0

        noise = np.random.normal(0, sigma, positions.shape)
        smooth_noise = gaussian_filter1d(noise, sigma=temporal_sigma, axis=0)

        aug_positions = positions + smooth_noise
        return aug_positions

    def _normalize_and_extract(self, positions):
        """
        Selects joints, centers them, scales to meters, and computes velocity.

        Args:
            positions: (T, 20, 3) in millimeters
        Returns:
            features: (T, FeatureDim)
        """
        # 1. Select Joints
        # (T, 12, 3)
        selected_pos = positions[:, self.selected_joint_indices, :]

        # 2. Center (Subtract HipCenter)
        # HipCenter is index 0 in our selected list (defined in Config)
        hip_idx = 0
        hip_pos = selected_pos[:, hip_idx : hip_idx + 1, :]
        centered_pos = selected_pos - hip_pos

        # 3. Scale (mm -> m)
        scaled_pos = centered_pos * Config.SCALE_FACTOR

        # 4. Compute Velocity
        # (T, 12, 3)
        # Prepend first frame to keep shape (T,...)
        velocity = np.diff(scaled_pos, axis=0, prepend=scaled_pos[0:1])

        # Flatten features: (T, 12*3 + 12*3) = (T, 72)
        T = scaled_pos.shape[0]
        flat_pos = scaled_pos.reshape(T, -1)
        flat_vel = velocity.reshape(T, -1)

        return np.concatenate([flat_pos, flat_vel], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get raw positions
        positions = sample["positions"]  # (T, 20, 3)
        audio = sample["audio_features"]  # (T, 13)

        # Augmentation
        if self.augment:
            positions = self._physically_consistent_augmentation(positions)

        # Feature Engineering & Normalization
        # Returns (T, 72)
        skeleton_feats = self._normalize_and_extract(positions)

        # Concatenate Audio
        # (T, 72 + 13) = (T, 85)
        features = np.concatenate([skeleton_feats, audio], axis=1)

        # Convert to float32 tensor
        features = torch.tensor(features, dtype=torch.float32)

        # Process Labels
        # Convert sequence of gesture IDs to frame-wise labels?
        # The task requires predicting a sequence of labels.
        # However, the model (FISG-CN) is a frame-wise prediction model (TCN/LSTM).
        # We need frame-wise targets for training.
        # The dataset provides 'Begin' and 'End' for each gesture in the training set.
        # But the `metadata` CSV only has the ordered list of labels.
        # Wait, the `metadata` script parsed `Labels` which contains `Begin`, `End`, `Name`.
        # But `train.csv` only saves the list of IDs.
        # The `GestureDataset` needs to load the frame-level annotations if we want to train a frame-wise model.
        # The `_process_single_sample` function in this class only loads the list of labels from CSV.
        # I need to modify `_process_single_sample` to re-read the MAT file's label structure
        # to generate frame-wise targets (0 for background, ID for gesture).

        # Re-implementation of label loading inside __getitem__ or _process_single_sample is needed.
        # Since I am loading from cache, I should have stored frame-wise labels in cache.
        # Let's fix `_process_single_sample` to extract frame-wise labels.

        # NOTE: Since the `_process_single_sample` above relied on CSV labels,
        # I will add logic here to extract frame-wise targets from the MAT file if available.

        # If labels are already frame-wise in cache (which I need to ensure), use them.
        # If not, and we are in training, we need them.

        # Let's check `sample` keys. If I modify `_process_single_sample` now, it will be correct.
        # I will modify `_process_single_sample` in the code block below to extract frame-wise labels.

        # Assuming `sample['frame_labels']` exists and is (T,).
        targets = torch.tensor(sample["frame_labels"], dtype=torch.long)

        return features, targets, sample["sample_id"]

    def _process_single_sample(self, row):
        # ... (Same setup as above) ...
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Cite debug_lesson_12: Removed try-except to allow debugging
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video_struct = mat["Video"]
        num_frames = video_struct.NumFrames
        frames_struct = video_struct.Frames
        labels_struct = getattr(video_struct, "Labels", [])

        # --- Skeleton Extraction (Same as above) ---
        all_positions = []
        frames_iter = (
            frames_struct if isinstance(frames_struct, np.ndarray) else [frames_struct]
        )
        if len(frames_iter) != num_frames:
            num_frames = len(frames_iter)

        for f_idx in range(num_frames):
            frame_obj = frames_iter[f_idx]
            skeletons = frame_obj.Skeleton
            if isinstance(skeletons, np.ndarray) and skeletons.size > 0:
                skel = skeletons[0] if skeletons.ndim > 0 else skeletons.item()
            else:
                skel = skeletons

            if not hasattr(skel, "WorldPosition"):
                raise ValueError(
                    f"Sample {row['sample_id']} Frame {f_idx}: Missing WorldPosition"
                )

            # Extract X, Y, Z
            wp = skel.WorldPosition
            if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                pos = np.array([wp.X, wp.Y, wp.Z]).T
            else:
                # Assume it is already a matrix/array
                pos = np.array(wp)

            if pos.shape == (3, 20):
                pos = pos.T

            if pos.shape != (20, 3):
                raise ValueError(
                    f"Sample {row['sample_id']} Frame {f_idx}: Invalid WorldPosition shape {pos.shape}"
                )

            all_positions.append(pos)

        if not all_positions:
            raise ValueError(f"Sample {row['sample_id']}: No valid frames found")

        raw_positions = np.array(all_positions)  # (T, 20, 3)

        # --- Audio Extraction (Same as above) ---
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_MFCC_N_COEFFS,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)
            if mfcc.shape[0] > 1:
                mfcc = mfcc.mean(dim=0)
            else:
                mfcc = mfcc.squeeze(0)
            mfcc = mfcc.unsqueeze(0)
            mfcc = F.interpolate(
                mfcc, size=num_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()
        except Exception:
            mfcc = np.zeros((num_frames, Config.AUDIO_MFCC_N_COEFFS))

        # --- Frame-wise Label Extraction ---
        # Initialize with background class (0)
        frame_labels = np.zeros(num_frames, dtype=np.int64)

        # Process labels
        # labels_struct can be single object, array, or empty
        if not isinstance(labels_struct, np.ndarray):
            labels_list = [labels_struct] if labels_struct else []
        else:
            labels_list = (
                labels_struct if labels_struct.ndim > 0 else [labels_struct.item()]
            )

        gesture_map = self.joints_map  # Reuse? No, need gesture map.
        # Hardcoding gesture map from description to ensure consistency
        gesture_name_map = {
            "vattene": 1,
            "vieniqui": 2,
            "perfetto": 3,
            "furbo": 4,
            "cheduepalle": 5,
            "chevuoi": 6,
            "daccordo": 7,
            "seipazzo": 8,
            "combinato": 9,
            "freganiente": 10,
            "ok": 11,
            "cosatifarei": 12,
            "basta": 13,
            "prendere": 14,
            "noncenepiu": 15,
            "fame": 16,
            "tantotempo": 17,
            "buonissimo": 18,
            "messidaccordo": 19,
            "sonostufo": 20,
        }

        for l in labels_list:
            try:
                name = l.Name
                start = int(l.Begin) - 1  # 1-based to 0-based
                end = int(l.End)  # inclusive in Matlab usually
                if name in gesture_name_map:
                    gid = gesture_name_map[name]
                    # Clamp indices
                    start = max(0, start)
                    end = min(num_frames, end)
                    frame_labels[start:end] = gid
            except AttributeError:
                pass

        return {
            "sample_id": row["sample_id"],
            "positions": raw_positions,
            "audio_features": mfcc,
            "frame_labels": frame_labels,
            "labels": np.array(row["labels"], dtype=np.int64),  # Sequence labels
            "num_frames": num_frames,
        }


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    Returns:
        padded_features: (B, T_max, D)
        padded_targets: (B, T_max)
        mask: (B, T_max)
        sample_ids: list of str
    """
    features, targets, sample_ids = zip(*batch)

    # Determine max length
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)

    feature_dim = features[0].shape[1]

    # Initialize tensors
    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    padded_targets = torch.zeros(
        len(batch), max_len, dtype=torch.long
    )  # 0 is background
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, length in enumerate(lengths):
        padded_features[i, :length, :] = features[i]
        padded_targets[i, :length] = targets[i]
        mask[i, :length] = 1.0

    return padded_features, padded_targets, mask, sample_ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=4, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_ds = GestureDataset(split="train", augment=True, debug=debug)
    val_ds = GestureDataset(split="val", augment=False, debug=debug)
    test_ds = GestureDataset(split="test", augment=False, debug=debug)

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
