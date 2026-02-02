import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    """
    Dataset class for Multi-Modal Gesture Recognition (RGB-D + Audio).
    Handles loading, alignment, preprocessing, and augmentation.
    """

    def __init__(self, split="train", load_cached_data=True, limit=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load/save data from cache.
            limit (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.limit = limit
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

        # Load metadata
        metadata_file = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        self.metadata = pd.read_csv(metadata_file)

        if self.limit:
            self.metadata = self.metadata.iloc[: self.limit]

        # Load data (from cache or raw files)
        self.data = self._load_all_data(load_cached_data)

    def _load_all_data(self, load_cached_data):
        """
        Loads data from cache if available, otherwise processes raw files.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached {self.split} data from {self.cache_path}...")
                loaded = np.load(self.cache_path, allow_pickle=True)
                return loaded["data"]
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from raw files.")

        print(f"Processing raw {self.split} data...")
        processed_data = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]

            # 1. Load Skeleton Data
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            try:
                mat = scipy.io.loadmat(
                    mat_path, squeeze_me=True, struct_as_record=False
                )
                video_struct = mat["Video"]
                num_frames = getattr(video_struct, "NumFrames", 0)
                frame_rate = getattr(
                    video_struct, "FrameRate", 20.0
                )  # Default to 20 if missing
                frames = getattr(video_struct, "Frames", [])

                # Extract Skeleton Joints
                # Handle case where Frames might be empty or different structure
                if num_frames == 0 or len(frames) == 0:
                    # Skip empty sequences
                    continue

                # Pre-allocate skeleton array: (T, NumJoints, 3)
                skeleton_frames = np.zeros(
                    (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
                )

                # Accessing struct array in loop (scipy.io loads as object array)
                # We need to handle potential missing frames or structure variations
                for t in range(min(num_frames, len(frames))):
                    try:
                        skel = frames[t].Skeleton
                        # WorldPosition is usually an object with X, Y, Z attributes or fields
                        # Based on description: WorldPosition.X, etc.
                        # Sometimes it might be a struct.

                        # Helper to extract joint pos
                        # We need to iterate over specific joints in order
                        # The structure contains an array of joints?
                        # Description: "Skeleton Frame... contains joint positions... JointsType can be..."
                        # Usually Kinect data in Matlab is struct with fields like 'HipCenter', etc.
                        # OR an array of joints.
                        # Let's assume the provided indices map to the array of joints if it's an array,
                        # or we map names if it's a struct.
                        # However, standard MSRDailyActivity/Chalearn format often has `Skeleton.WorldPosition`
                        # as a (Joints x 3) matrix or `Skeleton(j).WorldPosition`.
                        # Let's inspect the prompt description again:
                        # "Skeleton Frame: An array of Skeleton structures... JointsType... WorldPosition"
                        # This implies `frames[t].Skeleton` is an array of joints.

                        current_skel = skel  # This should be the array of joints

                        if isinstance(current_skel, np.ndarray) or isinstance(
                            current_skel, list
                        ):
                            for k, joint_idx in enumerate(Config.UPPER_BODY_INDICES):
                                if joint_idx < len(current_skel):
                                    joint = current_skel[joint_idx]
                                    pos = joint.WorldPosition
                                    skeleton_frames[t, k, 0] = pos.X
                                    skeleton_frames[t, k, 1] = pos.Y
                                    skeleton_frames[t, k, 2] = pos.Z
                        else:
                            # Fallback if structure is different (e.g. single object)
                            pass

                    except Exception:
                        # Keep zero if extraction fails
                        pass

            except Exception as e:
                print(f"Error processing MAT {sample_id}: {e}")
                continue

            # 2. Load Audio Data & Compute MFCC
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            mfcc_features = np.zeros(
                (num_frames, Config.INPUT_DIM_AUDIO), dtype=np.float32
            )

            try:
                if os.path.exists(audio_path):
                    waveform, sample_rate = torchaudio.load(audio_path)

                    # Calculate hop_length to align with video frames
                    # video_frame_duration = 1 / frame_rate
                    # hop_length = int(sample_rate * video_frame_duration)
                    hop_length = (
                        int(sample_rate / frame_rate) if frame_rate > 0 else 1600
                    )

                    # Compute MFCC
                    mfcc_transform = T.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc=Config.N_MFCC,
                        melkwargs={
                            "n_fft": 2048,
                            "hop_length": hop_length,
                            "n_mels": 64,
                            "center": False,  # To align better with frames
                        },
                    )

                    # (1, n_mfcc, time) -> (n_mfcc, time)
                    mfcc = mfcc_transform(waveform).squeeze(0)
                    mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

                    # Align lengths
                    audio_len = mfcc.shape[0]
                    if audio_len >= num_frames:
                        mfcc_features = mfcc[:num_frames, :].numpy()
                    else:
                        mfcc_features[:audio_len, :] = mfcc.numpy()
                        # Pad remainder with zeros or last frame
            except Exception as e:
                # print(f"Error processing Audio {sample_id}: {e}")
                pass  # Keep zeros

            # 3. Process Labels
            # Labels are provided as start/end frames and ID
            # Create dense frame-wise labels
            labels_dense = np.zeros(num_frames, dtype=np.int64)

            # Parse labels string from metadata
            # The metadata 'labels' column is just a list of IDs, it doesn't have start/end info.
            # We need to read start/end from the MAT file again.

            try:
                labels_raw = getattr(video_struct, "Labels", [])

                def process_label_entry(obj):
                    try:
                        name = obj.Name
                        start = int(obj.Begin) - 1  # Matlab 1-based
                        end = int(obj.End)  # Inclusive
                        if name in Config.GESTURE_MAP:
                            gid = Config.GESTURE_MAP[name]
                            # Clip to valid range
                            start = max(0, start)
                            end = min(num_frames, end)
                            labels_dense[start:end] = gid
                    except AttributeError:
                        pass

                if isinstance(labels_raw, np.ndarray):
                    if labels_raw.ndim == 0:
                        process_label_entry(labels_raw.item())
                    else:
                        for l in labels_raw:
                            process_label_entry(l)
                else:
                    process_label_entry(labels_raw)

            except Exception:
                pass

            # 4. Compute Boundaries (1 if label changes from previous frame)
            # We treat background (0) to gesture transitions as boundaries too
            boundaries = np.zeros(num_frames, dtype=np.float32)
            if num_frames > 1:
                diff = labels_dense[1:] != labels_dense[:-1]
                boundaries[1:] = diff.astype(np.float32)

            # Store processed item
            processed_data.append(
                {
                    "sample_id": sample_id,
                    "skeleton": skeleton_frames,  # (T, J, 3)
                    "audio": mfcc_features,  # (T, D_audio)
                    "labels": labels_dense,  # (T,)
                    "boundaries": boundaries,  # (T,)
                }
            )

        # Save to cache
        if load_cached_data:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez_compressed(self.cache_path, data=processed_data)
            print(f"Saved {self.split} data to cache.")

        return processed_data

    def _apply_physically_consistent_noise(self, positions):
        """
        Applies temporally smooth Gaussian noise to joint positions.
        positions: (T, J, 3)
        """
        T_dim, J_dim, C_dim = positions.shape

        # 1. Generate Gaussian Noise
        noise = np.random.normal(0, Config.NOISE_STD, size=(T_dim, J_dim, C_dim))

        # 2. Apply Temporal Low-Pass Filter (Moving Average)
        # Simple box filter
        window_size = Config.TEMPORAL_SMOOTHING_WINDOW
        kernel = np.ones(window_size) / window_size

        # Apply along time axis for each joint/coord
        # Efficient implementation using apply_along_axis or simple convolution loop
        # Since J*C is small (12*3=36), loop is fine
        noise_smooth = np.zeros_like(noise)
        for j in range(J_dim):
            for c in range(C_dim):
                noise_smooth[:, j, c] = np.convolve(noise[:, j, c], kernel, mode="same")

        # 3. Add to positions
        return positions + noise_smooth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load raw data
        # Scale skeleton to meters immediately
        skeleton = item["skeleton"].astype(np.float32) * Config.SCALE_FACTOR
        audio = item["audio"].astype(np.float32)
        labels = item["labels"].astype(np.int64)
        boundaries = item["boundaries"].astype(np.float32)

        # Augmentation (Train only)
        if self.split == "train":
            skeleton = self._apply_physically_consistent_noise(skeleton)

        # Normalization (Centering)
        # Center relative to HipCenter (Config.HIP_CENTER_INDEX)
        # HipCenter is index 0 in our extracted subset if UPPER_BODY_INDICES[0] corresponds to HipCenter
        # Config.UPPER_BODY_INDICES = [0, 1, ...] where 0 is HipCenter.
        # So in our extracted (T, 12, 3) array, index 0 is HipCenter.
        hip_center = skeleton[:, 0:1, :]  # (T, 1, 3)
        skeleton_centered = skeleton - hip_center

        # Compute Velocity
        # v[t] = p[t] - p[t-1], v[0] = 0
        velocity = np.zeros_like(skeleton_centered)
        velocity[1:] = skeleton_centered[1:] - skeleton_centered[:-1]

        # Flatten Skeleton Features: (T, J, 3) -> (T, J*3)
        T_dim = skeleton.shape[0]
        pos_flat = skeleton_centered.reshape(T_dim, -1)
        vel_flat = velocity.reshape(T_dim, -1)

        # Concatenate all features
        # [Position, Velocity, Audio]
        features = np.concatenate([pos_flat, vel_flat, audio], axis=1)

        # Convert to tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        boundaries = torch.from_numpy(boundaries).float()

        return features, labels, boundaries


def collate_fn(batch):
    """
    Collates a batch of variable length sequences.
    """
    features, labels, boundaries = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(f) for f in features])

    # Pad sequences
    # batch_first=True -> (B, T, D)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background
    boundaries_padded = pad_sequence(boundaries, batch_first=True, padding_value=0)

    # Create Mask (1 for valid, 0 for padding)
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()  # (B, T)

    return {
        "features": features_padded,  # (B, T, D)
        "labels": labels_padded,  # (B, T)
        "boundaries": boundaries_padded,  # (B, T)
        "mask": mask,  # (B, T)
        "lengths": lengths,  # (B,)
    }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Factory function to create dataloaders.
    """
    train_ds = GestureDataset("train", load_cached_data=True)
    val_ds = GestureDataset("val", load_cached_data=True)
    test_ds = GestureDataset("test", load_cached_data=True)

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
