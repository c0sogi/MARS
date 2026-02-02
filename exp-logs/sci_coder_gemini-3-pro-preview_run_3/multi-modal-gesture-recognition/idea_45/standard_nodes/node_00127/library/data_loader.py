import os
import json
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library import config

# ==========================================
# 1. Helper Functions
# ==========================================


def polymorphic_mat_parser(mat_path):
    """
    Robustly parses .mat files to extract skeleton data, handling
    variations in structure (struct vs array vs cell).
    Returns: numpy array of shape (NumFrames, 20, 3)
    """
    try:
        # Load mat file
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        # Access dictionary keys (Cite debug_lesson_2)
        if "Video" not in mat:
            return None

        video = mat["Video"]
        # Unwrap 0-d array if necessary (Cite debug_lesson_16)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames

        # Handle case where Frames is a single object or empty
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        num_frames = len(frames)
        # 20 joints, 3 coordinates (X, Y, Z)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Polymorphic handling of Skeleton field
            target_skel = None

            # Case 1: Skeleton is a single object (struct)
            if hasattr(skel, "WorldPosition"):
                target_skel = skel
            # Case 2: Skeleton is an array/list (multiple skeletons)
            elif isinstance(skel, (np.ndarray, list)) and len(skel) > 0:
                # Assume first skeleton is the target (or could filter by UserIndex if strictly needed)
                # Usually the first valid tracked skeleton is the main user
                if hasattr(skel[0], "WorldPosition"):
                    target_skel = skel[0]

            if target_skel is not None:
                wp = target_skel.WorldPosition
                # WorldPosition should have X, Y, Z fields
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # Check if X, Y, Z are arrays (20 joints) or scalars
                    # They should be arrays of length 20
                    try:
                        # Ensure we have data for 20 joints.
                        # If data is missing/corrupt, we might get fewer.
                        # We fill what we can.
                        x = np.atleast_1d(wp.X)
                        y = np.atleast_1d(wp.Y)
                        z = np.atleast_1d(wp.Z)

                        n_joints = min(len(x), 20)
                        skeleton_data[i, :n_joints, 0] = x[:n_joints]
                        skeleton_data[i, :n_joints, 1] = y[:n_joints]
                        skeleton_data[i, :n_joints, 2] = z[:n_joints]
                    except Exception:
                        pass

        return skeleton_data

    except Exception as e:
        # print(f"Error parsing {mat_path}: {e}")
        return None


def process_audio(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCCs, and aligns to video frames.
    Returns: numpy array (target_num_frames, n_mfcc)
    """
    try:
        # Load audio
        # soundfile is robust
        y, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample if necessary (though config says 16k, data might vary)
        # We use torchaudio for consistency if we were using it for loading,
        # but here we just assume roughly correct or rely on MFCC parameters.
        # For simplicity and speed, we assume dataset is mostly consistent or
        # that MFCC extraction handles it. Ideally we should resample.
        # Let's use torchaudio for the whole pipeline to be safe on resampling.

        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, samples)

        if sr != config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=config.AUDIO_SR
            )
            waveform = resampler(waveform)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SR,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 2048, "hop_length": config.HOP_LENGTH, "n_mels": 64},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Resize to match video frames
        # Input to interpolate must be (Batch, Channels, Time)
        mfcc = mfcc.transpose(0, 1).unsqueeze(0)  # (1, n_mfcc, time)

        mfcc_resized = F.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )  # (1, n_mfcc, target_frames)

        mfcc_final = (
            mfcc_resized.squeeze(0).transpose(0, 1).numpy()
        )  # (target_frames, n_mfcc)

        # Per-sample Standardization (CMVN)
        mean = np.mean(mfcc_final, axis=0, keepdims=True)
        std = np.std(mfcc_final, axis=0, keepdims=True) + 1e-6
        mfcc_final = (mfcc_final - mean) / std

        return mfcc_final

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_num_frames, config.N_MFCC), dtype=np.float32)


def process_skeleton_features(raw_skeleton, augment=False):
    """
    Applies Kinematically Consistent Augmentation and Feature Engineering.
    raw_skeleton: (T, 20, 3)
    Returns: (T, FeatureDim)
    """
    # 1. Root-Relative Centering
    # Assume Joint 0 is HipCenter
    root = raw_skeleton[:, 0:1, :]  # (T, 1, 3)
    centered_skel = raw_skeleton - root

    # 2. Augmentation (Train Only)
    if augment:
        # Random Rotation around Y-axis
        theta = np.random.uniform(-0.3, 0.3)  # +/- ~17 degrees
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: (T, 20, 3) @ (3, 3) -> (T, 20, 3)
        # Reshape to (T*20, 3) for matmul
        T, J, C = centered_skel.shape
        flat_skel = centered_skel.reshape(-1, 3)
        rotated_skel = np.dot(flat_skel, rotation_matrix.T).reshape(T, J, C)

        # Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        centered_skel = rotated_skel * scale

    # 3. Derivatives (Kinematically Consistent: Derive AFTER Augmentation)
    # Pad with first frame to maintain length
    velocity = np.diff(centered_skel, axis=0, prepend=centered_skel[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # 4. Physical Scaling (controlled by config)
    # Apply to all kinematic features
    pos_feat = centered_skel * config.SKELETON_SCALE
    vel_feat = velocity * config.SKELETON_SCALE
    acc_feat = acceleration * config.SKELETON_SCALE

    # 5. Flatten and Concatenate
    # (T, 20, 3) -> (T, 60)
    pos_flat = pos_feat.reshape(pos_feat.shape[0], -1)
    vel_flat = vel_feat.reshape(vel_feat.shape[0], -1)
    acc_flat = acc_feat.reshape(acc_feat.shape[0], -1)

    # Concatenate: (T, 180)
    features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)

    return features


# ==========================================
# 2. Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self, split, load_cached_data=True, augment=False, debug_sample_size=None
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
            augment (bool): Whether to apply augmentation (usually True for train).
            debug_sample_size (int): Limit dataset size for debugging.
        """
        self.split = split
        self.augment = augment
        self.debug_sample_size = debug_sample_size

        # Determine paths
        if split == "train":
            self.metadata_path = config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = config.VAL_METADATA_PATH
        else:
            self.metadata_path = config.TEST_METADATA_PATH

        self.cache_prefix = os.path.join(config.CACHE_DIR, f"dataset_{split}")

        # Load Data
        self.data_indices = (
            []
        )  # List of (sample_idx, start_frame, end_frame) for windows
        self.full_sequences = []  # List of dicts with 'features', 'labels', 'id'

        self._load_data(load_cached_data)

        # Prepare Windows if Training
        if self.split == "train":
            self._prepare_windows()

    def _load_data(self, load_cached_data):
        """
        Handles caching logic: Load from .npy or process from scratch.
        """
        feat_cache = self.cache_prefix + "_features.npy"
        lbl_cache = self.cache_prefix + "_labels.npy"
        offset_cache = self.cache_prefix + "_offsets.npy"
        ids_cache = self.cache_prefix + "_ids.json"

        loaded = False
        if load_cached_data:
            if (
                os.path.exists(feat_cache)
                and os.path.exists(lbl_cache)
                and os.path.exists(offset_cache)
                and os.path.exists(ids_cache)
            ):
                try:
                    # Load raw arrays
                    all_features = np.load(feat_cache)
                    all_labels = np.load(lbl_cache)
                    offsets = np.load(offset_cache)
                    with open(ids_cache, "r") as f:
                        sample_ids = json.load(f)

                    # Reconstruct list of sequences
                    for i, sid in enumerate(sample_ids):
                        start, end = offsets[i]
                        seq_feat = all_features[start:end]
                        seq_lbl = all_labels[start:end]
                        self.full_sequences.append(
                            {"id": sid, "features": seq_feat, "labels": seq_lbl}
                        )

                    print(f"Loaded {self.split} data from cache.")
                    loaded = True
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing...")

        if not loaded:
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads metadata, processes files, and saves to cache (No Pickle).
        """
        print(f"Processing {self.split} data from scratch...")
        df = pd.read_csv(self.metadata_path)

        if self.debug_sample_size:
            df = df.head(self.debug_sample_size)

        all_features_list = []
        all_labels_list = []
        offsets = []
        sample_ids = []
        current_offset = 0

        for _, row in df.iterrows():
            sid = row["sample_id"]
            # Paths
            mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

            # 1. Parse Skeleton
            raw_skel = polymorphic_mat_parser(mat_path)
            if raw_skel is None:
                continue  # Skip corrupt samples

            num_frames = raw_skel.shape[0]
            if num_frames < 5:
                continue  # Skip extremely short

            # 2. Process Features (No Augmentation during caching/loading base data)
            # We apply augmentation in __getitem__ for training
            # But wait, to cache efficiently, we should cache the "Base" features
            # and augment on the fly?
            # Yes. However, `process_skeleton_features` does scaling/derivatives.
            # If we augment on the fly, we need raw skeleton.
            # BUT, caching raw skeleton (20*3 floats) is smaller than features (180 floats).
            # Let's cache the PRE-PROCESSED BASE features (no aug) to save compute,
            # OR cache raw data.
            # The prompt says "Kinematically Consistent... Augment... Derive".
            # This implies we MUST augment raw data.
            # So for TRAINING, we should cache RAW SKELETON + AUDIO.
            # For Val/Test, we can cache processed.
            # To keep it simple and unified: Cache RAW SKELETON (T, 60) and AUDIO (T, 13).
            # Wait, raw skeleton is (T, 20, 3) -> 60 floats.
            # Audio is 13 floats.
            # Total 73 floats per frame.
            # Then in __getitem__, we do the math.
            # This is fast enough.

            # 3. Process Audio
            audio_feat = process_audio(audio_path, num_frames)

            # 4. Create Labels
            labels = np.zeros(num_frames, dtype=np.int32)
            if self.split != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    lid = l["id"]
                    start = max(0, l["begin"] - 1)  # 1-based to 0-based
                    end = min(num_frames, l["end"])
                    labels[start:end] = lid

            # Flatten raw skeleton for storage
            raw_skel_flat = raw_skel.reshape(num_frames, -1)  # (T, 60)

            # Combine Raw Skeleton + Audio
            # We will split them in __getitem__
            combined_data = np.concatenate([raw_skel_flat, audio_feat], axis=1)

            all_features_list.append(combined_data)
            all_labels_list.append(labels)
            sample_ids.append(sid)

            length = len(labels)
            offsets.append([current_offset, current_offset + length])
            current_offset += length

            # Update in-memory list
            self.full_sequences.append(
                {"id": sid, "features": combined_data, "labels": labels}
            )

        # Save to cache
        if all_features_list:
            big_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
            big_labels = np.concatenate(all_labels_list, axis=0).astype(np.int32)
            offsets_arr = np.array(offsets, dtype=np.int32)

            np.save(self.cache_prefix + "_features.npy", big_features)
            np.save(self.cache_prefix + "_labels.npy", big_labels)
            np.save(self.cache_prefix + "_offsets.npy", offsets_arr)
            with open(self.cache_prefix + "_ids.json", "w") as f:
                json.dump(sample_ids, f)

            print(f"Saved {self.split} data to cache.")
        else:
            print("Warning: No valid data found to cache.")

    def _prepare_windows(self):
        """
        Creates a list of windows for training.
        """
        self.windows = []
        for seq_idx, seq in enumerate(self.full_sequences):
            num_frames = seq["features"].shape[0]

            # If sequence is shorter than window, pad it later (or skip?)
            # We will handle padding in __getitem__
            # Just add one window starting at 0 if short
            if num_frames < config.WINDOW_SIZE:
                self.windows.append((seq_idx, 0))
                continue

            # Slide
            for start in range(0, num_frames - config.WINDOW_SIZE + 1, config.STRIDE):
                self.windows.append((seq_idx, start))

            # Handle remainder? usually strict sliding is fine.
            # If we want to cover the end:
            if (
                num_frames > config.WINDOW_SIZE
                and (num_frames - config.WINDOW_SIZE) % config.STRIDE != 0
            ):
                self.windows.append((seq_idx, num_frames - config.WINDOW_SIZE))

    def __len__(self):
        if self.split == "train":
            return len(self.windows)
        else:
            return len(self.full_sequences)

    def __getitem__(self, idx):
        if self.split == "train":
            # Window Mode
            seq_idx, start_frame = self.windows[idx]
            seq = self.full_sequences[seq_idx]

            raw_data = seq["features"]  # (T, 73) [60 skel, 13 audio]
            labels = seq["labels"]

            # Handle Short Sequences (Padding)
            total_frames = raw_data.shape[0]
            if total_frames < config.WINDOW_SIZE:
                # Pad with zeros
                pad_len = config.WINDOW_SIZE - total_frames
                raw_window = np.pad(raw_data, ((0, pad_len), (0, 0)), mode="constant")
                lbl_window = np.pad(
                    labels, (0, pad_len), mode="constant", constant_values=0
                )
            else:
                raw_window = raw_data[start_frame : start_frame + config.WINDOW_SIZE]
                lbl_window = labels[start_frame : start_frame + config.WINDOW_SIZE]

            # Split Skeleton and Audio
            # Skeleton is first 60 cols (20 joints * 3)
            raw_skel_window = raw_window[:, :60].reshape(-1, 20, 3)
            audio_window = raw_window[:, 60:]

            # Process Skeleton (Augment -> Derive -> Scale)
            skel_features = process_skeleton_features(
                raw_skel_window, augment=self.augment
            )

            # Combine
            # Audio is already standardized in process_audio
            final_features = np.concatenate([skel_features, audio_window], axis=1)

            return torch.tensor(final_features, dtype=torch.float32), torch.tensor(
                lbl_window, dtype=torch.long
            )

        else:
            # Full Sequence Mode (Val/Test)
            seq = self.full_sequences[idx]
            raw_data = seq["features"]
            labels = seq["labels"]
            sid = seq["id"]

            raw_skel = raw_data[:, :60].reshape(-1, 20, 3)
            audio = raw_data[:, 60:]

            # Process Skeleton (No Augment)
            skel_features = process_skeleton_features(raw_skel, augment=False)

            final_features = np.concatenate([skel_features, audio], axis=1)

            # Return tuple with ID for tracking
            return (
                torch.tensor(final_features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                sid,
            )
