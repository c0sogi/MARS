import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library import config, utils


def load_skeleton_data(mat_path, num_frames):
    """
    Parses .mat file, extracts selected joints, normalizes, and computes velocity.
    Returns: (T, 72) numpy array (Pos + Vel flattened).
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        frames = video.Frames

        # Initialize array: T x NumJoints x 3
        # Handle case where Frames might be a single object or array
        if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
            frames = [frames]

        actual_frames = len(frames)
        # Use the min of metadata num_frames and actual frames found
        T = min(num_frames, actual_frames)

        skeleton_data = np.zeros((T, len(config.SELECTED_JOINTS), 3), dtype=np.float32)

        for t in range(T):
            frame_skel = frames[t].Skeleton
            # frame_skel might be an array of skeletons (multiple users), we assume user 1 or index 0
            # The documentation says "User Index ... signifies that a tracked subject occupies the pixel"
            # We will take the first skeleton found if multiple exist, or the object itself
            if isinstance(frame_skel, np.ndarray):
                if frame_skel.size > 0:
                    joints = frame_skel[0].WorldPosition  # Assuming first tracked user
                else:
                    continue  # No skeleton
            else:
                joints = frame_skel.WorldPosition

            # Extract selected joints
            # The structure has fields like HipCenter, Spine, etc.
            # We need to access them dynamically
            for i, joint_name in enumerate(config.SELECTED_JOINTS):
                if hasattr(joints, joint_name):
                    pos = getattr(joints, joint_name)
                    # pos should be an object with X, Y, Z or array
                    # Documentation: "The X value represents..."
                    # Usually loaded as struct with .X, .Y, .Z or array [x,y,z] depending on scipy ver
                    # Let's try to access .X, .Y, .Z
                    try:
                        skeleton_data[t, i, 0] = pos.X
                        skeleton_data[t, i, 1] = pos.Y
                        skeleton_data[t, i, 2] = pos.Z
                    except AttributeError:
                        # Fallback if it's a simple array or different structure
                        pass

        # Normalization
        # 1. Center relative to HipCenter (Index 0 in SELECTED_JOINTS)
        hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_center

        # 2. Scale units (mm to meters)
        skeleton_data = skeleton_data * config.SCALE_FACTOR

        # Compute Velocity
        # V_t = P_t - P_{t-1}, V_0 = 0
        velocity = np.zeros_like(skeleton_data)
        velocity[1:] = skeleton_data[1:] - skeleton_data[:-1]

        # Flatten: (T, Joints*3)
        pos_flat = skeleton_data.reshape(T, -1)
        vel_flat = velocity.reshape(T, -1)

        # Concatenate: (T, Joints*3*2)
        features = np.concatenate([pos_flat, vel_flat], axis=1)

        return features

    except Exception as e:
        # Return zeros if failure, matching expected dim
        # print(f"Error loading skeleton {mat_path}: {e}")
        return np.zeros((num_frames, len(config.SELECTED_JOINTS) * 6), dtype=np.float32)


def load_audio_data(audio_path, target_frames):
    """
    Loads audio, extracts MFCC, and aligns to target_frames.
    Returns: (T, N_MFCC) numpy array.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract MFCC
        # We use a standard hop length, then interpolate
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

        # Align to video frames using interpolation
        if mfcc.shape[1] > 0 and target_frames > 0:
            mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (target_frames, n_mfcc)
            return mfcc.numpy()
        else:
            return np.zeros((target_frames, config.N_MFCC), dtype=np.float32)

    except Exception as e:
        # print(f"Error loading audio {audio_path}: {e}")
        return np.zeros((target_frames, config.N_MFCC), dtype=np.float32)


def get_frame_labels(mat_path, num_frames):
    """
    Extracts frame-wise labels and boundaries from .mat file.
    Returns:
        labels: (T,) int array (0=Background, 1-20=Gesture)
        boundaries: (T,) float array (1.0 at transition, 0.0 otherwise)
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    boundaries = np.zeros(num_frames, dtype=np.float32)

    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return labels, boundaries

        video = mat["Video"]
        raw_labels = getattr(video, "Labels", [])

        # Helper to process single label entry
        def process_entry(entry):
            try:
                name = entry.Name
                start = int(entry.Begin) - 1  # 1-based to 0-based
                end = int(entry.End) - 1

                if name in config.GESTURE_MAP:
                    gid = config.GESTURE_MAP[name]
                    # Clamp indices
                    start = max(0, min(start, num_frames - 1))
                    end = max(0, min(end, num_frames - 1))

                    # Fill labels
                    labels[start : end + 1] = gid

                    # Fill boundaries (start and end frames)
                    boundaries[start] = 1.0
                    boundaries[end] = 1.0
                    # Optional: dilate boundary slightly? keeping it sharp as per idea
            except AttributeError:
                pass

        if isinstance(raw_labels, np.ndarray):
            if raw_labels.ndim == 0:
                process_entry(raw_labels.item())
            else:
                for l in raw_labels:
                    process_entry(l)
        else:
            process_entry(raw_labels)

    except Exception:
        pass

    return labels, boundaries


def augment_physically_consistent(features):
    """
    Applies Gaussian noise to positions and re-derives velocity.
    features: (T, 85) -> [Pos(36), Vel(36), Audio(13)]
    """
    T = features.shape[0]
    num_pos = len(config.SELECTED_JOINTS) * 3

    # Split features
    pos = features[:, :num_pos]
    # vel = features[:, num_pos:num_pos*2] # We will overwrite this
    audio = features[:, num_pos * 2 :]

    # 1. Generate Noise
    noise = np.random.normal(0, 0.005, size=pos.shape).astype(
        np.float32
    )  # 5mm noise (since units are meters)

    # 2. Temporal Smoothing of Noise (Low-pass filter)
    # Simple moving average kernel of size 3
    noise_smooth = np.zeros_like(noise)
    for i in range(num_pos):
        noise_smooth[:, i] = np.convolve(noise[:, i], np.ones(3) / 3, mode="same")

    # 3. Add noise to positions
    pos_aug = pos + noise_smooth

    # 4. Re-calculate Velocity
    vel_aug = np.zeros_like(pos_aug)
    vel_aug[1:] = pos_aug[1:] - pos_aug[:-1]

    # Re-assemble
    return np.concatenate([pos_aug, vel_aug, audio], axis=1)


class GestureDataset(Dataset):
    def __init__(self, metadata_path, is_train=True, load_cached_data=True):
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_path)

        # Cache setup
        self.cache_dir = config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        split_name = (
            "train"
            if "train.csv" in metadata_path
            else "val" if "val.csv" in metadata_path else "test"
        )
        self.cache_file = os.path.join(self.cache_dir, f"{split_name}_data.npz")

        self.data = self._load_or_create_cache(load_cached_data)

    def _load_or_create_cache(self, load_cached):
        if load_cached and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}...")
            try:
                loaded = np.load(self.cache_file, allow_pickle=True)
                # Reconstruct list of dicts
                data_list = []
                # Keys in npz: 'sample_ids', 'features_0', 'labels_0', ...
                # This approach can be slow for many keys.
                # Better: save as object array
                return loaded["data"]
            except Exception as e:
                print(f"Cache load failed ({e}), reprocessing...")

        print(f"Processing dataset (Cached: {load_cached})...")
        data_list = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])
            num_frames = int(row["num_frames"])

            # Load Features
            skel_feats = load_skeleton_data(mat_path, num_frames)
            audio_feats = load_audio_data(audio_path, skel_feats.shape[0])

            # Ensure lengths match (Skeleton is master)
            T = skel_feats.shape[0]
            if audio_feats.shape[0] != T:
                # Truncate or pad audio
                if audio_feats.shape[0] > T:
                    audio_feats = audio_feats[:T]
                else:
                    pad_len = T - audio_feats.shape[0]
                    audio_feats = np.pad(audio_feats, ((0, pad_len), (0, 0)))

            features = np.concatenate([skel_feats, audio_feats], axis=1).astype(
                np.float32
            )

            # Load Targets
            if (
                self.is_train or "val" in self.cache_file
            ):  # Train and Val have labels in .mat
                labels, boundaries = get_frame_labels(mat_path, T)
            else:
                labels = np.zeros(T, dtype=np.int64)
                boundaries = np.zeros(T, dtype=np.float32)

            data_list.append(
                {
                    "sample_id": sample_id,
                    "features": features,
                    "labels": labels,
                    "boundaries": boundaries,
                }
            )

        # Save to cache
        print(f"Saving cache to {self.cache_file}...")
        np.savez_compressed(self.cache_file, data=np.array(data_list, dtype=object))

        return np.array(data_list, dtype=object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        features = item["features"]  # (T, 85)
        labels = item["labels"]  # (T,)
        boundaries = item["boundaries"]  # (T,)
        sample_id = item["sample_id"]

        # Augmentation
        if self.is_train:
            features = augment_physically_consistent(features)

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "boundaries": torch.from_numpy(boundaries),
            "sample_id": sample_id,
        }


def collate_fn(batch):
    # Sort by length for packing (optional but good practice)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    boundaries = [x["boundaries"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)

    # Pad sequences
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background
    padded_boundaries = pad_sequence(boundaries, batch_first=True, padding_value=0.0)

    # Generate Masks (Batch, Time)
    # 1 for valid, 0 for padding
    max_len = padded_features.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "features": padded_features,  # (B, T, D)
        "labels": padded_labels,  # (B, T)
        "boundaries": padded_boundaries,  # (B, T)
        "mask": mask,  # (B, T)
        "lengths": lengths,
        "sample_ids": sample_ids,
    }
