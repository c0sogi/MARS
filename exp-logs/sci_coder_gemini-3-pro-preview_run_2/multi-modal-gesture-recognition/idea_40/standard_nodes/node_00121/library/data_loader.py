import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torchaudio
import torch.nn.functional as F
from library.config import Config
from library.utils import compute_bone_vectors

# Gesture Vocabulary Mapping (Injected here as it's not in the provided Config class code)
GESTURE_MAP = {
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


class GestureDataset(Dataset):
    def __init__(self, samples, augment=False):
        """
        Args:
            samples (list): List of dictionaries containing 'joints', 'audio', 'frame_labels', 'id'.
            augment (bool): Whether to apply augmentation.
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, joints):
        """
        Applies temporally correlated Gaussian noise to joint positions.
        joints: (T, J, 3) numpy array
        """
        noise_std = 0.005  # 5mm noise
        noise = np.random.normal(0, noise_std, size=joints.shape)

        # Apply temporal smoothing (moving average)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size

        noise_smooth = np.zeros_like(noise)
        # Apply convolution along time axis (axis 0) for each joint/coord
        # Optimization: Reshape to (T, J*3) -> convolve -> reshape back
        T, J, C = joints.shape
        noise_flat = noise.reshape(T, -1)
        noise_smooth_flat = np.zeros_like(noise_flat)

        for i in range(noise_flat.shape[1]):
            noise_smooth_flat[:, i] = np.convolve(noise_flat[:, i], kernel, mode="same")

        noise_smooth = noise_smooth_flat.reshape(T, J, C)
        return joints + noise_smooth

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Raw Data
        # joints: (T, 20, 3)
        raw_joints = sample["joints"]
        audio_mfcc = sample["audio"]  # (T, 13)
        labels = sample["frame_labels"]  # (T,)

        # Select upper body joints
        joints = raw_joints[:, Config.SELECTED_JOINTS, :]  # (T, 12, 3)

        # 2. Normalization
        # Center around HipCenter (Index 0 in selected list)
        hip_center = joints[:, [Config.REF_JOINT_INDEX], :]  # (T, 1, 3)
        joints = joints - hip_center

        # Scale to meters
        joints = joints * Config.SCALE_FACTOR

        # 3. Augmentation (if enabled)
        if self.augment:
            joints = self._apply_augmentation(joints)

        # Convert to Tensor
        joints_t = torch.tensor(joints, dtype=torch.float32)
        audio_t = torch.tensor(audio_mfcc, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # 4. Feature Engineering

        # Velocities: P_t - P_{t-1}
        # Pad first frame with 0
        vel = torch.zeros_like(joints_t)
        vel[1:] = joints_t[1:] - joints_t[:-1]

        # Bone Vectors
        # joints_t is (T, 12, 3)
        bones_t = compute_bone_vectors(joints_t)  # (T, 11, 3)

        # Flatten features
        joints_flat = joints_t.reshape(joints_t.size(0), -1)  # (T, 36)
        vel_flat = vel.reshape(vel.size(0), -1)  # (T, 36)
        bones_flat = bones_t.reshape(bones_t.size(0), -1)  # (T, 33)

        # Concatenate: (T, 36+36+33+13) = (T, 118)
        features = torch.cat([joints_flat, vel_flat, bones_flat, audio_t], dim=1)

        return features, labels_t, sample["id"]


def parse_mat_file(mat_path):
    """
    Parses the .mat file to extract skeleton joints and frame-level labels.
    Returns:
        joints: (T, 20, 3) numpy array
        frame_labels: (T,) numpy array (int)
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, None

        video = mat["Video"]

        # Handle Frames (Skeleton)
        frames = getattr(video, "Frames", [])
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        actual_frames = len(frames)
        if actual_frames == 0:
            return None, None

        joints_list = []

        for i in range(actual_frames):
            frame = frames[i]
            # Skeleton might be an array (multiple users), take first
            skel = getattr(frame, "Skeleton", None)

            if isinstance(skel, np.ndarray) and len(skel) > 0:
                skel = skel[0]
            elif isinstance(skel, np.ndarray) and len(skel) == 0:
                joints_list.append(np.zeros((20, 3)))
                continue
            elif skel is None:
                joints_list.append(np.zeros((20, 3)))
                continue

            # Extract WorldPosition
            wp = getattr(skel, "WorldPosition", None)
            current_joints = np.zeros((20, 3))

            if wp is not None:
                if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                    current_joints = wp
                elif isinstance(wp, np.ndarray) and len(wp) == 20:
                    # Array of structs
                    for j in range(20):
                        try:
                            current_joints[j, 0] = wp[j].X
                            current_joints[j, 1] = wp[j].Y
                            current_joints[j, 2] = wp[j].Z
                        except:
                            pass

            joints_list.append(current_joints)

        joints = np.array(joints_list)  # (T, 20, 3)

        # Handle Labels
        labels_raw = getattr(video, "Labels", [])
        frame_labels = np.zeros(actual_frames, dtype=int)

        def process_label(obj):
            try:
                name = obj.Name
                start = int(obj.Begin) - 1  # 1-based to 0-based
                end = int(obj.End)

                if name in GESTURE_MAP:
                    gid = GESTURE_MAP[name]
                    start = max(0, start)
                    end = min(actual_frames, end)
                    if start < end:
                        frame_labels[start:end] = gid
            except:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0 and labels_raw.size > 0:
                process_label(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label(l)
        elif labels_raw is not None:
            process_label(labels_raw)

        return joints, frame_labels

    except Exception:
        return None, None


def process_audio(audio_path, target_frames):
    """
    Loads audio, extracts MFCC, and aligns to target_frames.
    """
    try:
        if not os.path.exists(audio_path):
            return torch.zeros((target_frames, Config.AUDIO_N_MFCC))

        waveform, sr = torchaudio.load(audio_path)

        if sr != Config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sr, Config.AUDIO_SR)
            waveform = resampler(waveform)

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SR,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={
                "n_fft": Config.AUDIO_N_FFT,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "n_mels": 64,
                "center": False,
            },
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

        if mfcc.size(1) == 0:
            return torch.zeros((target_frames, Config.AUDIO_N_MFCC))

        # Align to video frames using interpolation
        mfcc_in = mfcc.unsqueeze(0)  # (1, 13, Time)
        mfcc_out = F.interpolate(
            mfcc_in, size=target_frames, mode="linear", align_corners=False
        )

        return mfcc_out.squeeze(0).transpose(0, 1)  # (Target_frames, 13)

    except Exception:
        return torch.zeros((target_frames, Config.AUDIO_N_MFCC))


def process_dataset_split(metadata_path, cache_path, is_test=False):
    """
    Reads metadata, loads and parses raw files, saves to cache.
    """
    if not os.path.exists(metadata_path):
        return []

    df = pd.read_csv(metadata_path)
    samples = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        joints, frame_labels = parse_mat_file(data_path)

        if joints is None:
            continue

        num_frames = joints.shape[0]

        # If test set or parsing failed to find labels, frame_labels is zeros
        # For test set, this is expected.

        mfcc = process_audio(audio_path, num_frames)

        samples.append(
            {
                "id": sample_id,
                "joints": joints,  # (T, 20, 3)
                "audio": mfcc.numpy(),  # (T, 13)
                "frame_labels": frame_labels,  # (T,)
            }
        )

    np.savez_compressed(cache_path, samples=np.array(samples, dtype=object))
    return samples


def load_data(load_cached_data=True):
    """
    Loads train, val, and test data, using cache if available.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    splits = [
        (
            "train",
            os.path.join(Config.METADATA_DIR, "train.csv"),
            os.path.join(Config.CACHE_DIR, "train_data.npz"),
            False,
        ),
        (
            "val",
            os.path.join(Config.METADATA_DIR, "val.csv"),
            os.path.join(Config.CACHE_DIR, "val_data.npz"),
            False,
        ),
        (
            "test",
            os.path.join(Config.METADATA_DIR, "test.csv"),
            os.path.join(Config.CACHE_DIR, "test_data.npz"),
            True,
        ),
    ]

    loaded_splits = []

    for name, meta_path, cache_path, is_test in splits:
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)["samples"].tolist()
                loaded_splits.append(data)
            except:
                # Corrupt cache, reprocess
                data = process_dataset_split(meta_path, cache_path, is_test)
                loaded_splits.append(data)
        else:
            data = process_dataset_split(meta_path, cache_path, is_test)
            loaded_splits.append(data)

    return loaded_splits[0], loaded_splits[1], loaded_splits[2]


def collate_fn(batch):
    """
    Pads sequences and generates masks.
    """
    features, labels, ids = zip(*batch)

    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    max_len = lengths.max().item()

    # Pad features
    padded_features = torch.zeros(len(features), max_len, features[0].size(1))
    for i, f in enumerate(features):
        padded_features[i, : lengths[i], :] = f

    # Pad labels
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, : lengths[i]] = l

    # Create Mask (True for padding)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    return padded_features, padded_labels, lengths, mask, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_samples, val_samples, test_samples = load_data(load_cached_data)

    if debug:
        train_samples = train_samples[: Config.DEBUG_SUBSET_SIZE]
        val_samples = val_samples[: Config.DEBUG_SUBSET_SIZE]
        test_samples = test_samples[: Config.DEBUG_SUBSET_SIZE]

    train_dataset = GestureDataset(train_samples, augment=True)
    val_dataset = GestureDataset(val_samples, augment=False)
    test_dataset = GestureDataset(test_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
