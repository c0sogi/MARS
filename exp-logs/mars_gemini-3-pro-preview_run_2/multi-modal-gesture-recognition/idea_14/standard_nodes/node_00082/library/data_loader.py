import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import scipy.io
import torchaudio
from library.utils import set_seed


class DataLoaderConfig:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Data Parameters
    # 12 Joints: Head, Shoulders, Elbows, Wrists, Hands, Spine, HipCenter
    # Indices based on standard Kinect mapping for the subset
    JOINTS_LIST = [3, 2, 4, 8, 5, 9, 6, 10, 7, 11, 1, 0]
    NUM_JOINTS = 12
    AUDIO_N_MFCC = 13
    # Input Dim: (12*3 Pos) + (12*3 Vel) + 13 Audio = 85
    INPUT_DIM = (NUM_JOINTS * 3) * 2 + AUDIO_N_MFCC

    # Augmentation Parameters
    NOISE_SIGMA = 5.0  # Millimeters
    NOISE_SMOOTH_KERNEL = 5

    # Gesture Vocabulary
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


def load_mat_file(path):
    """Safely loads a .mat file."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return None


def process_audio(audio_path, target_frames):
    """Loads audio, computes MFCCs, and aligns them to video frames."""
    try:
        if not os.path.exists(audio_path):
            return torch.zeros((target_frames, DataLoaderConfig.AUDIO_N_MFCC))

        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=DataLoaderConfig.AUDIO_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = transform(waveform)  # (n_mfcc, time)
        mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

        # Interpolate to match target video frames
        if mfcc.shape[0] != target_frames:
            if mfcc.shape[0] == 0:
                return torch.zeros((target_frames, DataLoaderConfig.AUDIO_N_MFCC))

            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.transpose(1, 2).squeeze(0)  # (target_frames, n_mfcc)

        return mfcc
    except Exception:
        return torch.zeros((target_frames, DataLoaderConfig.AUDIO_N_MFCC))


def get_skeleton_features(mat_data):
    """Extracts 3D positions for the specific 12 upper-body joints."""
    try:
        if "Video" not in mat_data:
            return None
        video = mat_data["Video"]
        frames = video.Frames
        num_frames = getattr(video, "NumFrames", 0)

        if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
            frames = [frames]

        actual_frames = len(frames)
        if num_frames == 0:
            num_frames = actual_frames

        skeleton_data = np.zeros(
            (num_frames, DataLoaderConfig.NUM_JOINTS * 3), dtype=np.float32
        )

        for i in range(num_frames):
            if i >= len(frames):
                break
            frame = frames[i]

            # Extract Skeleton
            skel = frame.Skeleton
            if isinstance(skel, np.ndarray):
                if len(skel) > 0:
                    skel = skel[0]
                else:
                    continue

            # Extract Joints
            joints_xyz = []
            if hasattr(skel, "WorldPosition"):
                wp = skel.WorldPosition
                # Handle WorldPosition as array of structs or matrix
                if isinstance(wp, np.ndarray):
                    for j_idx in DataLoaderConfig.JOINTS_LIST:
                        if j_idx < len(wp):
                            j_pos = wp[j_idx]
                            if hasattr(j_pos, "X"):
                                joints_xyz.extend([j_pos.X, j_pos.Y, j_pos.Z])
                            else:
                                joints_xyz.extend([0.0, 0.0, 0.0])
                        else:
                            joints_xyz.extend([0.0, 0.0, 0.0])
                else:
                    joints_xyz = [0.0] * (DataLoaderConfig.NUM_JOINTS * 3)
            else:
                joints_xyz = [0.0] * (DataLoaderConfig.NUM_JOINTS * 3)

            skeleton_data[i] = joints_xyz

        return torch.tensor(skeleton_data, dtype=torch.float32)
    except Exception:
        return None


def get_labels(mat_data, num_frames):
    """Parses label annotations to create frame-wise target tensor."""
    labels = torch.zeros(num_frames, dtype=torch.long)  # Default 0 (Background)
    try:
        video = mat_data["Video"]
        if not hasattr(video, "Labels"):
            return labels

        raw_labels = video.Labels

        def process_lbl(obj):
            try:
                name = obj.Name
                start = int(obj.Begin) - 1  # Convert 1-based to 0-based
                end = int(obj.End)
                if name in DataLoaderConfig.GESTURE_MAP:
                    gid = DataLoaderConfig.GESTURE_MAP[name]
                    start = max(0, start)
                    end = min(num_frames, end)
                    if end > start:
                        labels[start:end] = gid
            except:
                pass

        if isinstance(raw_labels, np.ndarray):
            if raw_labels.ndim == 0:
                process_lbl(raw_labels.item())
            else:
                for l in raw_labels:
                    process_lbl(l)
        else:
            process_lbl(raw_labels)

        return labels
    except Exception:
        return labels


def process_sample(row):
    """Reads all files for a single sample and assembles the feature vector."""
    sample_id = row["sample_id"]
    data_path = os.path.join(DataLoaderConfig.INPUT_DIR, row["data_path"])
    audio_path = os.path.join(DataLoaderConfig.INPUT_DIR, row["audio_path"])

    mat = load_mat_file(data_path)
    if mat is None:
        return None

    # 1. Skeleton Features (Pos)
    skel_feats = get_skeleton_features(mat)  # (T, 36)
    if skel_feats is None or skel_feats.shape[0] == 0:
        return None

    T = skel_feats.shape[0]

    # 2. Audio Features
    audio_feats = process_audio(audio_path, T)  # (T, 13)

    # 3. Velocity Features (Derived from Pos)
    vel_feats = torch.zeros_like(skel_feats)
    vel_feats[1:] = skel_feats[1:] - skel_feats[:-1]

    # Concatenate: Pos(36) + Vel(36) + Audio(13) = 85
    features = torch.cat([skel_feats, vel_feats, audio_feats], dim=1)

    # 4. Labels
    labels = get_labels(mat, T)

    return {
        "sample_id": sample_id,
        "features": features.numpy(),  # Convert to numpy for storage
        "labels": labels.numpy(),
    }


def get_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split.
    Implements strict No-Pickle caching using np.savez.
    """
    os.makedirs(DataLoaderConfig.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(DataLoaderConfig.CACHE_DIR, f"{split}_data.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache ({cache_path})...")
        try:
            # allow_pickle=False ensures strict compliance
            data = np.load(cache_path, allow_pickle=False)
            features_flat = data["features_flat"]
            labels_flat = data["labels_flat"]
            limits = data["limits"]
            ids = data["ids"]

            dataset_list = []
            for i, sample_id in enumerate(ids):
                start, length = limits[i]
                feats = torch.from_numpy(features_flat[start : start + length])
                lbls = torch.from_numpy(labels_flat[start : start + length])
                dataset_list.append(
                    {"sample_id": str(sample_id), "features": feats, "labels": lbls}
                )
            return dataset_list
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {split} data from scratch...")
    csv_path = os.path.join(DataLoaderConfig.METADATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"Metadata file {csv_path} not found.")
        return []

    df = pd.read_csv(csv_path)
    # Drop rows with missing file paths to avoid TypeErrors during path construction
    df = df.dropna(subset=["data_path", "audio_path"])

    all_features = []
    all_labels = []
    limits = []
    ids = []

    current_idx = 0
    dataset_list = []

    for _, row in df.iterrows():
        sample = process_sample(row)
        if sample is not None:
            f = sample["features"]
            l = sample["labels"]
            length = f.shape[0]

            all_features.append(f)
            all_labels.append(l)
            limits.append([current_idx, length])
            ids.append(sample["sample_id"])

            current_idx += length

            # Keep in memory for return
            dataset_list.append(
                {
                    "sample_id": sample["sample_id"],
                    "features": torch.from_numpy(f),
                    "labels": torch.from_numpy(l),
                }
            )

    # 3. Save Cache
    if all_features:
        features_flat = np.concatenate(all_features, axis=0).astype(np.float32)
        labels_flat = np.concatenate(all_labels, axis=0).astype(np.int64)
        limits = np.array(limits, dtype=np.int32)
        ids = np.array(ids, dtype="U")  # Unicode string array

        np.savez(
            cache_path,
            features_flat=features_flat,
            labels_flat=labels_flat,
            limits=limits,
            ids=ids,
        )
        print(f"Saved {split} data to cache.")

    return dataset_list


class GestureDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item["features"].clone()  # (T, 85)
        labels = item["labels"].clone()

        if self.augment:
            # Physically Consistent Augmentation
            # Features structure: [Pos(36), Vel(36), Audio(13)]
            pos = features[:, :36]
            audio = features[:, 72:]

            # 1. Generate Gaussian Noise
            noise = torch.randn_like(pos) * DataLoaderConfig.NOISE_SIGMA

            # 2. Temporal Low-Pass Filter
            # Transpose to (1, C, T) for pooling
            noise_t = noise.transpose(0, 1).unsqueeze(0)
            noise_smooth = F.avg_pool1d(
                noise_t,
                kernel_size=DataLoaderConfig.NOISE_SMOOTH_KERNEL,
                stride=1,
                padding=DataLoaderConfig.NOISE_SMOOTH_KERNEL // 2,
            )
            # Remove dimensions
            noise_smooth = noise_smooth.squeeze(0).transpose(0, 1)  # (T, C)

            # 3. Add to Position
            pos_aug = pos + noise_smooth

            # 4. Derive Velocity from Augmented Position
            vel_aug = torch.zeros_like(pos_aug)
            vel_aug[1:] = pos_aug[1:] - pos_aug[:-1]

            # Reconstruct Feature Vector
            features = torch.cat([pos_aug, vel_aug, audio], dim=1)

        return features, labels, item["sample_id"]


def collate_fn(batch):
    """
    Pads sequences and generates masks.
    Returns:
        features_padded: (B, C, T) - Permuted for TCN
        labels_padded: (B, T)
        mask: (B, T)
        ids: List of sample IDs
    """
    features, labels, ids = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])

    # Pad sequences (B, T, C)
    features_padded = pad_sequence(features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Create Mask (1 for valid, 0 for pad)
    mask = torch.zeros(len(features), features_padded.shape[1])
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    # Transpose features to (B, C, T) for TCN/Conv input
    features_padded = features_padded.permute(0, 2, 1)

    return features_padded, labels_padded, mask, ids
