import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import gaussian_filter1d
from library.config import Config
from library.utils import set_seed


def load_skeleton_from_mat(mat_path, num_frames_expected):
    """
    Parses the MAT file to extract skeleton joint positions.
    Returns: (T, 12, 3) tensor of joint positions.
    """
    try:
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if "Video" not in mat:
            return np.zeros(
                (num_frames_expected, Config.NUM_JOINTS, 3), dtype=np.float32
            )

        video = mat["Video"]
        frames = video.Frames

        # Handle cases where Frames is a single object or empty
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames]) if frames is not None else np.array([])

        actual_frames = min(len(frames), num_frames_expected)
        skeleton_data = np.zeros((num_frames_expected, 20, 3), dtype=np.float32)

        for i in range(actual_frames):
            frame = frames[i]
            # Defensive check for Skeletons existence
            if not hasattr(frame, "Skeletons"):
                continue

            skels = frame.Skeletons
            # Handle empty or array skeletons
            target_skel = None
            if isinstance(skels, np.ndarray):
                if skels.size > 0:
                    target_skel = skels[0] if skels.ndim > 0 else skels.item()
            elif skels is not None:
                # Single struct
                target_skel = skels

            if target_skel is not None and hasattr(target_skel, "WorldPosition"):
                wp = target_skel.WorldPosition
                # WorldPosition might be an array of structs or a struct of arrays
                # Based on description: X, Y, Z values.
                # Assuming standard format: array of 20 joints, each with X,Y,Z
                if isinstance(wp, np.ndarray) and len(wp) >= 20:
                    for j in range(20):
                        # Check if element is struct or array
                        joint = wp[j]
                        if hasattr(joint, "X"):
                            skeleton_data[i, j, 0] = joint.X
                            skeleton_data[i, j, 1] = joint.Y
                            skeleton_data[i, j, 2] = joint.Z

        # Select specific joints
        selected_skeleton = skeleton_data[:, Config.SELECTED_JOINTS, :]
        return selected_skeleton

    except Exception as e:
        # Return zeros on failure to allow pipeline to continue (masked out later if needed)
        return np.zeros((num_frames_expected, Config.NUM_JOINTS, 3), dtype=np.float32)


def get_dense_labels(mat_path, num_frames):
    """
    Parses MAT file to generate frame-wise labels.
    Returns: (T,) array of integers.
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    try:
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if "Video" not in mat:
            return labels

        video = mat["Video"]
        if not hasattr(video, "Labels"):
            return labels

        raw_labels = video.Labels
        if not isinstance(raw_labels, np.ndarray):
            raw_labels = (
                np.array([raw_labels]) if raw_labels is not None else np.array([])
            )

        if raw_labels.ndim == 0 and raw_labels.size > 0:
            raw_labels = np.array([raw_labels.item()])

        for lbl in raw_labels:
            try:
                if not hasattr(lbl, "Name"):
                    continue
                name = lbl.Name
                if name in Config.GESTURE_MAP:
                    gid = Config.GESTURE_MAP[name]
                    # Matlab 1-based indexing
                    start = int(lbl.Begin) - 1
                    end = int(lbl.End)

                    # Clip to valid range
                    start = max(0, start)
                    end = min(num_frames, end)

                    if end > start:
                        labels[start:end] = gid
            except:
                continue
    except:
        pass
    return labels


def load_audio_features(audio_path, target_frames):
    """
    Loads audio, computes MFCC, and interpolates to match target_frames.
    Returns: (T, 13) tensor.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "n_mels": 64,
                "hop_length": Config.HOP_LENGTH,
                "mel_scale": "htk",
            },
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Interpolate to match video frames
        if mfcc.shape[0] != target_frames:
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
            mfcc = torch.nn.functional.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.transpose(1, 2).squeeze(0)  # (target_frames, n_mfcc)

        return mfcc.numpy()

    except Exception:
        return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)


def process_dataset(metadata_path, split_name, cache_dir):
    """
    Iterates metadata, loads raw data, and saves to cache.
    """
    df = pd.read_csv(metadata_path)

    all_skeletons = {}
    all_audios = {}
    all_labels = {}

    print(f"Processing {split_name} data...")

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        num_frames = int(row["num_frames"])

        # Construct full paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Load Data
        skel = load_skeleton_from_mat(mat_path, num_frames)
        audio = load_audio_features(audio_path, num_frames)

        if split_name == "test":
            lbls = np.zeros(num_frames, dtype=np.int64)
        else:
            lbls = get_dense_labels(mat_path, num_frames)

        all_skeletons[sample_id] = skel
        all_audios[sample_id] = audio
        all_labels[sample_id] = lbls

    # Save to NPZ
    save_path = os.path.join(cache_dir, f"{split_name}_data.npz")
    np.savez_compressed(
        save_path,
        skeletons=all_skeletons,
        audios=all_audios,
        labels=all_labels,
        sample_ids=df["sample_id"].values,
    )
    print(f"Saved {split_name} cache to {save_path}")
    return all_skeletons, all_audios, all_labels, df["sample_id"].values


def load_cached_dataset(split_name, cache_dir):
    path = os.path.join(cache_dir, f"{split_name}_data.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return (
            data["skeletons"].item(),
            data["audios"].item(),
            data["labels"].item(),
            data["sample_ids"],
        )
    except:
        return None


class GestureDataset(Dataset):
    def __init__(self, skeletons, audios, labels, sample_ids, is_train=True):
        self.skeletons = skeletons
        self.audios = audios
        self.labels = labels
        self.sample_ids = sample_ids
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        # Get raw data
        skel = self.skeletons[sid].copy()  # (T, 12, 3)
        audio = self.audios[sid].copy()  # (T, 13)
        lbl = self.labels[sid].copy()  # (T,)

        T_len = skel.shape[0]

        # 1. Normalization (Meters and Centering)
        # Convert mm to meters
        skel = skel / 1000.0

        # Center around HipCenter (Joint 0) for each frame
        # skel shape: (T, 12, 3). Joint 0 is at index 0.
        root = skel[:, 0:1, :]  # (T, 1, 3)
        skel = skel - root

        # 2. Augmentation (Physically Consistent Smooth Noise)
        if self.is_train:
            # Generate noise
            sigma = Config.AUG_SIGMA
            noise = np.random.normal(0, sigma, skel.shape).astype(np.float32)

            # Temporal Smoothing
            # Apply along time axis (0)
            noise = gaussian_filter1d(noise, sigma=Config.AUG_SMOOTH_KERNEL, axis=0)

            # Add to position
            skel = skel + noise

        # 3. Compute Velocity
        # V_t = P_t - P_{t-1}. Pad first frame with 0.
        vel = np.zeros_like(skel)
        vel[1:] = skel[1:] - skel[:-1]

        # 4. Flatten Skeleton and Velocity
        # (T, 12, 3) -> (T, 36)
        skel_flat = skel.reshape(T_len, -1)
        vel_flat = vel.reshape(T_len, -1)

        # 5. Concatenate Features
        # [Skeleton, Velocity, Audio] -> (T, 36+36+13) = (T, 85)
        features = np.concatenate([skel_flat, vel_flat, audio], axis=1).astype(
            np.float32
        )

        # 6. Generate Boundary Labels
        # 1 if label changes from previous, else 0
        bnd_lbl = np.zeros(T_len, dtype=np.float32)
        if T_len > 1:
            # Compare t with t-1
            diff = lbl[1:] != lbl[:-1]
            bnd_lbl[1:] = diff.astype(np.float32)

        return {
            "features": torch.from_numpy(features),
            "cls_targets": torch.from_numpy(lbl).long(),
            "bnd_targets": torch.from_numpy(bnd_lbl).float(),
            "sample_id": sid,
        }


def collate_fn(batch):
    # Sort by length for potential packing (optional, but good practice)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    cls_targets = [x["cls_targets"] for x in batch]
    bnd_targets = [x["bnd_targets"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    cls_targets_padded = pad_sequence(
        cls_targets, batch_first=True, padding_value=0
    )  # 0 is background
    bnd_targets_padded = pad_sequence(bnd_targets, batch_first=True, padding_value=0.0)

    # Create Mask
    # Lengths of each sequence
    lengths = torch.tensor([len(x) for x in features])
    max_len = features_padded.size(1)

    # (B, T) mask
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.float()  # or bool

    return {
        "features": features_padded,
        "mask": mask,
        "cls_targets": cls_targets_padded,
        "bnd_targets": bnd_targets_padded,
        "sample_ids": ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get data loaders.
    """
    set_seed(Config.SEED)

    splits = [
        ("train", Config.TRAIN_METADATA_PATH, True),
        ("val", Config.VAL_METADATA_PATH, False),
        ("test", Config.TEST_METADATA_PATH, False),
    ]

    loaders = {}

    for split_name, meta_path, is_train in splits:
        # Try load cache
        data = None
        if load_cached_data:
            data = load_cached_dataset(split_name, Config.CACHE_DIR)

        if data is None:
            data = process_dataset(meta_path, split_name, Config.CACHE_DIR)

        skeletons, audios, labels, sample_ids = data

        dataset = GestureDataset(
            skeletons, audios, labels, sample_ids, is_train=is_train
        )

        shuffle = is_train
        bs = (
            Config.BATCH_SIZE if is_train else 1
        )  # Test/Val often better with 1 for eval simplicity or same batch
        if split_name == "test":
            bs = 1

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
