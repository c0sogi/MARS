import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import (
    DATA_PARAMS,
    AUDIO_PARAMS,
    TRAIN_PARAMS,
    CACHE_DIR,
    INPUT_DIR,
    METADATA_DIR,
    GESTURE_MAP,
    SEED,
)

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class GestureDataset(Dataset):
    def __init__(self, data, is_train=False):
        self.features = data["features"]
        self.labels = data["labels"]
        self.ids = data["ids"]
        self.is_train = is_train
        self.noise_std = TRAIN_PARAMS["noise_std"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (T, D)
        feat = self.features[idx]
        # Labels: (T,)
        lab = self.labels[idx]

        # Convert to tensor
        feat_tensor = torch.tensor(feat, dtype=torch.float32)
        lab_tensor = torch.tensor(lab, dtype=torch.long)

        # Augmentation: Add Gaussian noise to features during training
        if self.is_train and self.noise_std > 0:
            noise = torch.randn_like(feat_tensor) * self.noise_std
            feat_tensor += noise

        return feat_tensor, lab_tensor, self.ids[idx]


def load_mat_file(path):
    try:
        # Load matlab file with squeeze_me to simplify structure access
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception as e:
        # In case of corruption or read error
        return None


def extract_skeleton_features(mat_data, num_frames):
    """
    Extracts skeleton features from the loaded MAT structure.
    Returns: (T, J*6) matrix of Position + Velocity
    """
    if "Video" not in mat_data._fieldnames:
        return np.zeros((num_frames, len(DATA_PARAMS["selected_joints"]) * 6))

    video = mat_data.Video

    # Handle Frames structure
    if not hasattr(video, "Frames"):
        return np.zeros((num_frames, len(DATA_PARAMS["selected_joints"]) * 6))

    frames = video.Frames

    # Ensure frames is iterable (handle single frame case)
    if not isinstance(frames, np.ndarray):
        frames = [frames]

    # Use actual number of frames found in the struct
    actual_frames = len(frames)
    T = actual_frames
    J = len(DATA_PARAMS["selected_joints"])

    # Shape: (T, J, 3)
    skeleton_pos = np.zeros((T, J, 3))

    for t, frame in enumerate(frames):
        if not hasattr(frame, "Skeleton"):
            continue

        skel = frame.Skeleton

        # Handle multiple users: take the first valid skeleton found
        if isinstance(skel, np.ndarray):
            if skel.size > 0:
                skel = skel[0]
            else:
                continue

        # Extract WorldPosition
        if not hasattr(skel, "WorldPosition"):
            continue

        wp = skel.WorldPosition

        # Parse coordinates
        coords = np.zeros((20, 3))
        try:
            # Case 1: wp is a struct with X, Y, Z fields
            if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                coords[:, 0] = wp.X
                coords[:, 1] = wp.Y
                coords[:, 2] = wp.Z
            # Case 2: wp is a numpy array (20x3 or 3x20)
            elif isinstance(wp, np.ndarray):
                if wp.shape == (20, 3):
                    coords = wp
                elif wp.shape == (3, 20):
                    coords = wp.T
        except:
            pass

        # Select specific upper-body joints
        selected_indices = DATA_PARAMS["selected_joints"]
        selected_coords = coords[selected_indices]  # (J, 3)

        skeleton_pos[t] = selected_coords

    # Normalization relative to Reference Joint (HipCenter)
    if DATA_PARAMS["normalize_skeleton"]:
        try:
            # Find index of ref joint within the selected joints list
            ref_idx = DATA_PARAMS["selected_joints"].index(DATA_PARAMS["ref_joint"])
            ref_coords = skeleton_pos[:, ref_idx : ref_idx + 1, :]  # (T, 1, 3)
            skeleton_pos = skeleton_pos - ref_coords
        except ValueError:
            pass  # Ref joint not in selected list

    # Compute Velocity: V_t = P_t - P_{t-1}
    velocity = np.zeros_like(skeleton_pos)
    # Pad first frame with 0 velocity (already initialized)
    velocity[1:] = skeleton_pos[1:] - skeleton_pos[:-1]

    # Flatten features: (T, J*3)
    pos_flat = skeleton_pos.reshape(T, -1)
    vel_flat = velocity.reshape(T, -1)

    # Concatenate Position and Velocity: (T, J*6)
    features = np.concatenate([pos_flat, vel_flat], axis=1)

    return features


def extract_audio_features(audio_path, target_frames):
    """
    Extracts MFCC features and aligns them to the number of video frames.
    """
    full_path = os.path.join(INPUT_DIR, audio_path)
    if not os.path.exists(full_path):
        return np.zeros((target_frames, AUDIO_PARAMS["n_mfcc"]))

    try:
        waveform, sample_rate = torchaudio.load(full_path)

        # Resample if necessary
        if sample_rate != AUDIO_PARAMS["sample_rate"]:
            resampler = torchaudio.transforms.Resample(
                sample_rate, AUDIO_PARAMS["sample_rate"]
            )
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_PARAMS["sample_rate"],
            n_mfcc=AUDIO_PARAMS["n_mfcc"],
            melkwargs={
                "n_fft": AUDIO_PARAMS["n_fft"],
                "hop_length": AUDIO_PARAMS["hop_length"],
                "n_mels": 64,
            },
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, L_audio)

        # Interpolate to match target_frames (Video FPS alignment)
        # Input to interpolate needs to be (Batch, Channels, Length)
        # mfcc is already (1, n_mfcc, L_audio)

        mfcc_out = F.interpolate(
            mfcc, size=target_frames, mode="linear", align_corners=False
        )  # (1, n_mfcc, target_frames)

        features = mfcc_out.squeeze(0).transpose(0, 1)  # (target_frames, n_mfcc)
        return features.numpy()

    except Exception:
        # Return zeros on failure
        return np.zeros((target_frames, AUDIO_PARAMS["n_mfcc"]))


def process_labels(mat_data, num_frames):
    """
    Constructs dense frame-wise label vector from sparse annotations.
    """
    labels = np.zeros(num_frames, dtype=int)

    if "Video" not in mat_data._fieldnames:
        return labels

    video = mat_data.Video
    if not hasattr(video, "Labels"):
        return labels

    raw_labels = video.Labels

    # Helper to process single label object
    def add_label(obj):
        try:
            name = obj.Name
            start = int(obj.Begin)
            end = int(obj.End)
            if name in GESTURE_MAP:
                gid = GESTURE_MAP[name]
                # Matlab 1-based indexing -> Python 0-based
                s = max(0, start - 1)
                e = min(num_frames, end)
                if e > s:
                    labels[s:e] = gid
        except:
            pass

    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 0:
            add_label(raw_labels.item())
        else:
            for l in raw_labels:
                add_label(l)
    else:
        add_label(raw_labels)

    return labels


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    cache_path = os.path.join(CACHE_DIR, cache_name)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "features": data["features"],
                "labels": data["labels"],
                "ids": data["ids"],
            }
        except Exception:
            print("Cache corrupted, reprocessing...")

    # 2. Process Data from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_features = []
    all_labels = []
    all_ids = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = row["audio_path"]

        # Load MAT file
        mat = load_mat_file(data_path)
        if mat is None:
            continue

        # Determine number of frames
        try:
            num_frames = int(mat.Video.NumFrames)
        except:
            num_frames = row["num_frames"]

        # Extract Features
        skel_feat = extract_skeleton_features(mat, num_frames)
        real_frames = skel_feat.shape[0]

        audio_feat = extract_audio_features(audio_path, real_frames)

        # Concatenate Multi-modal Features
        combined_feat = np.concatenate([skel_feat, audio_feat], axis=1)

        # Extract Labels (Training/Validation only)
        # For test set, we create dummy labels of correct length
        if (
            "labels" in row
            and pd.notna(row["labels"])
            and str(row["labels"]).strip() != ""
        ):
            frame_labels = process_labels(mat, num_frames)
            # Align label length with feature length
            if len(frame_labels) > real_frames:
                frame_labels = frame_labels[:real_frames]
            elif len(frame_labels) < real_frames:
                pad = np.zeros(real_frames - len(frame_labels), dtype=int)
                frame_labels = np.concatenate([frame_labels, pad])
        else:
            frame_labels = np.zeros(real_frames, dtype=int)

        all_features.append(combined_feat)
        all_labels.append(frame_labels)
        all_ids.append(sample_id)

    # Convert to object arrays to handle variable sequence lengths
    features_arr = np.array(all_features, dtype=object)
    labels_arr = np.array(all_labels, dtype=object)
    ids_arr = np.array(all_ids)

    # 3. Save Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path, features=features_arr, labels=labels_arr, ids=ids_arr
    )

    return {"features": features_arr, "labels": labels_arr, "ids": ids_arr}


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Returns padded features, padded labels, lengths, and IDs.
    """
    # batch is a list of tuples (features, labels, id)
    features, labels, ids = zip(*batch)

    # Calculate lengths for masking
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)

    # Pad features: (Batch, Max_Len, Input_Dim)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Pad labels: (Batch, Max_Len)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_features, padded_labels, lengths, ids


def get_data_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Define metadata paths
    train_meta = os.path.join(METADATA_DIR, "train.csv")
    val_meta = os.path.join(METADATA_DIR, "val.csv")
    test_meta = os.path.join(METADATA_DIR, "test.csv")

    # Process or Load Data
    train_data = process_dataset(train_meta, "train_data.npz", load_cached_data)
    val_data = process_dataset(val_meta, "val_data.npz", load_cached_data)
    test_data = process_dataset(test_meta, "test_data.npz", load_cached_data)

    # Initialize Datasets
    train_dataset = GestureDataset(train_data, is_train=True)
    val_dataset = GestureDataset(val_data, is_train=False)
    test_dataset = GestureDataset(test_data, is_train=False)

    # Initialize DataLoaders
    # Pin memory enables faster data transfer to CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
