import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# Set fixed seeds
utils.set_seed(config.SEED)


def load_mat_file(path):
    """Safely load .mat file."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def compute_skeleton_features(mat_path):
    """
    Extracts raw Position from skeleton data.
    Derivatives will be computed on-the-fly to allow consistent augmentation.
    Returns: numpy array of shape (NumFrames, 60) -> Flattened (20*3)
    """
    mat = load_mat_file(mat_path)
    if mat is None or "Video" not in mat:
        return None

    video = mat["Video"]
    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames
    num_frames = len(frames)

    # 20 joints, 3 coords (X, Y, Z)
    pos_data = np.zeros(
        (num_frames, config.SKELETON_JOINTS, config.SKELETON_COORDS), dtype=np.float32
    )

    for i, frame in enumerate(frames):
        if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
            wp = frame.Skeleton.WorldPosition
            try:
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    pos_data[i, :, 0] = wp.X
                    pos_data[i, :, 1] = wp.Y
                    pos_data[i, :, 2] = wp.Z
                elif (
                    isinstance(wp, (np.ndarray, list))
                    and len(wp) == config.SKELETON_JOINTS
                ):
                    pos_data[i] = wp
            except:
                if i > 0:
                    pos_data[i] = pos_data[i - 1]
        else:
            if i > 0:
                pos_data[i] = pos_data[i - 1]

    # Convert mm to meters for stability
    pos_data = pos_data / 1000.0

    # Flatten to (T, 60) for storage
    return pos_data.reshape(num_frames, -1)


def compute_audio_features(wav_path, target_num_frames):
    """
    Extracts MFCC features and aligns them to video frames.
    Returns: numpy array of shape (NumFrames, N_MFCC)
    """
    if not os.path.exists(wav_path):
        return np.zeros((target_num_frames, config.INPUT_DIM_AUDIO), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(wav_path)

        # Resample if necessary (though config says 16000 is expected)
        if sample_rate != config.AUDIO_SR:
            resampler = T.Resample(sample_rate, config.AUDIO_SR)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=config.AUDIO_SR,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)
        # Shape: (Channels, n_mfcc, time) -> usually (1, 13, T_audio)

        if mfcc.dim() == 3:
            mfcc = mfcc.mean(dim=0)  # Average over channels if stereo

        # mfcc shape: (n_mfcc, time)

        # Interpolate to match video frame count
        # Input to interpolate needs to be (Batch, Channels, Time)
        # We treat n_mfcc as channels
        mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)

        mfcc_aligned = F.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Shape: (1, n_mfcc, target_frames) -> (target_frames, n_mfcc)
        mfcc_aligned = mfcc_aligned.squeeze(0).transpose(0, 1)

        return mfcc_aligned.numpy()

    except Exception as e:
        print(f"Error processing audio {wav_path}: {e}")
        return np.zeros((target_num_frames, config.INPUT_DIM_AUDIO), dtype=np.float32)


def process_sample(row):
    """
    Process a single sample row from metadata.
    Returns: ((pos_feats, audio_feats), labels)
    """
    sample_id = row["sample_id"]
    data_path = os.path.join(config.INPUT_DIR, row["data_path"])
    audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

    # 1. Skeleton Features (Raw Position: T, 60)
    pos_feats = compute_skeleton_features(data_path)
    if pos_feats is None:
        print(f"Warning: Failed to load skeleton for {sample_id}")
        return None, None

    num_frames = pos_feats.shape[0]

    # 2. Audio Features (T, 13)
    audio_feats = compute_audio_features(audio_path, num_frames)

    # 3. Labels
    labels = np.zeros(num_frames, dtype=np.int64)
    if "parsed_labels" in row and isinstance(row["parsed_labels"], list):
        for label_info in row["parsed_labels"]:
            start_frame = max(0, int(label_info["begin"]) - 1)
            end_frame = min(num_frames, int(label_info["end"]))
            label_id = int(label_info["id"])
            if start_frame < end_frame:
                labels[start_frame:end_frame] = label_id

    # Return components separately to allow on-the-fly augmentation
    return (pos_feats.astype(np.float32), audio_feats.astype(np.float32)), labels


def load_data_dict(metadata_path, cache_name, load_cached_data=True):
    """
    Loads data into a dictionary {sample_id: ((pos, audio), labels)}.
    Uses caching to speed up subsequent runs.
    Updated to use _v2 cache for separated features.
    """
    # Change cache name to avoid loading incompatible data
    cache_name = f"{cache_name}_v2"
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                ids = data["ids"]
                features = data["features"]  # Array of tuples (pos, audio)
                labels = data["labels"]

                data_dict = {}
                for i, sample_id in enumerate(ids):
                    data_dict[sample_id] = (features[i], labels[i])
                return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if "labels" in df.columns:
        df["parsed_labels"] = df["labels"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    data_dict = {}
    ids_list = []
    features_list = []
    labels_list = []

    for idx, row in df.iterrows():
        result = process_sample(row)
        if result[0] is not None:
            # result is ((pos, audio), labels)
            feats, labs = result
            data_dict[row["sample_id"]] = (feats, labs)

            ids_list.append(row["sample_id"])
            features_list.append(feats)
            labels_list.append(labs)

    np.savez_compressed(
        cache_path,
        ids=np.array(ids_list),
        features=np.array(features_list, dtype=object),
        labels=np.array(labels_list, dtype=object),
    )
    print(f"Saved cache to {cache_path}")

    return data_dict


class GestureDataset(Dataset):
    def __init__(
        self,
        data_dict,
        mode="train",
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        augment=False,
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.data = []

        sorted_ids = sorted(data_dict.keys())

        if self.mode == "train":
            for sample_id in sorted_ids:
                (pos, audio), labs = data_dict[sample_id]
                num_frames = pos.shape[0]

                if num_frames < self.window_size:
                    self.data.append(
                        {
                            "sample_id": sample_id,
                            "pos": pos,
                            "audio": audio,
                            "labels": labs,
                            "start_idx": 0,
                            "is_short": True,
                        }
                    )
                else:
                    for start_idx in range(
                        0, num_frames - self.window_size + 1, self.stride
                    ):
                        self.data.append(
                            {
                                "sample_id": sample_id,
                                "pos": pos,
                                "audio": audio,
                                "labels": labs,
                                "start_idx": start_idx,
                                "is_short": False,
                            }
                        )
        else:
            for sample_id in sorted_ids:
                (pos, audio), labs = data_dict[sample_id]
                self.data.append(
                    {"sample_id": sample_id, "pos": pos, "audio": audio, "labels": labs}
                )

    def _augment_position(self, pos_data):
        # pos_data: (T, 60) -> (T, 20, 3)
        T_dim = pos_data.shape[0]
        pos_3d = pos_data.reshape(T_dim, 20, 3)

        # Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        pos_3d = pos_3d * scale

        # Random Rotation around Y-axis (Gravity)
        angle_deg = np.random.uniform(-15, 15)
        rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # Ry matrix
        # [cos  0  sin]
        # [0    1  0  ]
        # [-sin 0  cos]
        # Apply to each point
        x = pos_3d[:, :, 0]
        z = pos_3d[:, :, 2]
        new_x = x * cos_a + z * sin_a
        new_z = -x * sin_a + z * cos_a
        pos_3d[:, :, 0] = new_x
        pos_3d[:, :, 2] = new_z

        return pos_3d.reshape(T_dim, 60)

    def _compute_derivatives_and_concat(self, pos, audio):
        # pos: (T, 60)
        # audio: (T, 13)

        # Reshape to (T, 20, 3) for derivative calc
        T_dim = pos.shape[0]
        p = pos.reshape(T_dim, 20, 3)

        # Velocity
        v = np.zeros_like(p)
        v[1:] = p[1:] - p[:-1]

        # Acceleration
        a = np.zeros_like(v)
        a[1:] = v[1:] - v[:-1]

        # Flatten and Concat
        # (T, 20, 3) -> (T, 60)
        p_flat = p.reshape(T_dim, -1)
        v_flat = v.reshape(T_dim, -1)
        a_flat = a.reshape(T_dim, -1)

        # (T, 180 + 13)
        return np.concatenate([p_flat, v_flat, a_flat, audio], axis=1).astype(
            np.float32
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == "train":
            pos = item["pos"]
            audio = item["audio"]
            labs = item["labels"]

            # Slice
            if item["is_short"]:
                pos_window = pos
                audio_window = audio
                lab_window = labs
                # Padding handled after feature computation
            else:
                start = item["start_idx"]
                end = start + self.window_size
                pos_window = pos[start:end]
                audio_window = audio[start:end]
                lab_window = labs[start:end]

            # Augment Position (Kinematically Consistent)
            # Cite solution_lesson_node_00032
            if self.augment and torch.rand(1).item() < config.AUGMENT_PROB:
                pos_window = self._augment_position(pos_window)

            # Compute Features on-the-fly
            # Cite solution_lesson_node_00029 (Consistency)
            features = self._compute_derivatives_and_concat(pos_window, audio_window)

            # Pad if short
            if item["is_short"]:
                pad_len = self.window_size - features.shape[0]
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                lab_window = np.pad(
                    lab_window, (0, pad_len), mode="constant", constant_values=0
                )

            return torch.from_numpy(features), torch.from_numpy(lab_window)

        else:
            # Full sequence
            pos = item["pos"]
            audio = item["audio"]
            labs = item["labels"]

            # No augmentation for val/test
            features = self._compute_derivatives_and_concat(pos, audio)

            return (
                torch.from_numpy(features),
                torch.from_numpy(labs),
                item["sample_id"],
            )


def collate_fn_padd(batch):
    """
    Collate function to handle variable length sequences in validation/test.
    Actually, for TCN/GRU we can just return the list or pad to max in batch.
    However, for simplicity in evaluation loop, we often use batch_size=1.
    If batch_size > 1, we need padding.
    """
    # Check if this is a train batch (tensors) or val batch (tuples with sample_id)
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        # Val/Test mode
        # Sort by length for efficiency (optional)
        # batch.sort(key=lambda x: x[0].shape[0], reverse=True)

        features = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        ids = [x[2] for x in batch]

        # Pad features and labels
        features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0
        )

        # Create mask
        lengths = torch.tensor([x.shape[0] for x in features])

        return features_padded, labels_padded, ids, lengths
    else:
        # Train mode (fixed window size)
        # Default collate is fine
        return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Load raw data dictionaries
    train_dict = load_data_dict(
        config.TRAIN_METADATA_PATH, "train_features", load_cached_data
    )
    val_dict = load_data_dict(
        config.VAL_METADATA_PATH, "val_features", load_cached_data
    )
    test_dict = load_data_dict(
        config.TEST_METADATA_PATH, "test_features", load_cached_data
    )

    # Create Datasets
    # Enable augmentation for training
    train_ds = GestureDataset(train_dict, mode="train", augment=True)
    val_ds = GestureDataset(val_dict, mode="val", augment=False)
    test_ds = GestureDataset(
        test_dict, mode="test", augment=False
    )  # 'test' mode behaves like 'val' (full seq)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val/Test loaders use batch_size=1 to handle variable lengths easily without complex masking in model
    # Or we can use the collate_fn_padd. Let's use batch_size=1 for safety and simplicity in inference loop.
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader
