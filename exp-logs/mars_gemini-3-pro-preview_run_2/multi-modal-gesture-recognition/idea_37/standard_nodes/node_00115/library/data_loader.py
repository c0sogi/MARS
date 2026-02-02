import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import library.config as config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(
        self, metadata_file, mode="train", augment=False, load_cached_data=True
    ):
        """
        Args:
            metadata_file (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply augmentation.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.mode = mode
        self.augment = augment
        self.metadata = pd.read_csv(metadata_file)

        # Parse labels column from string to list of ints for reference (though we use MAT for frame-level)
        if "labels" in self.metadata.columns:
            self.metadata["labels"] = self.metadata["labels"].apply(
                lambda x: (
                    [int(i) for i in str(x).split()]
                    if pd.notna(x) and str(x).strip() != ""
                    else []
                )
            )

        self.cache_dir = os.path.join(config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{mode}_data.npz")

        # Load or Process Data
        self.data = self._load_and_cache_data(load_cached_data)

    def _load_and_cache_data(self, load_cached):
        if load_cached and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}...")
            try:
                loaded = np.load(self.cache_file)
                packed_features = loaded["packed_features"]
                packed_labels = loaded["packed_labels"]
                packed_boundaries = loaded["packed_boundaries"]
                seq_lengths = loaded["seq_lengths"]
                sample_ids = loaded["sample_ids"]

                return self._reconstruct_data(
                    packed_features,
                    packed_labels,
                    packed_boundaries,
                    seq_lengths,
                    sample_ids,
                )
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Processing {self.mode} data from scratch...")

        all_features = []
        all_labels = []
        all_boundaries = []
        seq_lengths = []
        sample_ids = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            try:
                mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
                audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

                # --- Process Skeleton ---
                try:
                    mat = scipy.io.loadmat(
                        mat_path, squeeze_me=True, struct_as_record=False
                    )
                except Exception:
                    mat = scipy.io.loadmat(
                        mat_path, squeeze_me=True, struct_as_record=True
                    )

                video = mat["Video"]
                num_frames = int(video.NumFrames)
                frames_data = video.Frames

                # Extract Joints (T, 12, 3)
                raw_positions = np.zeros(
                    (num_frames, config.NUM_JOINTS, 3), dtype=np.float32
                )

                if (
                    isinstance(frames_data, np.ndarray)
                    and len(frames_data) == num_frames
                ):
                    for f_idx, frame in enumerate(frames_data):
                        # Extract Skeleton
                        skel = None
                        if hasattr(frame, "Skeleton"):
                            skel = frame.Skeleton
                        elif isinstance(frame, dict) and "Skeleton" in frame:
                            skel = frame["Skeleton"]

                        if skel is None:
                            continue

                        # Handle array of skeletons (multiple users) -> take first
                        if isinstance(skel, np.ndarray):
                            if skel.size > 0:
                                skel = skel[0]
                            else:
                                continue

                        # Extract WorldPosition
                        if hasattr(skel, "WorldPosition"):
                            wp = skel.WorldPosition
                            if isinstance(wp, np.ndarray) and len(wp) >= 20:
                                for j_k, j_idx in enumerate(config.UPPER_BODY_INDICES):
                                    joint = wp[j_idx]
                                    if hasattr(joint, "X"):
                                        raw_positions[f_idx, j_k] = [
                                            joint.X,
                                            joint.Y,
                                            joint.Z,
                                        ]
                                    elif (
                                        isinstance(joint, np.ndarray)
                                        and len(joint) == 3
                                    ):
                                        raw_positions[f_idx, j_k] = joint

                # Normalize Skeleton
                # 1. Center relative to HipCenter (Index 0 in UPPER_BODY_INDICES)
                hip_center = raw_positions[:, 0:1, :].copy()
                centered_positions = raw_positions - hip_center

                # 2. Scale to meters
                centered_positions = centered_positions * config.SKELETON_SCALE

                # 3. Compute Velocity
                velocity = np.zeros_like(centered_positions)
                velocity[1:] = centered_positions[1:] - centered_positions[:-1]

                # Flatten: (T, 12, 3) -> (T, 36)
                pos_flat = centered_positions.reshape(num_frames, -1)
                vel_flat = velocity.reshape(num_frames, -1)
                skeleton_features = np.concatenate(
                    [pos_flat, vel_flat], axis=1
                )  # (T, 72)

                # --- Process Audio ---
                waveform, sample_rate = torchaudio.load(audio_path)

                # Compute MFCC
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=config.N_MFCC,
                    melkwargs={
                        "n_fft": 400,
                        "hop_length": 160,
                        "n_mels": 23,
                        "center": False,
                    },
                )
                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

                # Interpolate to match video frames
                # mfcc is (channels, n_mfcc, time). Usually (1, 13, T).
                # F.interpolate linear mode expects (Batch, Channels, Length).
                # Ensure 3D input: (Batch, Channels, Length)
                if mfcc.dim() == 2:
                    mfcc = mfcc.unsqueeze(0)

                # Resize to num_frames
                mfcc_interpolated = F.interpolate(
                    mfcc, size=num_frames, mode="linear", align_corners=False
                )
                mfcc_features = (
                    mfcc_interpolated.squeeze(0).permute(1, 0).numpy()
                )  # (T, n_mfcc)

                # --- Combine ---
                # Ensure length match
                min_len = min(len(skeleton_features), len(mfcc_features))
                final_features = np.concatenate(
                    [skeleton_features[:min_len], mfcc_features[:min_len]], axis=1
                )

                # If audio was shorter/longer or skeleton mismatch, pad/trim to num_frames
                if len(final_features) < num_frames:
                    pad_len = num_frames - len(final_features)
                    final_features = np.pad(final_features, ((0, pad_len), (0, 0)))
                elif len(final_features) > num_frames:
                    final_features = final_features[:num_frames]

                # --- Process Labels ---
                labels = np.zeros(num_frames, dtype=np.int64)
                boundaries = np.zeros(num_frames, dtype=np.float32)

                if self.mode != "test" and hasattr(video, "Labels"):
                    raw_labels = video.Labels
                    if not isinstance(raw_labels, np.ndarray):
                        raw_labels = [raw_labels]
                    elif raw_labels.ndim == 0:
                        raw_labels = [raw_labels.item()]

                    for l in raw_labels:
                        try:
                            name = l.Name
                            start = int(l.Begin) - 1
                            end = int(l.End) - 1
                            if name in config.GESTURE_MAP:
                                gid = config.GESTURE_MAP[name]
                                start = max(0, start)
                                end = min(num_frames - 1, end)
                                labels[start : end + 1] = gid
                                boundaries[start] = 1.0
                                boundaries[end] = 1.0
                        except AttributeError:
                            pass

                all_features.append(final_features.astype(np.float32))
                all_labels.append(labels)
                all_boundaries.append(boundaries)
                seq_lengths.append(num_frames)
                sample_ids.append(sample_id)

            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue

        # Pack
        packed_features = np.concatenate(all_features, axis=0)
        packed_labels = np.concatenate(all_labels, axis=0)
        packed_boundaries = np.concatenate(all_boundaries, axis=0)
        seq_lengths = np.array(seq_lengths, dtype=np.int32)
        sample_ids = np.array(sample_ids, dtype=str)

        np.savez(
            self.cache_file,
            packed_features=packed_features,
            packed_labels=packed_labels,
            packed_boundaries=packed_boundaries,
            seq_lengths=seq_lengths,
            sample_ids=sample_ids,
        )

        return self._reconstruct_data(
            packed_features, packed_labels, packed_boundaries, seq_lengths, sample_ids
        )

    def _reconstruct_data(self, features, labels, boundaries, lengths, ids):
        data = []
        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            data.append(
                {
                    "sample_id": str(ids[i]),
                    "features": features[start:end],
                    "labels": labels[start:end],
                    "boundaries": boundaries[start:end],
                }
            )
            start = end
        return data

    def _augment_sample(self, features):
        """
        Apply physically consistent augmentation.
        features: (T, 85) -> [pos(36), vel(36), mfcc(13)]
        """
        T = features.shape[0]
        skeleton = features[:, :72]
        mfcc = features[:, 72:]

        pos = skeleton[:, :36].reshape(T, 12, 3)

        # Generate Noise
        noise = np.random.normal(0, 0.005, size=(T, 12, 3))

        # Temporal Low-Pass Filter
        b, a = scipy.signal.butter(1, 0.1)
        noise_smooth = scipy.signal.lfilter(b, a, noise, axis=0)

        # Add to positions
        pos_aug = pos + noise_smooth

        # Recompute Velocity
        vel_aug = np.zeros_like(pos_aug)
        vel_aug[1:] = pos_aug[1:] - pos_aug[:-1]

        # Flatten and Reassemble
        pos_aug_flat = pos_aug.reshape(T, 36)
        vel_aug_flat = vel_aug.reshape(T, 36)
        skeleton_aug = np.concatenate([pos_aug_flat, vel_aug_flat], axis=1)

        return np.concatenate([skeleton_aug, mfcc], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = sample["features"]
        labels = sample["labels"]
        boundaries = sample["boundaries"]

        if self.augment:
            features = self._augment_sample(features)

        return {
            "features": torch.from_numpy(features).float(),
            "labels": torch.from_numpy(labels).long(),
            "boundaries": torch.from_numpy(boundaries).float(),
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    boundaries = [x["boundaries"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([x.shape[0] for x in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    boundaries_padded = pad_sequence(boundaries, batch_first=True, padding_value=0.0)

    mask = torch.zeros(features_padded.shape[0], features_padded.shape[1])
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    return {
        "features": features_padded,
        "labels": labels_padded,
        "boundaries": boundaries_padded,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_loaders(batch_size=None, num_workers=None):
    if batch_size is None:
        batch_size = config.HYPERPARAMS["batch_size"]
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    train_ds = GestureDataset(
        os.path.join(config.METADATA_DIR, "train.csv"), mode="train", augment=True
    )
    val_ds = GestureDataset(
        os.path.join(config.METADATA_DIR, "val.csv"), mode="val", augment=False
    )
    test_ds = GestureDataset(
        os.path.join(config.METADATA_DIR, "test.csv"), mode="test", augment=False
    )

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
