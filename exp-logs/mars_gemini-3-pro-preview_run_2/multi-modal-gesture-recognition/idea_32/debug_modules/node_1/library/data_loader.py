import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.io
import torchaudio
from torch.utils.data import Dataset, DataLoader
import library.config as config
import library.utils as utils

# Ensure deterministic behavior
utils.set_seed(config.SEED)


def _parse_mat_file(mat_path):
    """
    Parses the .mat file to extract skeleton data and labels.
    Returns:
        skeleton_data: (NumFrames, NumJoints, 3)
        cls_labels: (NumFrames,)
        bnd_labels: (NumFrames,)
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, None, None

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames_struct = getattr(video, "Frames", None)
        labels_raw = getattr(video, "Labels", [])

        # Initialize Skeleton Data: (NumFrames, NumSelectedJoints, 3)
        skeleton_data = np.zeros(
            (num_frames, len(config.SELECTED_JOINTS), 3), dtype=np.float32
        )

        # Extract Skeleton
        if (
            frames_struct is not None
            and isinstance(frames_struct, np.ndarray)
            and len(frames_struct) == num_frames
        ):
            for f_idx, frame_obj in enumerate(frames_struct):
                try:
                    skel_obj = frame_obj.Skeleton
                    # Handle multiple users (take first) or single user
                    target_skel = None
                    if isinstance(skel_obj, np.ndarray) and len(skel_obj) > 0:
                        target_skel = skel_obj[0]
                    elif hasattr(skel_obj, "WorldPosition"):
                        target_skel = skel_obj

                    if target_skel is not None:
                        # Assume WorldPosition is available and indexable
                        # If WorldPosition is (NumJoints, 3)
                        wp = target_skel.WorldPosition
                        if hasattr(wp, "shape") and wp.shape[0] >= 20:
                            for i, j_idx in enumerate(config.SELECTED_JOINTS):
                                skeleton_data[f_idx, i, :] = wp[j_idx, :]
                except AttributeError:
                    pass

        # Process Labels
        cls_labels = np.zeros(num_frames, dtype=np.int64)
        bnd_labels = np.zeros(num_frames, dtype=np.float32)

        def process_label_obj(obj):
            try:
                name = obj.Name
                start = int(obj.Begin) - 1  # Convert 1-based to 0-based
                end = int(obj.End) - 1

                if name in config.GESTURE_MAP:
                    gid = config.GESTURE_MAP[name]
                    start = max(0, start)
                    end = min(num_frames - 1, end)

                    if start <= end:
                        cls_labels[start : end + 1] = gid
                        # Sharp boundary targets
                        bnd_labels[start] = 1.0
                        bnd_labels[end] = 1.0
            except AttributeError:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                process_label_obj(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label_obj(l)
        else:
            process_label_obj(labels_raw)

        return skeleton_data, cls_labels, bnd_labels

    except Exception:
        return None, None, None


def _process_audio(audio_path, num_frames):
    """
    Loads audio, extracts MFCCs, and aligns to video frames via interpolation.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample
        if sample_rate != config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # Align to video frames
        if num_frames > 0:
            # Interpolate expects (Batch, Channels, Length)
            mfcc = F.interpolate(
                mfcc, size=num_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).permute(1, 0)  # (num_frames, n_mfcc)
            return mfcc.numpy()
        else:
            return np.zeros((0, config.N_MFCC), dtype=np.float32)

    except Exception:
        # Return zeros on failure
        return np.zeros((num_frames, config.N_MFCC), dtype=np.float32)


def _preprocess_batch(metadata_df, load_cached_data=True, mode="train"):
    """
    Loads raw data, preprocesses, and caches it to disk.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return (
                data["positions"],
                data["audio"],
                data["cls_labels"],
                data["bnd_labels"],
                data["offsets"],
                data["sample_ids"],
            )
        except Exception:
            pass

    all_positions = []
    all_audio = []
    all_cls = []
    all_bnd = []
    offsets = []
    sample_ids = []

    current_offset = 0

    for _, row in metadata_df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        skel, cls, bnd = _parse_mat_file(data_path)

        # Handle invalid/missing data by creating zeros if num_frames known
        n_frames = row["num_frames"]
        if skel is None:
            if n_frames == 0:
                continue
            skel = np.zeros(
                (n_frames, len(config.SELECTED_JOINTS), 3), dtype=np.float32
            )
            cls = np.zeros(n_frames, dtype=np.int64)
            bnd = np.zeros(n_frames, dtype=np.float32)

        # Ensure consistency
        n_frames = min(skel.shape[0], n_frames) if n_frames > 0 else skel.shape[0]
        skel = skel[:n_frames]
        cls = cls[:n_frames]
        bnd = bnd[:n_frames]

        # Process Audio
        aud = _process_audio(audio_path, n_frames)

        # Normalize Skeleton: Center to HipCenter (Index 0) and Scale
        hip_center = skel[:, 0:1, :]
        skel_norm = (skel - hip_center) * config.SKELETON_SCALE

        all_positions.append(skel_norm)
        all_audio.append(aud)
        all_cls.append(cls)
        all_bnd.append(bnd)
        sample_ids.append(sample_id)

        offsets.append([current_offset, current_offset + n_frames])
        current_offset += n_frames

    if not all_positions:
        raise RuntimeError(f"No valid data found for mode {mode}")

    cat_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
    cat_audio = np.concatenate(all_audio, axis=0).astype(np.float32)
    cat_cls = np.concatenate(all_cls, axis=0).astype(np.int64)
    cat_bnd = np.concatenate(all_bnd, axis=0).astype(np.float32)
    arr_offsets = np.array(offsets, dtype=np.int64)
    arr_ids = np.array(sample_ids)

    np.savez_compressed(
        cache_path,
        positions=cat_positions,
        audio=cat_audio,
        cls_labels=cat_cls,
        bnd_labels=cat_bnd,
        offsets=arr_offsets,
        sample_ids=arr_ids,
    )

    return cat_positions, cat_audio, cat_cls, cat_bnd, arr_offsets, arr_ids


class GestureDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True, limit=None):
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)

        if limit:
            self.metadata = self.metadata.head(limit)

        (
            self.positions,
            self.audio,
            self.cls_labels,
            self.bnd_labels,
            self.offsets,
            self.sample_ids,
        ) = _preprocess_batch(self.metadata, load_cached_data, mode)

        # Map sample_id to index to handle subsetting if metadata was limited
        self.id_to_idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        self.valid_indices = []
        for sid in self.metadata["sample_id"]:
            if sid in self.id_to_idx:
                self.valid_indices.append(self.id_to_idx[sid])

    def __len__(self):
        return len(self.valid_indices)

    def _augment_skeleton(self, positions):
        """
        Applies temporally correlated noise to positions.
        """
        T, J, C = positions.shape
        # Gaussian Noise
        noise = np.random.normal(0, 0.01, (T, J, C)).astype(np.float32)

        # Temporal Smoothing (Moving Average)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size

        noise_filtered = np.zeros_like(noise)
        for j in range(J):
            for c in range(C):
                noise_filtered[:, j, c] = np.convolve(
                    noise[:, j, c], kernel, mode="same"
                )

        return positions + noise_filtered

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        start, end = self.offsets[real_idx]

        pos = self.positions[start:end].copy()
        aud = self.audio[start:end].copy()
        cls = self.cls_labels[start:end].copy()
        bnd = self.bnd_labels[start:end].copy()

        # Augmentation
        if self.mode == "train":
            pos = self._augment_skeleton(pos)

        # Compute Velocity: Vel[t] = Pos[t] - Pos[t-1]
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # Flatten
        T = pos.shape[0]
        pos_flat = pos.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)

        # Fuse
        features = np.concatenate([pos_flat, vel_flat, aud], axis=1)

        return {
            "features": torch.from_numpy(features).float(),
            "cls_labels": torch.from_numpy(cls).long(),
            "bnd_labels": torch.from_numpy(bnd).float(),
            "sample_id": self.sample_ids[real_idx],
        }


def collate_fn(batch):
    # Sort by length descending
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    cls_labels = [x["cls_labels"] for x in batch]
    bnd_labels = [x["bnd_labels"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([f.shape[0] for f in features])
    max_len = lengths.max().item()

    padded_features = torch.zeros(len(features), max_len, features[0].shape[1])
    padded_cls = torch.zeros(len(cls_labels), max_len, dtype=torch.long)
    padded_bnd = torch.zeros(len(bnd_labels), max_len, dtype=torch.float)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)

    for i, (f, c, b, l) in enumerate(zip(features, cls_labels, bnd_labels, lengths)):
        padded_features[i, :l, :] = f
        padded_cls[i, :l] = c
        padded_bnd[i, :l] = b
        mask[i, :l] = True

    return {
        "features": padded_features,
        "cls_labels": padded_cls,
        "bnd_labels": padded_bnd,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_dataloaders(
    train_path=config.TRAIN_METADATA_PATH,
    val_path=config.VAL_METADATA_PATH,
    test_path=config.TEST_METADATA_PATH,
    batch_size=config.TRAIN_CONFIG["batch_size"],
    num_workers=config.TRAIN_CONFIG["num_workers"],
    limit=config.TRAIN_CONFIG["debug_subset_size"],
):

    train_ds = GestureDataset(train_path, mode="train", limit=limit)
    val_ds = GestureDataset(val_path, mode="val", limit=limit)
    test_ds = GestureDataset(test_path, mode="test", limit=None)

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
