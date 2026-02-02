import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    SELECTED_JOINTS,
    GESTURE_MAP,
    NUM_MFCC,
    RANDOM_SEED,
)

# Set fixed seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_audio_features(audio_path, num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    """
    full_path = os.path.join(INPUT_DIR, audio_path)
    if not os.path.exists(full_path):
        return np.zeros((num_frames, NUM_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(full_path)

        # Compute MFCC
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = transform(waveform)  # (Channels, n_mfcc, time)

        # Average channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = mfcc.mean(dim=0)
        else:
            mfcc = mfcc.squeeze(0)

        # Interpolate to match video frames
        # Input to interpolate must be (Batch, Channels, Time)
        mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
        mfcc = torch.nn.functional.interpolate(
            mfcc, size=num_frames, mode="linear", align_corners=False
        )
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (num_frames, n_mfcc)

        return mfcc.numpy()

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((num_frames, NUM_MFCC), dtype=np.float32)


def physically_consistent_augmentation(joints_3d):
    """
    Applies temporally correlated noise to positions and derives velocity.
    """
    T, J, C = joints_3d.shape

    # 1. Generate Gaussian Noise (5mm std dev)
    noise = np.random.normal(0, 0.005, size=(T, J, C))

    # 2. Temporal Low-Pass Filter (Sigma=1.0)
    noise_smooth = scipy.ndimage.gaussian_filter1d(noise, sigma=1.0, axis=0)

    # 3. Add to positions
    aug_pos = joints_3d + noise_smooth

    # 4. Derive Velocity
    aug_vel = np.zeros_like(aug_pos)
    if T > 1:
        aug_vel[1:] = aug_pos[1:] - aug_pos[:-1]
        aug_vel[0] = aug_vel[1]

    return aug_pos.astype(np.float32), aug_vel.astype(np.float32)


def parse_mat_file(data_path):
    """
    Robustly parses .mat file to extract skeleton and labels.
    """
    full_path = os.path.join(INPUT_DIR, data_path)
    try:
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = video.NumFrames
        frames = video.Frames

        # Validate frames
        if isinstance(frames, np.ndarray):
            if len(frames) != num_frames:
                num_frames = len(frames)
        elif not isinstance(frames, np.ndarray) and num_frames == 1:
            frames = [frames]
        else:
            raise ValueError(f"Ambiguous Frames structure in {data_path}")

        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        for i in range(num_frames):
            frame_obj = frames[i]
            if not hasattr(frame_obj, "Skeleton"):
                continue

            skel = frame_obj.Skeleton
            # Handle array of skeletons (multi-user) -> take first
            if isinstance(skel, np.ndarray) and len(skel) > 0:
                skel = skel[0]

            if hasattr(skel, "WorldPosition"):
                wp = skel.WorldPosition
                if isinstance(wp, np.ndarray):
                    if wp.shape == (20, 3):
                        skeleton_data[i] = wp
                    elif wp.shape == (3, 20):
                        skeleton_data[i] = wp.T

        # Parse Labels
        labels_parsed = []
        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0:
                    raw_labels = [raw_labels.item()]
            elif not isinstance(raw_labels, list):
                raw_labels = [raw_labels]

            for l in raw_labels:
                try:
                    name = l.Name
                    if name in GESTURE_MAP:
                        labels_parsed.append(
                            {
                                "id": GESTURE_MAP[name],
                                "start": int(l.Begin),
                                "end": int(l.End),
                            }
                        )
                except AttributeError:
                    pass

        return skeleton_data, labels_parsed, num_frames

    except Exception as e:
        raise ValueError(f"Failed to parse {data_path}: {e}")


class GestureDataset(Dataset):
    def __init__(self, metadata_path, is_train=True, load_cached_data=True, limit=None):
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_path)
        if limit:
            self.metadata = self.metadata.iloc[:limit]

        dataset_type = (
            "train" if is_train else "val" if "val" in metadata_path else "test"
        )
        self.cache_path = os.path.join(CACHE_DIR, f"{dataset_type}_data.npz")

        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache()

    def _process_and_cache(self):
        print(f"Processing data for {self.cache_path}...")

        # Temporary lists to collect data
        all_skeleton = []
        all_mfcc = []
        all_cls = []
        all_bnd = []
        boundaries = []  # [start_index, length]
        sample_ids = []

        current_idx = 0

        for _, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            try:
                # 1. Parse
                skeleton, labels_info, num_frames = parse_mat_file(row["data_path"])

                # Defensive Check
                if skeleton.shape[1] != 20 or skeleton.shape[2] != 3:
                    raise ValueError(f"Invalid skeleton shape {skeleton.shape}")

                # 2. Extract & Normalize
                # Select joints
                skel_sel = skeleton[:, SELECTED_JOINTS, :]  # (T, 12, 3)
                # Center to Hip (HipCenter is index 0 in SELECTED_JOINTS)
                hip_pos = skel_sel[:, 0:1, :]
                skel_norm = (skel_sel - hip_pos) * 0.001  # Scale to meters

                # 3. Audio
                mfcc = load_audio_features(row["audio_path"], num_frames)

                # 4. Targets
                t_cls = np.zeros(num_frames, dtype=np.int64)
                t_bnd = np.zeros(num_frames, dtype=np.float32)

                # Only generate targets if we have labels (Train/Val)
                # Test set has empty labels, so this loop won't run or labels_info is empty
                for l in labels_info:
                    gid = l["id"]
                    start = max(0, l["start"] - 1)
                    end = min(num_frames, l["end"])
                    t_cls[start:end] = gid
                    if start < num_frames:
                        t_bnd[start] = 1.0
                    if end - 1 >= 0 and end - 1 < num_frames:
                        t_bnd[end - 1] = 1.0

                # Collect
                all_skeleton.append(skel_norm.astype(np.float32))
                all_mfcc.append(mfcc.astype(np.float32))
                all_cls.append(t_cls)
                all_bnd.append(t_bnd)

                boundaries.append([current_idx, num_frames])
                sample_ids.append(sample_id)

                current_idx += num_frames

            except Exception as e:
                print(f"Skipping {sample_id}: {e}")
                # Fail loudly if strictness is required, but skipping allows partial dataset
                # Given prompt requirements, we raise to ensure data integrity
                raise e

        # Concatenate
        self.skeleton_flat = np.concatenate(all_skeleton, axis=0)
        self.mfcc_flat = np.concatenate(all_mfcc, axis=0)
        self.cls_flat = np.concatenate(all_cls, axis=0)
        self.bnd_flat = np.concatenate(all_bnd, axis=0)
        self.boundaries = np.array(boundaries, dtype=np.int32)
        self.sample_ids = np.array(sample_ids, dtype="S")  # Bytes to avoid pickle

        # Save
        np.savez_compressed(
            self.cache_path,
            skeleton=self.skeleton_flat,
            mfcc=self.mfcc_flat,
            cls=self.cls_flat,
            bnd=self.bnd_flat,
            boundaries=self.boundaries,
            sample_ids=self.sample_ids,
        )
        print(f"Cached {len(self.sample_ids)} samples.")

    def _load_cache(self):
        print(f"Loading cache from {self.cache_path}...")
        data = np.load(
            self.cache_path
        )  # allow_pickle=False by default usually fine for these types
        self.skeleton_flat = data["skeleton"]
        self.mfcc_flat = data["mfcc"]
        self.cls_flat = data["cls"]
        self.bnd_flat = data["bnd"]
        self.boundaries = data["boundaries"]
        self.sample_ids = data["sample_ids"]

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, idx):
        start, length = self.boundaries[idx]
        end = start + length

        # Slice data
        skel = self.skeleton_flat[start:end]  # (T, 12, 3)
        mfcc = self.mfcc_flat[start:end]  # (T, 13)
        t_cls = self.cls_flat[start:end]
        t_bnd = self.bnd_flat[start:end]
        s_id = self.sample_ids[idx].astype(str)

        # Augmentation
        if self.is_train:
            pos, vel = physically_consistent_augmentation(skel)
        else:
            pos = skel
            vel = np.zeros_like(pos)
            if len(pos) > 1:
                vel[1:] = pos[1:] - pos[:-1]
                vel[0] = vel[1]

        # Flatten and Concatenate Features
        # pos: (T, 12, 3) -> (T, 36)
        pos_flat = pos.reshape(length, -1)
        vel_flat = vel.reshape(length, -1)

        features = np.concatenate([pos_flat, vel_flat, mfcc], axis=1)

        return {
            "features": torch.from_numpy(features).float(),
            "target_cls": torch.from_numpy(t_cls).long(),
            "target_bnd": torch.from_numpy(t_bnd).float(),
            "sample_id": str(s_id),
        }


def collate_fn(batch):
    # Sort for packing (descending length)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    target_cls = [x["target_cls"] for x in batch]
    target_bnd = [x["target_bnd"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    # Pad
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    target_cls_padded = pad_sequence(target_cls, batch_first=True, padding_value=0)
    target_bnd_padded = pad_sequence(target_bnd, batch_first=True, padding_value=0.0)

    # Mask
    lengths = torch.tensor([len(x) for x in features])
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return {
        "features": features_padded,
        "target_cls": target_cls_padded,
        "target_bnd": target_bnd_padded,
        "mask": mask.float(),
        "sample_ids": ids,
    }


def get_dataloaders(batch_size=32, num_workers=4, load_cached=True):
    train_ds = GestureDataset(
        TRAIN_METADATA_PATH, is_train=True, load_cached_data=load_cached
    )
    val_ds = GestureDataset(
        VAL_METADATA_PATH, is_train=False, load_cached_data=load_cached
    )
    test_ds = GestureDataset(
        TEST_METADATA_PATH, is_train=False, load_cached_data=load_cached
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
