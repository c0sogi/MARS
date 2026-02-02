import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import scipy.signal
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


def parse_skeleton_data(mat_data, num_frames):
    """
    Parses the nested struct from scipy.io.loadmat to extract skeleton joints.
    Returns: (NumFrames, NumJoints, 3)
    """
    try:
        video = mat_data["Video"]
        frames = video.Frames

        # Pre-allocate
        # Config.SELECTED_JOINTS indices correspond to the order in the dataset description
        # The dataset description lists 20 joints. We need to map Config.SELECTED_JOINTS to the raw indices.
        # Assuming the order in 'JointsType' matches the index (0-based)
        # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head, ...

        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        # Handle case where frames might be a single object or array
        if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
            frames = [frames]

        iter_frames = min(len(frames), num_frames)

        for f_idx in range(iter_frames):
            frame_obj = frames[f_idx]
            # Access Skeleton
            if hasattr(frame_obj, "Skeleton"):
                skel = frame_obj.Skeleton
                # Access WorldPosition
                if hasattr(skel, "WorldPosition"):
                    wp = skel.WorldPosition
                    # Check if WorldPosition is an array of structs (one per joint)
                    # or if we need to iterate joints

                    # In many chalearn datasets, wp is a struct array of size 20
                    # We extract specific joints
                    if isinstance(wp, np.ndarray) or isinstance(wp, list):
                        for i, joint_idx in enumerate(Config.SELECTED_JOINTS):
                            if joint_idx < len(wp):
                                joint = wp[joint_idx]
                                if hasattr(joint, "X"):
                                    skeleton_data[f_idx, i, 0] = joint.X
                                    skeleton_data[f_idx, i, 1] = joint.Y
                                    skeleton_data[f_idx, i, 2] = joint.Z
                                else:
                                    skeleton_data[f_idx, i, 0] = joint[0]
                                    skeleton_data[f_idx, i, 1] = joint[1]
                                    skeleton_data[f_idx, i, 2] = joint[2]

        return skeleton_data
    except Exception as e:
        # Return zeros if parsing fails
        print(f"Error parsing skeleton: {e}")
        return np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)


def extract_audio_features(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCC, and aligns to video frames.
    Returns: (NumFrames, N_MFCC)
    """
    if not os.path.exists(audio_path):
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sample_rate, Config.AUDIO_SR)
            waveform = resampler(waveform)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SR,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = mfcc.mean(dim=0, keepdim=True)

        # mfcc shape: (1, n_mfcc, time_steps)
        # We need to interpolate to (1, n_mfcc, target_num_frames)
        mfcc = F.interpolate(
            mfcc.unsqueeze(0),
            size=target_num_frames,
            mode="linear",
            align_corners=False,
        )

        # Squeeze to (n_mfcc, target_num_frames) -> transpose to (target_num_frames, n_mfcc)
        mfcc = mfcc.squeeze(0).squeeze(0).permute(1, 0)

        return mfcc.numpy()

    except Exception as e:
        # print(f"Error extracting audio: {e}")
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def generate_targets(mat_data, num_frames):
    """
    Parses labels to generate frame-wise class and boundary targets.
    """
    cls_target = np.zeros(num_frames, dtype=np.int64)  # 0 is background
    bnd_target = np.zeros(num_frames, dtype=np.float32)

    try:
        video = mat_data["Video"]
        if not hasattr(video, "Labels"):
            return cls_target, bnd_target

        labels_raw = video.Labels

        # Helper to process single label
        def process_label(obj):
            try:
                name = obj.Name
                start = int(obj.Begin) - 1  # Matlab 1-based to Python 0-based
                end = int(
                    obj.End
                )  # Exclusive in slice, so End is fine if we treat it as inclusive in Matlab
                # Actually Matlab 1:10 means 10 frames. Python 0:10 means 10 frames.
                # So start=Begin-1, end=End.

                if name in Config.GESTURE_MAP:
                    gid = Config.GESTURE_MAP[name]
                    # Clip to valid range
                    s = max(0, start)
                    e = min(num_frames, end)
                    if s < e:
                        cls_target[s:e] = gid
            except:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                process_label(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label(l)
        elif isinstance(labels_raw, list):
            for l in labels_raw:
                process_label(l)
        else:
            process_label(labels_raw)

        # Generate boundaries: 1 where label changes
        # Diff != 0
        diff = np.diff(cls_target, prepend=cls_target[0])
        bnd_target[diff != 0] = 1.0

        return cls_target, bnd_target

    except Exception as e:
        # print(f"Error generating targets: {e}")
        return cls_target, bnd_target


class GestureDataset(Dataset):
    def __init__(self, metadata_path, split="train", load_cached_data=True):
        self.split = split
        self.metadata = pd.read_csv(metadata_path)
        # Drop rows with missing file paths to prevent TypeErrors during path construction
        self.metadata = self.metadata.dropna(subset=["data_path", "audio_path"])

        # Debugging subset
        if Config.DEBUG:
            self.metadata = self.metadata.iloc[: Config.DEBUG_SUBSET_SIZE]

        self.sample_ids = self.metadata["sample_id"].tolist()

        # Cache paths
        cache_name = f"{split}_data_v2.npz"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        self.data_indices = []  # List of [start, length]
        self.all_features = None
        self.all_labels_cls = None
        self.all_labels_bnd = None

        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache()

    def _process_and_cache(self):
        print(f"Processing {self.split} data from scratch...")

        features_list = []
        labels_cls_list = []
        labels_bnd_list = []
        indices_list = []

        current_idx = 0

        for idx, row in self.metadata.iterrows():
            # Paths
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Load MAT
            try:
                mat = scipy.io.loadmat(
                    mat_path, squeeze_me=True, struct_as_record=False
                )
                num_frames = getattr(mat["Video"], "NumFrames", 0)
            except:
                num_frames = 0
                mat = {}

            if num_frames == 0:
                # Handle empty/corrupt
                # Create dummy 1-frame data
                num_frames = 1
                feat = np.zeros((1, Config.INPUT_DIM), dtype=np.float32)
                l_cls = np.zeros(1, dtype=np.int64)
                l_bnd = np.zeros(1, dtype=np.float32)
            else:
                # 1. Skeleton
                skel = parse_skeleton_data(mat, num_frames)  # (T, J, 3)

                # Normalize
                # Center around HipCenter (Index 0 in SELECTED_JOINTS)
                hip_pos = skel[
                    :, Config.CENTER_JOINT_IDX : Config.CENTER_JOINT_IDX + 1, :
                ]
                skel = (skel - hip_pos) * Config.SCALE_FACTOR

                # Velocity
                vel = np.zeros_like(skel)
                vel[1:] = skel[1:] - skel[:-1]

                # Flatten Skeleton: (T, J*3)
                skel_flat = skel.reshape(num_frames, -1)
                vel_flat = vel.reshape(num_frames, -1)

                # 2. Audio
                audio_feat = extract_audio_features(
                    audio_path, num_frames
                )  # (T, N_MFCC)

                # Concatenate
                feat = np.concatenate(
                    [skel_flat, vel_flat, audio_feat], axis=1
                )  # (T, Dim)

                # 3. Targets
                if self.split in ["train", "val"]:
                    l_cls, l_bnd = generate_targets(mat, num_frames)
                else:
                    l_cls = np.zeros(num_frames, dtype=np.int64)
                    l_bnd = np.zeros(num_frames, dtype=np.float32)

            # Store
            length = feat.shape[0]
            features_list.append(feat)
            labels_cls_list.append(l_cls)
            labels_bnd_list.append(l_bnd)
            indices_list.append([current_idx, length])
            current_idx += length

        # Concatenate all
        self.all_features = np.concatenate(features_list, axis=0).astype(np.float32)
        self.all_labels_cls = np.concatenate(labels_cls_list, axis=0).astype(np.int64)
        self.all_labels_bnd = np.concatenate(labels_bnd_list, axis=0).astype(np.float32)
        self.data_indices = np.array(indices_list, dtype=np.int64)

        # Save to cache (No Pickle)
        np.savez(
            self.cache_path,
            features=self.all_features,
            labels_cls=self.all_labels_cls,
            labels_bnd=self.all_labels_bnd,
            indices=self.data_indices,
        )
        print(f"Saved cache to {self.cache_path}")

    def _load_cache(self):
        print(f"Loading cached {self.split} data from {self.cache_path}...")
        data = np.load(self.cache_path)
        self.all_features = data["features"]
        self.all_labels_cls = data["labels_cls"]
        self.all_labels_bnd = data["labels_bnd"]
        self.data_indices = data["indices"]

    def _augment_physically_consistent(self, features):
        """
        Applies smooth noise to positions and re-derives velocity.
        Features: [Pos(J*3), Vel(J*3), Audio(MFCC)]
        """
        T, D = features.shape
        num_joints = Config.NUM_JOINTS
        pos_dim = num_joints * 3

        # Split features
        pos = features[:, :pos_dim].reshape(T, num_joints, 3)
        # vel = features[:, pos_dim:pos_dim*2] # We will overwrite this
        audio = features[:, pos_dim * 2 :]

        # Generate Noise
        sigma = 0.005  # 5mm noise (since scaled to meters)
        noise = np.random.normal(0, sigma, size=(T, num_joints, 3))

        # Temporal Low-Pass Filter (Simple Moving Average)
        # Kernel size 5
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        # Apply along time axis (0)
        noise_smooth = np.zeros_like(noise)
        for j in range(num_joints):
            for c in range(3):
                noise_smooth[:, j, c] = np.convolve(noise[:, j, c], kernel, mode="same")

        # Add noise
        pos_aug = pos + noise_smooth

        # Re-derive Velocity
        vel_aug = np.zeros_like(pos_aug)
        vel_aug[1:] = pos_aug[1:] - pos_aug[:-1]

        # Flatten
        pos_aug_flat = pos_aug.reshape(T, -1)
        vel_aug_flat = vel_aug.reshape(T, -1)

        # Re-assemble
        return np.concatenate([pos_aug_flat, vel_aug_flat, audio], axis=1).astype(
            np.float32
        )

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        start, length = self.data_indices[idx]

        # Extract slices
        feat = self.all_features[
            start : start + length
        ].copy()  # Copy to allow augmentation
        target_cls = self.all_labels_cls[start : start + length]
        target_bnd = self.all_labels_bnd[start : start + length]

        # Augmentation (Train only)
        if self.split == "train":
            feat = self._augment_physically_consistent(feat)

        return {
            "features": torch.from_numpy(feat),
            "target_cls": torch.from_numpy(target_cls),
            "target_bnd": torch.from_numpy(target_bnd),
            "sample_id": self.sample_ids[idx],
        }


def collate_fn(batch):
    """
    Pads sequences to max length in batch.
    """
    # Sort by length (descending) for packing if needed (though we use masking)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    target_cls = [x["target_cls"] for x in batch]
    target_bnd = [x["target_bnd"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    max_len = lengths.max().item()

    # Pad Features
    # (B, T, D)
    padded_features = torch.zeros(len(batch), max_len, features[0].shape[1])
    # Pad Targets
    padded_cls = torch.zeros(len(batch), max_len, dtype=torch.long)  # 0 is background
    padded_bnd = torch.zeros(len(batch), max_len, dtype=torch.float)
    # Mask
    mask = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, (f, tc, tb, l) in enumerate(zip(features, target_cls, target_bnd, lengths)):
        padded_features[i, :l, :] = f
        padded_cls[i, :l] = tc
        padded_bnd[i, :l] = tb
        mask[i, :l] = 1.0

    return {
        "features": padded_features,
        "target_cls": padded_cls,
        "target_bnd": padded_bnd,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_loaders(load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test.
    """
    set_seed(Config.SEED)

    train_ds = GestureDataset(
        Config.TRAIN_METADATA_PATH, split="train", load_cached_data=load_cached_data
    )
    val_ds = GestureDataset(
        Config.VAL_METADATA_PATH, split="val", load_cached_data=load_cached_data
    )
    test_ds = GestureDataset(
        Config.TEST_METADATA_PATH, split="test", load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
