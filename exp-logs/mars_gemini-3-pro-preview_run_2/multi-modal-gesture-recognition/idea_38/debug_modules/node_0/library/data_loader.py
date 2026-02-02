import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from library.config import Config


class GestureDataset(Dataset):
    def __init__(self, metadata_file, split="train", load_cached_data=True):
        """
        Dataset for Multimodal Gesture Recognition.
        Handles loading, preprocessing, caching, and augmentation.
        """
        self.split = split
        self.is_train = split == "train"
        self.metadata = pd.read_csv(metadata_file)

        # Ensure cache directory exists
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{split}_data.npz")

        # Joint Mapping (based on dataset description order)
        self.joint_map = {
            "HipCenter": 0,
            "Spine": 1,
            "ShoulderCenter": 2,
            "Head": 3,
            "ShoulderLeft": 4,
            "ElbowLeft": 5,
            "WristLeft": 6,
            "HandLeft": 7,
            "ShoulderRight": 8,
            "ElbowRight": 9,
            "WristRight": 10,
            "HandRight": 11,
            "HipLeft": 12,
            "KneeLeft": 13,
            "AnkleLeft": 14,
            "FootLeft": 15,
            "HipRight": 16,
            "KneeRight": 17,
            "AnkleRight": 18,
            "FootRight": 19,
        }
        # Get indices for the 12 upper body joints defined in Config
        self.target_joint_indices = [
            self.joint_map[j] for j in Config.UPPER_BODY_JOINTS
        ]

        # Data Containers
        self.features = []
        self.labels = []
        self.boundaries = []
        self.lengths = []
        self.ids = []

        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache()

    def _load_cache(self):
        # print(f"Loading {self.split} data from cache: {self.cache_path}")
        try:
            data = np.load(self.cache_path)
            padded_features = data["features"]
            padded_labels = data["labels"]
            padded_boundaries = data["boundaries"]
            lengths = data["lengths"]
            self.ids = data["ids"]

            # Slice back to variable lengths
            self.features = [padded_features[i, :l] for i, l in enumerate(lengths)]
            self.labels = [padded_labels[i, :l] for i, l in enumerate(lengths)]
            self.boundaries = [padded_boundaries[i, :l] for i, l in enumerate(lengths)]
            self.lengths = lengths
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            self._process_and_cache()

    def _process_and_cache(self):
        # print(f"Processing {self.split} data from scratch...")

        features_list = []
        labels_list = []
        boundaries_list = []
        ids_list = []
        lengths_list = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            try:
                # --- 1. Load Skeleton Data ---
                mat = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                if "Video" not in mat:
                    continue
                video = mat["Video"]
                num_frames = video.NumFrames

                # Robustly extract skeleton frames
                frames_struct = video.Frames
                if not isinstance(frames_struct, np.ndarray):
                    frames_struct = np.array([frames_struct])

                # Extract positions: (T, 12, 3)
                skeleton_pos = np.zeros(
                    (num_frames, len(self.target_joint_indices), 3), dtype=np.float32
                )

                for f_idx, frame_obj in enumerate(frames_struct):
                    if f_idx >= num_frames:
                        break
                    try:
                        skel = frame_obj.Skeleton
                        # Handle array of skeletons (multi-user) -> take first
                        if isinstance(skel, np.ndarray) and len(skel) > 0:
                            skel = skel[0]

                        if hasattr(skel, "WorldPosition"):
                            wp = skel.WorldPosition
                            # wp is expected to be (20, 3)
                            if isinstance(wp, np.ndarray) and wp.shape[0] >= 20:
                                skeleton_pos[f_idx] = wp[self.target_joint_indices, :3]
                    except AttributeError:
                        pass

                # --- 2. Normalize Skeleton ---
                # Center to HipCenter (Index 0 in UPPER_BODY_JOINTS)
                hip_pos = skeleton_pos[:, 0:1, :]  # (T, 1, 3)
                skeleton_pos = skeleton_pos - hip_pos

                # Scale mm to meters
                skeleton_pos = skeleton_pos * Config.SCALE_FACTOR

                # --- 3. Compute Velocity ---
                # V_t = P_t - P_{t-1}
                velocity = np.zeros_like(skeleton_pos)
                velocity[1:] = skeleton_pos[1:] - skeleton_pos[:-1]

                # Flatten: (T, 36)
                feat_pos = skeleton_pos.reshape(num_frames, -1)
                feat_vel = velocity.reshape(num_frames, -1)

                # --- 4. Audio Features ---
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    # Mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # Compute MFCC aligned to video frames
                    total_samples = waveform.shape[1]
                    hop_length = max(1, int(total_samples / num_frames))

                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc=Config.AUDIO_MFCC_DIM,
                        melkwargs={
                            "n_fft": 2048,
                            "hop_length": hop_length,
                            "n_mels": 64,
                            "center": False,
                        },
                    )
                    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
                    mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

                    # Interpolate to match exact frame count
                    if mfcc.shape[0] != num_frames:
                        mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, C, T)
                        mfcc = F.interpolate(
                            mfcc, size=num_frames, mode="linear", align_corners=False
                        )
                        mfcc = mfcc.transpose(1, 2).squeeze(0)  # (T, C)

                    feat_audio = mfcc.numpy()
                except Exception:
                    # Fallback for missing/corrupt audio
                    feat_audio = np.zeros(
                        (num_frames, Config.AUDIO_MFCC_DIM), dtype=np.float32
                    )

                # Combine Features
                full_features = np.concatenate(
                    [feat_pos, feat_vel, feat_audio], axis=1
                )  # (T, 85)

                # --- 5. Labels ---
                frame_labels = np.zeros(num_frames, dtype=np.int64)
                boundary_labels = np.zeros(num_frames, dtype=np.float32)

                if hasattr(video, "Labels"):
                    labels_raw = video.Labels
                    if not isinstance(labels_raw, np.ndarray):
                        labels_raw = (
                            np.array([labels_raw]) if labels_raw is not None else []
                        )
                    elif labels_raw.ndim == 0:
                        labels_raw = (
                            np.array([labels_raw.item()])
                            if labels_raw.item() is not None
                            else []
                        )

                    for l in labels_raw:
                        try:
                            name = l.Name
                            # 1-based to 0-based
                            start = int(l.Begin) - 1
                            end = int(l.End) - 1

                            if name in Config.GESTURE_MAP:
                                gid = Config.GESTURE_MAP[name]
                                start = max(0, start)
                                end = min(num_frames - 1, end)

                                frame_labels[start : end + 1] = gid
                                boundary_labels[start] = 1.0
                                boundary_labels[end] = 1.0
                        except AttributeError:
                            pass

                features_list.append(full_features.astype(np.float32))
                labels_list.append(frame_labels)
                boundaries_list.append(boundary_labels)
                ids_list.append(sample_id)
                lengths_list.append(num_frames)

            except Exception as e:
                # print(f"Error processing {sample_id}: {e}")
                if self.split == "test":
                    # Add dummy for test set to maintain alignment
                    features_list.append(
                        np.zeros((10, Config.INPUT_DIM), dtype=np.float32)
                    )
                    labels_list.append(np.zeros(10, dtype=np.int64))
                    boundaries_list.append(np.zeros(10, dtype=np.float32))
                    ids_list.append(sample_id)
                    lengths_list.append(10)

        # Pad and Save
        if lengths_list:
            max_len = max(lengths_list)
            N = len(features_list)

            padded_features = np.zeros((N, max_len, Config.INPUT_DIM), dtype=np.float32)
            padded_labels = np.zeros((N, max_len), dtype=np.int64)
            padded_boundaries = np.zeros((N, max_len), dtype=np.float32)

            for i in range(N):
                l = lengths_list[i]
                padded_features[i, :l] = features_list[i]
                padded_labels[i, :l] = labels_list[i]
                padded_boundaries[i, :l] = boundaries_list[i]

            np.savez(
                self.cache_path,
                features=padded_features,
                labels=padded_labels,
                boundaries=padded_boundaries,
                lengths=np.array(lengths_list),
                ids=ids_list,
            )

            self.features = features_list
            self.labels = labels_list
            self.boundaries = boundaries_list
            self.lengths = lengths_list
            self.ids = ids_list
        else:
            print("Warning: No data processed.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # (T, 85)
        label = self.labels[idx]  # (T,)
        bnd = self.boundaries[idx]  # (T,)

        # Physically Consistent Augmentation (Train Only)
        if self.is_train:
            # Split features
            pos = feat[:, : Config.SKELETON_DIM]
            vel = feat[
                :, Config.SKELETON_DIM : Config.SKELETON_DIM + Config.VELOCITY_DIM
            ]
            audio = feat[:, -Config.AUDIO_MFCC_DIM :]

            # 1. Generate Noise
            noise = np.random.normal(0, Config.AUG_NOISE_SIGMA, pos.shape).astype(
                np.float32
            )
            # 2. Smooth Noise (Temporal Low Pass)
            noise = gaussian_filter1d(noise, sigma=Config.AUG_FILTER_SIZE, axis=0)

            # 3. Add to Position
            aug_pos = pos + noise

            # 4. Recompute Velocity from Augmented Position
            # Reshape to (T, J, 3) for correct vector subtraction, then flatten
            T = aug_pos.shape[0]
            aug_pos_reshaped = aug_pos.reshape(T, -1, 3)
            aug_vel_reshaped = np.zeros_like(aug_pos_reshaped)
            aug_vel_reshaped[1:] = aug_pos_reshaped[1:] - aug_pos_reshaped[:-1]
            aug_vel = aug_vel_reshaped.reshape(T, -1)

            # Reassemble
            feat = np.concatenate([aug_pos, aug_vel, audio], axis=1)

        return torch.from_numpy(feat), torch.from_numpy(label), torch.from_numpy(bnd)


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch and creates masks.
    """
    features, labels, boundaries = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    max_len = lengths.max().item()

    B = len(features)
    D = features[0].shape[1]

    padded_features = torch.zeros(B, max_len, D)
    padded_labels = torch.zeros(B, max_len, dtype=torch.long)
    padded_boundaries = torch.zeros(B, max_len, dtype=torch.float)

    for i in range(B):
        l = lengths[i]
        padded_features[i, :l] = features[i]
        padded_labels[i, :l] = labels[i]
        padded_boundaries[i, :l] = boundaries[i]

    # Create Boolean Mask (True where data exists)
    mask = torch.arange(max_len).expand(B, max_len) < lengths.unsqueeze(1)

    return padded_features, padded_labels, padded_boundaries, mask, lengths


def get_dataloaders():
    """Factory function for DataLoaders."""
    train_ds = GestureDataset(Config.TRAIN_METADATA, split="train")
    val_ds = GestureDataset(Config.VAL_METADATA, split="val")
    test_ds = GestureDataset(Config.TEST_METADATA, split="test")

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
