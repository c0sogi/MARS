import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, collate_fn


class GestureDataset(Dataset):
    def __init__(self, metadata_df, cache_file, is_train=True, augment=False):
        self.is_train = is_train
        self.augment = augment
        self.data = self._load_data(metadata_df, cache_file)

    def _load_data(self, df, cache_file):
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if os.path.exists(cache_file):
            try:
                return self._load_from_cache(cache_file)
            except Exception as e:
                print(f"Failed to load cache {cache_file}: {e}. Recomputing...")

        data = self._process_raw_data(df)
        self._save_to_cache(data, cache_file)
        return data

    def _process_raw_data(self, df):
        processed_data = []

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Parse Labels (Target Sequence)
            target_seq = []
            if "labels" in row and pd.notna(row["labels"]):
                if isinstance(row["labels"], str):
                    target_seq = [int(x) for x in row["labels"].split() if x.strip()]
                elif isinstance(row["labels"], list):
                    target_seq = row["labels"]

            try:
                # Load MAT file
                # struct_as_record=False allows accessing fields as attributes
                # squeeze_me=True simplifies 1x1 matrices to scalars
                mat = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                if "Video" not in mat:
                    continue

                video = mat["Video"]
                num_frames = getattr(video, "NumFrames", 0)
                if num_frames == 0:
                    continue

                # Extract Skeleton Features
                skeleton_feats = self._extract_skeleton(video, num_frames)

                # Extract Audio Features
                audio_feats = self._extract_audio(audio_path, num_frames)

                # Concatenate: (T, SkeletonDim) + (T, AudioDim) -> (T, InputDim)
                features = np.concatenate([skeleton_feats, audio_feats], axis=1)

                # Generate Frame-wise Labels (for training/val)
                frame_labels = np.zeros(num_frames, dtype=np.int64)
                if self.is_train and hasattr(video, "Labels"):
                    labels_raw = video.Labels
                    # Normalize labels_raw to a list/array
                    if not isinstance(labels_raw, np.ndarray) and not isinstance(
                        labels_raw, list
                    ):
                        labels_raw = [labels_raw] if labels_raw is not None else []
                    elif isinstance(labels_raw, np.ndarray) and labels_raw.ndim == 0:
                        labels_raw = [labels_raw.item()]

                    for l in labels_raw:
                        try:
                            # Check for valid label object
                            if not hasattr(l, "Name"):
                                continue

                            name = l.Name
                            # MATLAB indices are 1-based, inclusive
                            # Convert to Python 0-based, exclusive for slice end
                            start = int(l.Begin) - 1
                            end = int(l.End)

                            if name in Config.GESTURE_MAP:
                                gid = Config.GESTURE_MAP[name]
                                # Clip to valid range
                                start = max(0, start)
                                end = min(num_frames, end)
                                if end > start:
                                    frame_labels[start:end] = gid
                        except Exception:
                            continue

                processed_data.append(
                    {
                        "sample_id": sample_id,
                        "features": features.astype(np.float32),
                        "frame_labels": frame_labels,
                        "target_sequence": target_seq,
                    }
                )

            except Exception as e:
                # Skip broken samples
                continue

        return processed_data

    def _extract_skeleton(self, video, num_frames):
        # Initialize (NumFrames, NumJoints, 3)
        joints_pos = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        frames = getattr(video, "Frames", [])

        # Ensure frames is iterable
        if isinstance(frames, np.ndarray) and frames.ndim == 0:
            frames = np.array([frames.item()])
        elif not isinstance(frames, (np.ndarray, list)):
            frames = []

        # Iterate over frames
        limit = min(len(frames), num_frames)
        for i in range(limit):
            f = frames[i]
            if hasattr(f, "Skeleton") and f.Skeleton is not None:
                skel = f.Skeleton
                # Handle array of skeletons (multiple users) - take first
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]

                if hasattr(skel, "WorldPosition"):
                    wp = skel.WorldPosition
                    # Check shape and transpose if necessary (3, 20) vs (20, 3)
                    if isinstance(wp, np.ndarray):
                        if wp.shape == (3, 20):
                            wp = wp.T

                        if wp.shape == (20, 3):
                            # Select specific joints
                            selected_wp = wp[Config.SELECTED_JOINTS, :]  # (12, 3)
                            joints_pos[i] = selected_wp

        # Normalize relative to HipCenter (Index 0 in SELECTED_JOINTS)
        # Config.SELECTED_JOINTS[0] corresponds to HipCenter
        hip_pos = joints_pos[:, 0:1, :]  # (T, 1, 3)
        joints_pos_norm = joints_pos - hip_pos

        # Flatten joints: (T, 12*3)
        joints_flat = joints_pos_norm.reshape(num_frames, -1)

        # Compute Velocity: V_t = P_t - P_{t-1}
        velocity = np.zeros_like(joints_flat)
        velocity[1:] = joints_flat[1:] - joints_flat[:-1]

        # Concatenate Position + Velocity -> (T, 72)
        return np.concatenate([joints_flat, velocity], axis=1)

    def _extract_audio(self, audio_path, num_frames):
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError

            waveform, sample_rate = torchaudio.load(audio_path)

            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = transform(waveform)  # (1, n_mfcc, time)

            # Interpolate to match video frames
            if mfcc.shape[-1] > 0:
                mfcc = F.interpolate(
                    mfcc.unsqueeze(0),
                    size=(mfcc.shape[1], num_frames),
                    mode="bilinear",
                    align_corners=False,
                )
                mfcc = mfcc.squeeze(0)  # (1, n_mfcc, num_frames)
            else:
                return np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

            # (1, n_mfcc, num_frames) -> (num_frames, n_mfcc)
            return mfcc.squeeze(0).transpose(0, 1).numpy()

        except Exception:
            # Return zeros if audio fails
            return np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _save_to_cache(self, data, cache_file):
        if not data:
            return

        # Prepare arrays
        all_features = []
        all_labels = []
        sample_ids = []
        target_seqs = []
        lengths = []

        for item in data:
            all_features.append(item["features"])
            all_labels.append(item["frame_labels"])
            sample_ids.append(item["sample_id"])
            # Serialize target sequence to string for storage
            target_seqs.append(" ".join(map(str, item["target_sequence"])))
            lengths.append(item["features"].shape[0])

        # Concatenate big arrays
        packed_features = np.concatenate(all_features, axis=0)
        packed_labels = np.concatenate(all_labels, axis=0)

        np.savez_compressed(
            cache_file,
            features=packed_features,
            labels=packed_labels,
            lengths=np.array(lengths, dtype=np.int32),
            sample_ids=np.array(sample_ids),
            target_seqs=np.array(target_seqs),
        )

    def _load_from_cache(self, cache_file):
        loaded = np.load(cache_file)
        features = loaded["features"]
        labels = loaded["labels"]
        lengths = loaded["lengths"]
        sample_ids = loaded["sample_ids"]
        target_seqs = loaded["target_seqs"]

        data = []
        start_idx = 0
        for i, length in enumerate(lengths):
            end_idx = start_idx + length

            # Parse target sequence
            seq_str = str(target_seqs[i])
            t_seq = (
                [int(x) for x in seq_str.split()] if len(seq_str.strip()) > 0 else []
            )

            data.append(
                {
                    "sample_id": str(sample_ids[i]),
                    "features": features[start_idx:end_idx],
                    "frame_labels": labels[start_idx:end_idx],
                    "target_sequence": t_seq,
                }
            )
            start_idx = end_idx

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        labels = torch.tensor(item["frame_labels"], dtype=torch.long)

        # Augmentation (Training only)
        if self.is_train and self.augment:
            # Apply Gaussian noise to skeleton features (first 72 dims)
            noise = torch.randn_like(features[:, : Config.SKELETON_INPUT_SIZE]) * 0.01
            features[:, : Config.SKELETON_INPUT_SIZE] += noise

        return {
            "sample_id": item["sample_id"],
            "features": features,
            "frame_labels": labels,
            "target_sequence": item["target_sequence"],
        }


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        train_df = train_df.iloc[:20]
        val_df = val_df.iloc[:10]
        test_df = test_df.iloc[:10]

    # Define Cache Paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_data.npz")

    # Create Datasets
    # Train: Augmentation ON
    train_ds = GestureDataset(train_df, train_cache, is_train=True, augment=True)

    # Val: Augmentation OFF
    val_ds = GestureDataset(val_df, val_cache, is_train=True, augment=False)

    # Test: Augmentation OFF, is_train=False
    test_ds = GestureDataset(test_df, test_cache, is_train=False, augment=False)

    # Create Loaders
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
