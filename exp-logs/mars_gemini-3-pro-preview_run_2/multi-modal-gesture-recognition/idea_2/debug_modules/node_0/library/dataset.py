import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, limit_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, try to load from .npz cache.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.is_train = split == "train"
        self.limit_size = limit_size

        # Select metadata file
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        else:
            self.metadata_path = Config.TEST_METADATA_PATH

        # Define cache path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

        # Load data (from cache or raw)
        self.features, self.labels, self.boundaries = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads data from cache or processes raw files.
        Returns:
            features (np.ndarray): Flattened feature array (TotalFrames, InputDim)
            labels (np.ndarray): Flattened label array (TotalFrames,)
            boundaries (np.ndarray): Index boundaries for each sample (NumSamples, 2)
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached data from {self.cache_path}...")
                data = np.load(self.cache_path)
                features = data["features"]
                labels = data["labels"]
                boundaries = data["boundaries"]

                if self.limit_size:
                    boundaries = boundaries[: self.limit_size]
                    # We don't slice features/labels strictly to save memory here,
                    # but boundaries control access.

                return features, labels, boundaries
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process Raw Data
        print(f"Processing raw data for {self.split} set...")
        df = pd.read_csv(self.metadata_path)

        # Parse labels column back to list of ints for train/val
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(
                lambda x: (
                    [int(i) for i in str(x).split()]
                    if pd.notna(x) and str(x).strip() != ""
                    else []
                )
            )

        if self.limit_size:
            df = df.iloc[: self.limit_size]

        all_features = []
        all_labels = []
        boundaries = []
        current_idx = 0

        for _, row in df.iterrows():
            sample_id = row["sample_id"]

            # Process single sample
            feats, labs = self._process_sample(row)

            # Check validity
            if feats is None:
                # Fallback for errors: create dummy data or skip
                # Ideally we shouldn't fail, but if file missing, we skip
                continue

            n_frames = feats.shape[0]

            all_features.append(feats)
            all_labels.append(labs)
            boundaries.append([current_idx, current_idx + n_frames])
            current_idx += n_frames

        # Concatenate
        if not all_features:
            raise RuntimeError("No data processed!")

        features_concat = np.concatenate(all_features, axis=0).astype(np.float32)
        labels_concat = np.concatenate(all_labels, axis=0).astype(np.int64)
        boundaries_concat = np.array(boundaries, dtype=np.int64)

        # 3. Save Cache (using np.savez to avoid pickle)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(
            self.cache_path,
            features=features_concat,
            labels=labels_concat,
            boundaries=boundaries_concat,
        )
        print(f"Saved cache to {self.cache_path}")

        return features_concat, labels_concat, boundaries_concat

    def _process_sample(self, row):
        """
        Reads raw files and extracts features/labels for a single sample.
        """
        try:
            # --- 1. Skeleton Features ---
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            # Load mat with struct_as_record=False to access fields as attributes
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)

            # Extract Skeleton
            # Assuming Video.Frames is an array of structs, each having Skeleton.WorldPosition
            frames_struct = getattr(video, "Frames", None)

            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            if frames_struct is not None and isinstance(frames_struct, np.ndarray):
                # Iterate frames
                for i, frame_obj in enumerate(frames_struct):
                    if i >= num_frames:
                        break
                    try:
                        # Access Skeleton -> WorldPosition
                        # Handle cases where Skeleton might be missing or different format
                        skel = getattr(frame_obj, "Skeleton", None)
                        if skel is not None:
                            # WorldPosition might be a 20x3 matrix or struct
                            wp = getattr(skel, "WorldPosition", None)
                            if (
                                wp is not None
                                and isinstance(wp, np.ndarray)
                                and wp.shape == (20, 3)
                            ):
                                skeleton_data[i] = wp
                    except AttributeError:
                        pass

            # Normalize Skeleton (Relative to HipCenter, index 0)
            hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_norm = skeleton_data - hip_center

            # Flatten joints: (T, 20*3)
            skel_feats = skeleton_norm.reshape(num_frames, -1)

            final_feats_list = [skel_feats]

            # Velocity
            if Config.USE_VELOCITY:
                # V_t = P_t - P_{t-1}
                # Pad first frame with 0
                velocity = np.zeros_like(skel_feats)
                velocity[1:] = skel_feats[1:] - skel_feats[:-1]
                final_feats_list.append(velocity)

            # --- 2. Audio Features ---
            if Config.USE_AUDIO:
                audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
                if os.path.exists(audio_path):
                    waveform, sample_rate = torchaudio.load(audio_path)

                    # MFCC
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc=Config.AUDIO_N_MFCC,
                        melkwargs={
                            "n_fft": 400,
                            "hop_length": 160,
                            "n_mels": 23,
                            "center": False,
                        },
                    )
                    mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

                    # Average over channels if stereo
                    if mfcc.shape[0] > 1:
                        mfcc = torch.mean(mfcc, dim=0, keepdim=True)

                    # Align to Video Frames using Interpolation
                    # Input to interpolate must be (Batch, Channels, Time)
                    # Current: (1, n_mfcc, time_audio)
                    mfcc = mfcc.unsqueeze(
                        0
                    )  # (1, 1, n_mfcc, time) -> actually interpolate expects (N, C, L)
                    # We want to resize the last dimension (Time)
                    # Shape: (1, n_mfcc, time_audio)
                    mfcc = F.interpolate(
                        mfcc, size=num_frames, mode="linear", align_corners=False
                    )
                    mfcc = (
                        mfcc.squeeze(0).transpose(0, 1).numpy()
                    )  # (NumFrames, n_mfcc)
                else:
                    # Missing audio fallback
                    mfcc = np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

                final_feats_list.append(mfcc)

            # Concatenate all features
            features = np.concatenate(final_feats_list, axis=1)  # (T, InputDim)

            # --- 3. Labels ---
            # Initialize with 0 (Background)
            labels = np.zeros(num_frames, dtype=np.int64)

            # Fill gestures if available (Train/Val)
            if self.split != "test":
                labels_struct = getattr(video, "Labels", [])

                # Helper to process single label object
                def process_label_obj(obj):
                    try:
                        name = obj.Name
                        start = int(obj.Begin) - 1  # Matlab 1-based -> Python 0-based
                        end = int(
                            obj.End
                        )  # End inclusive in Python slice? No, slice is exclusive.
                        # But we want to include frame 'End'.
                        # So slice should be start:end (if end is 1-based index)
                        # Example: Frames 1 to 10. Python indices 0 to 9.
                        # Begin=1, End=10.
                        # Python: 0 to 10 (exclusive).

                        # Map name to ID
                        # We need a reverse map or just use the provided list order
                        # The prompt provides a list 1..20.
                        # We can use the logic from metadata script or a hardcoded map.
                        # For robustness, I'll rely on the gesture map defined below.
                        g_id = self._get_gesture_id(name)
                        if g_id > 0:
                            # Clip to valid range
                            s = max(0, start)
                            e = min(num_frames, end)
                            labels[s:e] = g_id
                    except AttributeError:
                        pass

                if isinstance(labels_struct, np.ndarray):
                    if labels_struct.ndim == 0:
                        process_label_obj(labels_struct.item())
                    else:
                        for l in labels_struct:
                            process_label_obj(l)
                else:
                    process_label_obj(labels_struct)

            return features, labels

        except Exception as e:
            print(f"Error processing {row['sample_id']}: {e}")
            return None, None

    def _get_gesture_id(self, name):
        gesture_map = {
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
        return gesture_map.get(name, 0)

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, idx):
        start, end = self.boundaries[idx]

        # Extract sequence
        feat = self.features[start:end]  # (T, Dim)
        label = self.labels[start:end]  # (T,)

        # Convert to Tensor
        feat_tensor = torch.from_numpy(feat).float()
        label_tensor = torch.from_numpy(label).long()

        # Augmentation (Train only)
        if self.is_train:
            # Add Gaussian noise
            noise = torch.randn_like(feat_tensor) * 0.01
            feat_tensor += noise

        return feat_tensor, label_tensor

    @staticmethod
    def collate_fn(batch):
        """
        Pads sequences to the longest in the batch.
        Returns:
            features: (Batch, MaxTime, Dim)
            labels: (Batch, MaxTime)
            mask: (Batch, MaxTime) - 1 for valid, 0 for padded
        """
        features, labels = zip(*batch)

        # Get lengths
        lengths = [len(f) for f in features]
        max_len = max(lengths)

        # Pad features
        # pad_sequence expects list of tensors (L, D) -> (L_max, B, D) if batch_first=False
        # We want (Batch, MaxTime, Dim)
        features_padded = torch.nn.utils.rnn.pad_sequence(
            features, batch_first=True, padding_value=0
        )

        # Pad labels (fill with 0 - background)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0
        )

        # Create mask
        batch_size = len(features)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1

        return features_padded, labels_padded, mask


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=4):
    train_ds = GestureDataset(split="train")
    val_ds = GestureDataset(split="val")
    # Test set is loaded separately when needed, usually without shuffling

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=GestureDataset.collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=GestureDataset.collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
