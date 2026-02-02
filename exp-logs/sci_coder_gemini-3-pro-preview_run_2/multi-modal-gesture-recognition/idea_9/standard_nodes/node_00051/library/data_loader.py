import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


def extract_features(sample_info, is_train=True):
    """
    Extracts aligned multi-modal features (Skeleton + Audio) and frame-wise labels.

    Args:
        sample_info (dict): Row from the metadata dataframe.
        is_train (bool): Whether to extract ground truth labels.

    Returns:
        dict: {
            'features': np.ndarray (T, Input_Dim),
            'labels': np.ndarray (T,) or None,
            'num_frames': int
        }
    """
    # Initialize variables to ensure scope availability
    video = None

    # -------------------------------------------------------------------------
    # 1. Skeleton Processing
    # -------------------------------------------------------------------------
    try:
        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, str(sample_info["data_path"]))

        # Load MAT file with struct_as_record=False to access fields as attributes
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames_struct = getattr(video, "Frames", [])

        # Initialize skeleton array: (NumFrames, NumJoints, 3)
        # Standard Kinect skeleton has 20 joints
        full_skeleton = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Defensive parsing of the Frames structure
        if isinstance(frames_struct, np.ndarray) and len(frames_struct) == num_frames:
            for i, frame_obj in enumerate(frames_struct):
                try:
                    skel = frame_obj.Skeleton
                    # WorldPosition is usually (NumJoints, 3) or (3, NumJoints)
                    # We expect X, Y, Z coordinates
                    wp = skel.WorldPosition

                    if wp.shape == (20, 3):
                        full_skeleton[i] = wp
                    elif wp.shape == (3, 20):
                        full_skeleton[i] = wp.T
                    elif wp.size == 60:
                        full_skeleton[i] = wp.reshape(20, 3)
                    # If shape is completely wrong, leave as zeros (defensive)
                except AttributeError:
                    pass

        # Feature Selection: Upper Body Only
        # Shape: (T, 12, 3)
        upper_body = full_skeleton[:, Config.UPPER_BODY_JOINTS, :]

        # Normalization: Relative to HipCenter (Index 0 in Config.UPPER_BODY_JOINTS)
        # HipCenter is the first element because 0 is in the list
        hip_center = upper_body[:, 0:1, :]  # Keep dims for broadcasting
        norm_skeleton = upper_body - hip_center

        # Dynamics: Temporal Velocity
        # V_t = P_t - P_{t-1}
        # Pad first frame with zeros
        velocity = np.zeros_like(norm_skeleton)
        velocity[1:] = norm_skeleton[1:] - norm_skeleton[:-1]

        # Flatten Joint Features: (T, 12 * 6)
        # Concatenate Position and Velocity along the last axis
        joint_feats = np.concatenate([norm_skeleton, velocity], axis=-1)
        joint_feats_flat = joint_feats.reshape(num_frames, -1)

    except Exception as e:
        print(f"Error processing skeleton for {sample_info['sample_id']}: {e}")
        # Fallback: create zero features
        num_frames = int(sample_info.get("num_frames", 100))
        joint_feats_flat = np.zeros(
            (num_frames, Config.NUM_SKELETON_JOINTS * Config.JOINT_FEATS)
        )

    # -------------------------------------------------------------------------
    # 2. Audio Processing (MFCC)
    # -------------------------------------------------------------------------
    try:
        audio_path = os.path.join(Config.INPUT_DIR, str(sample_info["audio_path"]))
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary (though config says 16000)
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Compute MFCC
        # We need to align audio frames to video frames
        # Video FPS is roughly num_frames / duration, but we align by count
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.AUDIO_MFCC_DIM,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
        mfcc = mfcc.mean(dim=0).transpose(0, 1)  # (time, n_mfcc)

        # Align MFCC to Video Frames using Linear Interpolation
        # Input to interpolate must be (Batch, Channels, Length)
        mfcc_in = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, audio_frames)

        mfcc_aligned = F.interpolate(
            mfcc_in, size=num_frames, mode="linear", align_corners=False
        )

        # (num_frames, n_mfcc)
        audio_feats = mfcc_aligned.squeeze(0).transpose(0, 1).numpy()

    except Exception as e:
        # Fallback: zero audio features
        audio_feats = np.zeros((num_frames, Config.AUDIO_MFCC_DIM))

    # -------------------------------------------------------------------------
    # 3. Feature Concatenation
    # -------------------------------------------------------------------------
    # Ensure lengths match exactly
    min_len = min(len(joint_feats_flat), len(audio_feats))
    features = np.concatenate(
        [joint_feats_flat[:min_len], audio_feats[:min_len]], axis=1
    )

    # Update num_frames to match truncated length
    num_frames = min_len

    # -------------------------------------------------------------------------
    # 4. Label Generation (Frame-wise)
    # -------------------------------------------------------------------------
    labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

    if is_train and video is not None:
        try:
            # Parse Labels from MAT file
            # Video.Labels contains structures with Name, Begin, End
            if hasattr(video, "Labels"):
                raw_labels = video.Labels

                # Helper to process single label object
                def process_label_obj(obj):
                    try:
                        name = obj.Name
                        if name in Config.GESTURE_MAP:
                            gid = Config.GESTURE_MAP[name]
                            # Matlab is 1-based, Python 0-based
                            start = max(0, int(obj.Begin) - 1)
                            end = min(num_frames, int(obj.End))
                            if start < end:
                                labels[start:end] = gid
                    except AttributeError:
                        pass

                if isinstance(raw_labels, np.ndarray):
                    if raw_labels.ndim == 0:
                        process_label_obj(raw_labels.item())
                    else:
                        for l in raw_labels:
                            process_label_obj(l)
                else:
                    process_label_obj(raw_labels)

        except Exception as e:
            print(f"Error processing labels for {sample_info['sample_id']}: {e}")

    return {
        "features": features.astype(np.float32),
        "labels": labels if is_train else None,
        "num_frames": num_frames,
    }


class GestureDataset(Dataset):
    """
    Dataset class for MD-CRCN.
    Handles loading, caching, and serving multi-modal gesture data.
    """

    def __init__(self, split="train", load_cached_data=True, limit=None):
        super().__init__()
        self.split = split
        self.limit = limit

        # Select Metadata File
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.is_train = True
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.is_train = True
        else:
            self.metadata_path = Config.TEST_METADATA_PATH
            self.is_train = False

        # Load Metadata
        self.df = pd.read_csv(self.metadata_path)

        # Sanitize path columns to prevent NaN/float errors
        path_cols = ["rgb_path", "depth_path", "audio_path", "user_path", "data_path"]
        for col in path_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        if self.limit:
            self.df = self.df.head(self.limit)

        # Cache Path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

        # Load or Create Data
        self._load_data(load_cached_data)

    def _load_data(self, load_cached):
        """
        Loads data from cache if available, otherwise processes from scratch.
        Uses a flattened array structure with offsets to avoid pickle.
        """
        if load_cached and os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path)
                self.features_flat = data["features_flat"]
                self.feature_offsets = data["feature_offsets"]
                self.sample_ids = data["sample_ids"]

                if self.is_train:
                    self.labels_flat = data["labels_flat"]
                    self.label_offsets = data["label_offsets"]
                else:
                    self.labels_flat = None
                    self.label_offsets = None

                print(
                    f"Loaded {self.split} data from cache: {len(self.sample_ids)} samples."
                )
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {self.split} data from scratch...")

        features_list = []
        labels_list = []
        sample_ids = []

        # Iterate over metadata
        for _, row in self.df.iterrows():
            result = extract_features(row, is_train=self.is_train)

            features_list.append(result["features"])
            sample_ids.append(row["sample_id"])

            if self.is_train:
                labels_list.append(result["labels"])

        # Flatten and create offsets
        # Features
        self.feature_offsets = np.cumsum([0] + [len(f) for f in features_list])
        self.features_flat = np.concatenate(features_list, axis=0)

        # Labels
        if self.is_train:
            self.label_offsets = np.cumsum([0] + [len(l) for l in labels_list])
            self.labels_flat = np.concatenate(labels_list, axis=0)
        else:
            self.labels_flat = np.array([])
            self.label_offsets = np.array([])

        self.sample_ids = np.array(sample_ids)

        # Save to cache
        np.savez_compressed(
            self.cache_path,
            features_flat=self.features_flat,
            feature_offsets=self.feature_offsets,
            labels_flat=self.labels_flat,
            label_offsets=self.label_offsets,
            sample_ids=self.sample_ids,
        )
        print(f"Saved {self.split} data to cache.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Retrieve Features
        f_start = self.feature_offsets[idx]
        f_end = self.feature_offsets[idx + 1]
        features = self.features_flat[f_start:f_end]

        # Retrieve Labels
        if self.is_train:
            l_start = self.label_offsets[idx]
            l_end = self.label_offsets[idx + 1]
            labels = self.labels_flat[l_start:l_end]
        else:
            # Dummy labels for test set (length matches features)
            labels = np.zeros(len(features), dtype=np.int64)

        return (
            torch.from_numpy(features),
            torch.from_numpy(labels),
            self.sample_ids[idx],
        )


def collate_fn(batch):
    """
    Collates a batch of variable-length sequences.

    Args:
        batch: List of tuples (features, labels, sample_id)

    Returns:
        padded_features: (Batch, Max_Len, Input_Dim)
        padded_labels: (Batch, Max_Len)
        mask: (Batch, Max_Len) - 1 for valid, 0 for pad
        sample_ids: List of sample IDs
    """
    features, labels, sample_ids = zip(*batch)

    # Get lengths
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)

    # Pad Features
    # (Batch, Max_Len, Dim)
    padded_features = torch.zeros(len(features), max_len, features[0].shape[1])

    # Pad Labels
    # (Batch, Max_Len) - Fill with -1 (ignored by loss via mask anyway)
    padded_labels = torch.full((len(labels), max_len), -1, dtype=torch.long)

    # Create Mask
    # (Batch, Max_Len)
    mask = torch.zeros(len(features), max_len, dtype=torch.float32)

    for i, length in enumerate(lengths):
        padded_features[i, :length, :] = features[i]
        padded_labels[i, :length] = labels[i]
        mask[i, :length] = 1.0

    return padded_features, padded_labels, mask, sample_ids
