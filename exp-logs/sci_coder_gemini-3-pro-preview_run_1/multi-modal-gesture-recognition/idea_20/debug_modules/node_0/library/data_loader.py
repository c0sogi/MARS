import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import librosa
import warnings
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config

# Suppress warnings from librosa/scipy
warnings.filterwarnings("ignore")


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the Multimodal Gesture Recognition task.
    Handles loading of Skeleton (MAT) and Audio (WAV) data, preprocessing,
    caching, and augmentation.
    """

    def __init__(self, split="train", load_cached_data=True, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
            transform (bool): Whether to apply augmentations (usually True for train).
        """
        self.split = split
        self.load_cached_data = load_cached_data
        self.augment = transform if transform is not None else (split == "train")

        # Load Metadata
        if split == "train":
            self.metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif split == "val":
            self.metadata = pd.read_csv(Config.VAL_METADATA_PATH)
        elif split == "test":
            self.metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Cache Directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Joint Names ordered as per prompt description
        self.joint_names = [
            "HipCenter",
            "Spine",
            "ShoulderCenter",
            "Head",
            "ShoulderLeft",
            "ElbowLeft",
            "WristLeft",
            "HandLeft",
            "ShoulderRight",
            "ElbowRight",
            "WristRight",
            "HandRight",
            "HipLeft",
            "KneeLeft",
            "AnkleLeft",
            "FootLeft",
            "HipRight",
            "KneeRight",
            "AnkleRight",
            "FootRight",
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]

        # Cache Path
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Try Load Cache
        data = None
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                skeleton = data["skeleton"]
                audio = data["audio"]
                labels = data["labels"]
            except Exception:
                # Corrupt cache, re-process
                data = None

        # 2. Process from Scratch if needed
        if data is None:
            skeleton, audio, labels = self._process_sample(row)
            # Save to cache
            np.savez_compressed(
                cache_path, skeleton=skeleton, audio=audio, labels=labels
            )

        # 3. Convert to Torch Tensors
        skeleton = torch.tensor(skeleton, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 4. Augmentation (Train only)
        if self.augment:
            skeleton, audio, labels = self._augment(skeleton, audio, labels)

        return {
            "skeleton": skeleton,
            "audio": audio,
            "labels": labels,
            "sample_id": sample_id,
        }

    def _process_sample(self, row):
        """
        Reads raw files and extracts features.
        Returns:
            skeleton (np.ndarray): (T, 60)
            audio (np.ndarray): (T, 13)
            labels (np.ndarray): (T,)
        """
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(Config.INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )

        # --- A. Load MAT (Skeleton & Labels) ---
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]
            num_frames = video_struct.NumFrames

            # 1. Extract Skeleton
            # Initialize with zeros: (NumFrames, NumJoints * 3)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS * 3), dtype=np.float32
            )

            if hasattr(video_struct, "Frames"):
                frames = video_struct.Frames
                # Handle cases where Frames might be a single object or array
                if not isinstance(frames, np.ndarray):
                    frames = np.array([frames])

                # Iterate frames
                # Note: frames array might be shorter than NumFrames or have gaps,
                # but usually it matches. We iterate up to min length.
                iter_len = min(len(frames), num_frames)

                for f_idx in range(iter_len):
                    frame_obj = frames[f_idx]
                    if hasattr(frame_obj, "Skeleton"):
                        skel_obj = frame_obj.Skeleton
                        # If multiple skeletons, take first
                        if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                            skel_obj = skel_obj[0]

                        if hasattr(skel_obj, "WorldPosition"):
                            # WorldPosition should be 20 structs or arrays
                            # We need to extract X, Y, Z for each joint in order
                            # The prompt implies WorldPosition structure inside Skeleton.
                            # However, MATLAB struct arrays can be tricky.
                            # Usually skel_obj.WorldPosition is an array of size 20.
                            wp = skel_obj.WorldPosition

                            current_frame_joints = []
                            # If wp is an array of objects/structs
                            if (
                                isinstance(wp, np.ndarray)
                                and wp.size == Config.NUM_JOINTS
                            ):
                                for j_idx in range(Config.NUM_JOINTS):
                                    joint = wp[j_idx]
                                    current_frame_joints.append(
                                        [joint.X, joint.Y, joint.Z]
                                    )

                            current_frame_joints = np.array(
                                current_frame_joints
                            )  # (20, 3)

                            # Root Relative Normalization
                            # HipCenter is index 0
                            if current_frame_joints.shape == (20, 3):
                                root = current_frame_joints[0].copy()
                                current_frame_joints -= root
                                skeleton_data[f_idx] = current_frame_joints.flatten()

            # 2. Extract Labels (Frame-wise)
            # Default to background
            labels_seq = np.full(num_frames, Config.BACKGROUND_CLASS_ID, dtype=np.int64)

            # Only process labels if they exist (Train/Val)
            if hasattr(video_struct, "Labels"):
                lbls = video_struct.Labels
                if not isinstance(lbls, np.ndarray):
                    lbls = np.array([lbls]) if lbls is not None else np.array([])

                if lbls.size > 0:
                    # If it's a single element (0-d array), wrap it
                    if lbls.ndim == 0:
                        lbls = np.array([lbls.item()])

                    for l in lbls:
                        try:
                            if (
                                hasattr(l, "Name")
                                and hasattr(l, "Begin")
                                and hasattr(l, "End")
                            ):
                                name = l.Name
                                if name in Config.LABEL_MAP:
                                    lid = Config.LABEL_MAP[name]
                                    # MATLAB is 1-based, Python 0-based
                                    start = max(0, int(l.Begin) - 1)
                                    end = min(num_frames, int(l.End))
                                    labels_seq[start:end] = lid
                        except:
                            continue

        except Exception as e:
            # Fallback for broken MAT files
            # print(f"Error processing MAT {mat_path}: {e}")
            num_frames = 100  # Default dummy
            skeleton_data = np.zeros((num_frames, 60), dtype=np.float32)
            labels_seq = np.zeros(num_frames, dtype=np.int64)

        # --- B. Process Audio ---
        audio_features = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)
        if audio_path and os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
                # Physics-based alignment
                # n_fft=2048, hop_length=800
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=Config.N_MFCC,
                    n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH,
                )
                # mfcc shape: (n_mfcc, T_audio) -> transpose to (T_audio, n_mfcc)
                mfcc = mfcc.T

                # Align with video frames
                if mfcc.shape[0] >= num_frames:
                    audio_features = mfcc[:num_frames]
                else:
                    # Pad with zeros
                    pad_len = num_frames - mfcc.shape[0]
                    audio_features = np.vstack(
                        [mfcc, np.zeros((pad_len, Config.N_MFCC))]
                    )

            except Exception as e:
                # print(f"Error processing Audio {audio_path}: {e}")
                pass

        # Normalize Skeleton (Simple Z-score per sample to handle global scale diffs)
        # Avoid division by zero
        if np.std(skeleton_data) > 1e-5:
            skeleton_data = (skeleton_data - np.mean(skeleton_data)) / np.std(
                skeleton_data
            )

        # Normalize Audio
        if np.std(audio_features) > 1e-5:
            audio_features = (audio_features - np.mean(audio_features)) / np.std(
                audio_features
            )

        return (
            skeleton_data.astype(np.float32),
            audio_features.astype(np.float32),
            labels_seq.astype(np.int64),
        )

    def _augment(self, skeleton, audio, labels):
        """
        Applies Temporal Resampling and Channel Masking.
        """
        # 1. Temporal Resampling (Global Uniform Scaling)
        # Alpha ~ U(0.8, 1.2)
        alpha = np.random.uniform(0.8, 1.2)
        orig_len = skeleton.shape[0]
        new_len = int(orig_len * alpha)

        if new_len > 0 and new_len != orig_len:
            # Interpolate Features
            # Skeleton: (T, 60) -> (1, 60, T) for grid_sample or interpolate
            skel_t = skeleton.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
            audio_t = audio.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)

            skel_interp = torch.nn.functional.interpolate(
                skel_t, size=new_len, mode="linear", align_corners=False
            )
            audio_interp = torch.nn.functional.interpolate(
                audio_t, size=new_len, mode="linear", align_corners=False
            )

            skeleton = skel_interp.squeeze(0).permute(1, 0)  # (T_new, C)
            audio = audio_interp.squeeze(0).permute(1, 0)  # (T_new, C)

            # Interpolate Labels (Nearest Neighbor)
            labels_t = labels.unsqueeze(0).unsqueeze(0).float()  # (1, 1, T)
            labels_interp = torch.nn.functional.interpolate(
                labels_t, size=new_len, mode="nearest"
            )
            labels = labels_interp.squeeze().long()

        # 2. Channel Masking
        # Mask ~10% of channels
        if np.random.random() < 0.5:
            # Skeleton Masking
            mask_idx = torch.randperm(skeleton.shape[1])[: int(skeleton.shape[1] * 0.1)]
            skeleton[:, mask_idx] = 0

            # Audio Masking
            mask_idx_a = torch.randperm(audio.shape[1])[: int(audio.shape[1] * 0.1)]
            audio[:, mask_idx_a] = 0

        return skeleton, audio, labels


def collate_fn(batch):
    """
    Pads sequences to the max length in the batch.
    """
    # Filter out None samples if any
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    skeletons = [b["skeleton"] for b in batch]
    audios = [b["audio"] for b in batch]
    labels = [b["labels"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]

    # Lengths
    lengths = torch.tensor([s.size(0) for s in skeletons], dtype=torch.long)

    # Pad
    # batch_first=True -> (Batch, MaxLen, FeatDim)
    padded_skeletons = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Pad labels with Background Class ID
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return {
        "skeleton": padded_skeletons,
        "audio": padded_audios,
        "labels": padded_labels,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }
