import os
import random
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library import config, utils


def process_skeleton(mat_path):
    """
    Parses .mat file to extract skeleton joint coordinates.
    Normalizes joints relative to the HipCenter (assumed index 0).
    Returns: (T, 60) numpy array, num_frames
    """
    full_path = os.path.join(config.INPUT_DIR, mat_path)
    try:
        # Load mat file
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, 0

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        # Handle cases where Frames is a single object or empty
        if not isinstance(frames, np.ndarray):
            if frames:  # Single frame
                frames = [frames]
            else:
                frames = []

        actual_frames = min(len(frames), num_frames)
        if actual_frames == 0:
            return None, 0

        # Pre-allocate: (T, 20, 3) - 20 joints, 3 coords
        skeleton_data = np.zeros((actual_frames, 20, 3), dtype=np.float32)

        for i in range(actual_frames):
            frame_obj = frames[i]
            if hasattr(frame_obj, "Skeleton"):
                skel_obj = frame_obj.Skeleton
                # Handle multiple users (take first)
                if isinstance(skel_obj, np.ndarray) and len(skel_obj) > 0:
                    skel_obj = skel_obj[0]

                if hasattr(skel_obj, "WorldPosition"):
                    wp = skel_obj.WorldPosition
                    # Robustly extract X, Y, Z
                    # If wp is an array of objects (joints)
                    if isinstance(wp, np.ndarray) and len(wp) == 20:
                        for j in range(20):
                            joint = wp[j]
                            skeleton_data[i, j, 0] = getattr(joint, "X", 0)
                            skeleton_data[i, j, 1] = getattr(joint, "Y", 0)
                            skeleton_data[i, j, 2] = getattr(joint, "Z", 0)
                    # If wp is a single struct with arrays (rare but possible)
                    elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # Assuming direct array access if structure differs
                        pass

        # Normalize relative to HipCenter (Index 0)
        hip_pos = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_pos

        # Flatten features: (T, 20*3) -> (T, 60)
        skeleton_data = skeleton_data.reshape(actual_frames, -1)

        return skeleton_data, actual_frames

    except Exception as e:
        # Return None to trigger fallback
        return None, 0


def extract_audio_features(audio_path, target_frames):
    """
    Extracts MFCC features from audio file.
    Aligns the number of audio frames to the video frame count.
    Returns: (T, N_MFCC) numpy array
    """
    full_path = os.path.join(config.INPUT_DIR, audio_path)
    if not os.path.exists(full_path):
        return np.zeros((target_frames, config.N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(full_path)

        # Resample if necessary
        if sample_rate != config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sample_rate, config.AUDIO_SR)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract MFCC
        # Physics-based alignment: Hop length matches video frame duration
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SR,
            n_mfcc=config.N_MFCC,
            melkwargs={
                "n_fft": 2048,
                "hop_length": config.HOP_LENGTH,
                "n_mels": 64,
                "center": False,
            },
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Align length
        num_audio_frames = mfcc.shape[0]
        if num_audio_frames < target_frames:
            padding = torch.zeros((target_frames - num_audio_frames, config.N_MFCC))
            mfcc = torch.cat([mfcc, padding], dim=0)
        elif num_audio_frames > target_frames:
            mfcc = mfcc[:target_frames, :]

        return mfcc.numpy()

    except Exception:
        return np.zeros((target_frames, config.N_MFCC), dtype=np.float32)


def get_dense_labels(mat_path, num_frames):
    """
    Parses labels from MAT file and creates a dense frame-wise label array.
    Returns: (T,) numpy array of int
    """
    full_path = os.path.join(config.INPUT_DIR, mat_path)
    labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

    try:
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
        if "Video" in mat and hasattr(mat["Video"], "Labels"):
            raw_labels = mat["Video"].Labels

            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]

            for l in raw_labels:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in config.LABEL_MAP:
                        lid = config.LABEL_MAP[name]
                        # Matlab 1-based inclusive -> Python 0-based exclusive
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))
                        labels[start:end] = lid
    except Exception:
        pass

    return labels


class GestureDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True, limit=None):
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.df = pd.read_csv(metadata_path)

        if limit:
            self.df = self.df.head(limit)

        self.sample_ids = self.df["sample_id"].values
        self.data_paths = self.df["data_path"].values
        self.audio_paths = self.df["audio_path"].values

        # Stats file path
        self.stats_path = os.path.join(config.WORKING_DIR, "stats.npz")
        self.skel_mean = None
        self.skel_std = None
        self.audio_mean = None
        self.audio_std = None

        # Prepare stats
        if self.mode == "train":
            self._compute_global_stats()
        else:
            self._load_global_stats()

    def _compute_global_stats(self):
        """Computes mean and std for normalization over the dataset."""
        if os.path.exists(self.stats_path) and self.load_cached_data:
            self._load_global_stats()
            return

        print("Computing global statistics...")
        skel_sum = 0
        skel_sq_sum = 0
        skel_count = 0
        audio_sum = 0
        audio_sq_sum = 0
        audio_count = 0

        for idx in range(len(self.df)):
            # Load raw without normalization
            skel, audio, _, _ = self._get_item_raw(idx)

            if skel is not None:
                skel_sum += np.sum(skel, axis=0)
                skel_sq_sum += np.sum(skel**2, axis=0)
                skel_count += skel.shape[0]

            if audio is not None:
                audio_sum += np.sum(audio, axis=0)
                audio_sq_sum += np.sum(audio**2, axis=0)
                audio_count += audio.shape[0]

        skel_count = max(skel_count, 1)
        audio_count = max(audio_count, 1)

        self.skel_mean = skel_sum / skel_count
        self.skel_std = np.sqrt((skel_sq_sum / skel_count) - (self.skel_mean**2) + 1e-6)

        self.audio_mean = audio_sum / audio_count
        self.audio_std = np.sqrt(
            (audio_sq_sum / audio_count) - (self.audio_mean**2) + 1e-6
        )

        np.savez(
            self.stats_path,
            skel_mean=self.skel_mean,
            skel_std=self.skel_std,
            audio_mean=self.audio_mean,
            audio_std=self.audio_std,
        )

    def _load_global_stats(self):
        if os.path.exists(self.stats_path):
            data = np.load(self.stats_path)
            self.skel_mean = data["skel_mean"]
            self.skel_std = data["skel_std"]
            self.audio_mean = data["audio_mean"]
            self.audio_std = data["audio_std"]
        else:
            # Identity fallback
            self.skel_mean = 0
            self.skel_std = 1
            self.audio_mean = 0
            self.audio_std = 1

    def _get_item_raw(self, idx):
        """Loads data from cache or computes it."""
        sample_id = self.sample_ids[idx]
        cache_path = os.path.join(config.CACHE_DIR, f"{sample_id}.npz")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return (
                    data["skeleton"],
                    data["audio"],
                    data["labels"],
                    data["length"].item(),
                )
            except:
                pass

        # Compute
        mat_path = self.data_paths[idx]
        audio_path = self.audio_paths[idx]

        skeleton, num_frames = process_skeleton(mat_path)
        if skeleton is None:
            num_frames = 100
            skeleton = np.zeros((num_frames, 60), dtype=np.float32)

        audio = extract_audio_features(audio_path, num_frames)

        if self.mode == "test":
            labels = np.zeros(num_frames, dtype=np.int64)
        else:
            labels = get_dense_labels(mat_path, num_frames)

        # Cache
        if self.load_cached_data:
            np.savez(
                cache_path,
                skeleton=skeleton,
                audio=audio,
                labels=labels,
                length=num_frames,
            )

        return skeleton, audio, labels, num_frames

    def augment(self, skeleton, audio):
        """Applies random augmentations."""
        T, C_skel = skeleton.shape
        _, C_audio = audio.shape

        # 1. Additive Gaussian Noise (Skeleton)
        if random.random() < 0.5:
            noise = torch.randn_like(skeleton) * 0.05
            skeleton = skeleton + noise

        # 2. Random Channel Masking
        if random.random() < 0.5:
            mask_s = torch.randperm(C_skel)[: int(C_skel * 0.1)]
            skeleton[:, mask_s] = 0
            mask_a = torch.randperm(C_audio)[: int(C_audio * 0.1)]
            audio[:, mask_a] = 0

        # 3. Temporal Cutout
        if random.random() < 0.5:
            cut_len = random.randint(5, 15)
            if T > cut_len:
                start = random.randint(0, T - cut_len)
                skeleton[start : start + cut_len, :] = 0
                audio[start : start + cut_len, :] = 0

        return skeleton, audio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        skeleton_np, audio_np, labels_np, _ = self._get_item_raw(idx)

        # Normalize
        if self.skel_mean is not None:
            skeleton_np = (skeleton_np - self.skel_mean) / self.skel_std
        if self.audio_mean is not None:
            audio_np = (audio_np - self.audio_mean) / self.audio_std

        # To Tensor
        skeleton = torch.from_numpy(skeleton_np).float()
        audio = torch.from_numpy(audio_np).float()
        labels = torch.from_numpy(labels_np).long()

        # Augment
        if self.mode == "train":
            skeleton, audio = self.augment(skeleton, audio)

        return skeleton, audio, labels


def collate_fn(batch):
    """
    Pads sequences and creates masks.
    """
    skeletons, audios, labels = zip(*batch)

    # Pad sequences (Batch First)
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=config.BACKGROUND_LABEL
    )

    # Create mask based on lengths
    lengths = torch.tensor([len(s) for s in skeletons])
    # Mask is True for valid positions
    mask = torch.arange(skeletons_padded.size(1))[None, :] < lengths[:, None]

    return skeletons_padded, audios_padded, labels_padded, mask
