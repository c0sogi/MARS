import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed

# ==========================================
# Augmentation Transforms
# ==========================================


class RandomChannelMask:
    """
    Randomly zeros out feature channels (e.g., specific joints or MFCC coeffs).
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        # x: (Time, Dim)
        if np.random.rand() < self.p:
            return x

        # Create a mask of shape (Dim,)
        dim = x.shape[1]
        # Mask approx 10% of channels
        num_mask = max(1, int(dim * 0.1))
        mask_indices = np.random.choice(dim, num_mask, replace=False)

        x_aug = x.copy()
        x_aug[:, mask_indices] = 0
        return x_aug


class TemporalCutout:
    """
    Randomly zeros out a contiguous chunk of frames.
    """

    def __init__(self, max_len=15, p=0.5):
        self.max_len = max_len
        self.p = p

    def __call__(self, x):
        # x: (Time, Dim)
        if np.random.rand() > self.p:
            return x

        time_steps = x.shape[0]
        if time_steps <= self.max_len:
            return x

        # Select random start
        cut_len = np.random.randint(5, self.max_len + 1)
        start = np.random.randint(0, time_steps - cut_len)

        x_aug = x.copy()
        x_aug[start : start + cut_len, :] = 0
        return x_aug


# ==========================================
# Data Processing Helpers
# ==========================================


def load_audio_features(audio_path, target_frames):
    """
    Loads audio and extracts MFCCs aligned with video frames.
    """
    try:
        y, sr = sf.read(audio_path)
        # If multi-channel, average to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample if necessary (though dataset is consistent 16k)
        if sr != Config.AUDIO_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=Config.AUDIO_SAMPLE_RATE)

        # Extract MFCC
        # hop_length aligned to video fps
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.AUDIO_HOP_LENGTH,
        )
        # Result is (n_mfcc, time) -> Transpose to (time, n_mfcc)
        mfcc = mfcc.T

        # Align length with video frames
        current_frames = mfcc.shape[0]
        if current_frames < target_frames:
            # Pad
            pad_width = target_frames - current_frames
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
        elif current_frames > target_frames:
            # Trim
            mfcc = mfcc[:target_frames, :]

        return mfcc.astype(np.float32)

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)


def load_skeleton_features(mat_path):
    """
    Parses MAT file to extract and normalize skeleton joints.
    Returns: (Time, 60) numpy array, num_frames
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, 0

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        if num_frames == 0 or len(frames) == 0:
            return None, 0

        # Extract joints
        # Expecting 20 joints * 3 coords
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            if i >= num_frames:
                break

            # Robustly extract Skeleton struct
            skel = None
            if hasattr(frame, "Skeleton"):
                skel_obj = frame.Skeleton
                # Handle array of skeletons (multi-user) -> take first
                if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                    skel = skel_obj[0]
                elif hasattr(skel_obj, "WorldPosition"):
                    skel = skel_obj

            if skel is not None and hasattr(skel, "WorldPosition"):
                wp = skel.WorldPosition
                # wp might be a struct with x,y,z or an array
                # Based on prompt: "X value represents...", likely a struct-like access or array
                # We try to extract into a (20, 3) array
                # Often in these MAT files, WorldPosition is a (20, 3) or (3, 20) array directly
                # or a struct with fields.

                # Let's try to interpret wp as an array first
                if isinstance(wp, np.ndarray):
                    if wp.shape == (20, 3):
                        skeleton_data[i] = wp
                    elif wp.shape == (3, 20):
                        skeleton_data[i] = wp.T
                    else:
                        pass  # Unexpected shape
                # If it's a struct with X, Y, Z fields which are arrays
                elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # Assuming X, Y, Z are arrays of 20 elements
                    try:
                        x = np.atleast_1d(wp.X)
                        y = np.atleast_1d(wp.Y)
                        z = np.atleast_1d(wp.Z)
                        if len(x) == 20:
                            skeleton_data[i, :, 0] = x
                            skeleton_data[i, :, 1] = y
                            skeleton_data[i, :, 2] = z
                    except:
                        pass

        # Relative Coordinates: Subtract HipCenter (Index 0)
        # (Time, 20, 3)
        hip_center = skeleton_data[:, 0:1, :]  # Keep dims for broadcasting
        skeleton_data = skeleton_data - hip_center

        # Flatten to (Time, 60)
        skeleton_flat = skeleton_data.reshape(num_frames, -1)

        return skeleton_flat, num_frames

    except Exception as e:
        return None, 0


def create_label_sequence(labels_str, num_frames):
    """
    Converts label string "1,2,3" and MAT metadata into frame-wise labels.
    """
    # Initialize with background (0)
    label_seq = np.zeros(num_frames, dtype=np.int64)

    # We need the ground truth segmentation (start/end frames)
    # This info is inside the MAT file (Labels struct), not just the CSV string.
    # The CSV string just gives the order.
    # However, to be efficient, we should have extracted this during metadata generation or load it now.
    # Since we are loading the MAT file anyway in load_skeleton_features,
    # we should extract label timings there.
    # BUT, the function signature above separated them.
    # Let's refactor slightly to load labels from MAT if available.
    return label_seq


def load_sample_raw(row):
    """
    Loads raw data for a single sample row from DataFrame.
    Returns: skeleton (T, 60), audio (T, 13), labels (T,)
    """
    mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
    audio_path = (
        os.path.join(Config.INPUT_DIR, row["audio_path"])
        if pd.notna(row["audio_path"])
        else None
    )

    # 1. Load Skeleton & Frame Count
    skeleton, num_frames = load_skeleton_features(mat_path)
    if skeleton is None:
        # Fallback for broken files
        return None, None, None

    # 2. Load Audio
    if audio_path and os.path.exists(audio_path):
        audio = load_audio_features(audio_path, num_frames)
    else:
        audio = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)

    # 3. Construct Labels
    # We need to re-open MAT to get start/end times for labels
    label_seq = np.zeros(num_frames, dtype=np.int64)
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" in mat and hasattr(mat["Video"], "Labels"):
            labels_data = mat["Video"].Labels
            if not isinstance(labels_data, np.ndarray):
                labels_data = [labels_data]
            elif labels_data.size == 1:
                labels_data = [labels_data.item()]

            for l in labels_data:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in Config.LABEL_MAP:
                        lid = Config.LABEL_MAP[name]
                        # MATLAB is 1-based, inclusive
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))
                        label_seq[start:end] = lid
    except:
        pass  # Keep as zeros (background)

    return skeleton, audio, label_seq


# ==========================================
# Statistics & Normalization
# ==========================================


def compute_global_stats(df, cache_dir):
    """
    Computes global mean and std for skeleton and audio features.
    """
    stats_path = os.path.join(cache_dir, "stats.npz")
    if os.path.exists(stats_path):
        return np.load(stats_path)

    print("Computing global statistics...")

    # Accumulators
    skel_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    skel_sq_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    skel_count = 0

    audio_sum = np.zeros(Config.N_MFCC)
    audio_sq_sum = np.zeros(Config.N_MFCC)
    audio_count = 0

    # Iterate over a subset to save time (e.g., 200 samples)
    sample_df = df.sample(min(len(df), 200), random_state=Config.SEED)

    for _, row in sample_df.iterrows():
        skel, audio, _ = load_sample_raw(row)
        if skel is None:
            continue

        skel_sum += np.sum(skel, axis=0)
        skel_sq_sum += np.sum(skel**2, axis=0)
        skel_count += skel.shape[0]

        audio_sum += np.sum(audio, axis=0)
        audio_sq_sum += np.sum(audio**2, axis=0)
        audio_count += audio.shape[0]

    # Compute Mean and Std
    skel_mean = skel_sum / max(1, skel_count)
    skel_std = np.sqrt((skel_sq_sum / max(1, skel_count)) - skel_mean**2 + 1e-6)

    audio_mean = audio_sum / max(1, audio_count)
    audio_std = np.sqrt((audio_sq_sum / max(1, audio_count)) - audio_mean**2 + 1e-6)

    stats = {
        "skel_mean": skel_mean.astype(np.float32),
        "skel_std": skel_std.astype(np.float32),
        "audio_mean": audio_mean.astype(np.float32),
        "audio_std": audio_std.astype(np.float32),
    }

    np.savez(stats_path, **stats)
    print("Global statistics computed and saved.")
    return np.load(stats_path)


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, csv_path, stats, is_train=False, debug=False):
        self.df = pd.read_csv(csv_path)
        if debug:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE)

        self.stats = stats
        self.is_train = is_train

        # Augmentations
        self.aug_channel = RandomChannelMask(p=0.2)
        self.aug_time = TemporalCutout(max_len=15, p=0.3)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        # 1. Load or Create Cache
        if Config.LOAD_CACHED_DATA and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                skel = data["skel"]
                audio = data["audio"]
                labels = data["labels"]
            except:
                # Corrupt cache, recompute
                skel, audio, labels = self._process_and_cache(row, cache_path)
        else:
            skel, audio, labels = self._process_and_cache(row, cache_path)

        # Handle broken samples
        if skel is None:
            # Return dummy data
            skel = np.zeros((10, Config.SKELETON_INPUT_DIM), dtype=np.float32)
            audio = np.zeros((10, Config.N_MFCC), dtype=np.float32)
            labels = np.zeros(10, dtype=np.int64)

        # 2. Augmentation (Training Only)
        if self.is_train:
            skel = self.aug_channel(skel)
            skel = self.aug_time(skel)
            audio = self.aug_channel(audio)  # Apply to audio too? Yes, robustness.

        # 3. Convert to Tensor
        skel_t = torch.tensor(skel, dtype=torch.float32)
        audio_t = torch.tensor(audio, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        return skel_t, audio_t, labels_t

    def _process_and_cache(self, row, cache_path):
        # Load raw
        skel, audio, labels = load_sample_raw(row)

        if skel is None:
            return None, None, None

        # Normalize
        skel = (skel - self.stats["skel_mean"]) / self.stats["skel_std"]
        audio = (audio - self.stats["audio_mean"]) / self.stats["audio_std"]

        # Save
        np.savez(cache_path, skel=skel, audio=audio, labels=labels)

        return skel, audio, labels


def collate_fn(batch):
    """
    Pads sequences to the max length in the batch.
    """
    # Filter out None samples
    batch = [b for b in batch if b[0].shape[0] > 0]
    if not batch:
        return None, None, None, None

    skels, audios, labels = zip(*batch)

    # Pad
    # batch_first=True -> (Batch, Time, Dim)
    skels_padded = pad_sequence(skels, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    # Create mask/lengths if needed, but PyTorch RNNs usually handle this via pack_padded_sequence
    # or we can just use the lengths.
    lengths = torch.tensor([s.size(0) for s in skels], dtype=torch.long)

    return skels_padded, audios_padded, labels_padded, lengths


# ==========================================
# Public Interface
# ==========================================


def get_data_loaders(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG):
    set_seed(Config.SEED)

    # 1. Compute Stats from Train Data
    # We read the train CSV just to compute stats
    train_df = pd.read_csv(Config.TRAIN_CSV)
    stats = compute_global_stats(train_df, Config.CACHE_DIR)

    # 2. Datasets
    train_ds = GestureDataset(Config.TRAIN_CSV, stats, is_train=True, debug=debug)
    val_ds = GestureDataset(Config.VAL_CSV, stats, is_train=False, debug=debug)
    test_ds = GestureDataset(Config.TEST_CSV, stats, is_train=False, debug=debug)

    # 3. Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
