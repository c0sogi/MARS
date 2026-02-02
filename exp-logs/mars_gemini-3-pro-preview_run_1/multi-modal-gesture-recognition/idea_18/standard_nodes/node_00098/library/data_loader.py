import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class SkeletonAudioDataset(Dataset):
    def __init__(
        self, metadata_file, mode="train", stats=None, cache_dir=Config.CACHE_DIR
    ):
        """
        Args:
            metadata_file (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            stats (dict): Dictionary containing 'mean' and 'std' for normalization.
            cache_dir (str): Directory to store cached processed samples.
        """
        self.mode = mode
        self.stats = stats
        self.cache_dir = cache_dir
        self.df = pd.read_csv(metadata_file)

        # Filter out samples that don't have required files (just in case)
        self.df = self.df[self.df["data_path"].notna() & self.df["audio_path"].notna()]

        # Audio transform setup
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLING_RATE,
            n_mfcc=Config.AUDIO_N_MELS,  # 13
            melkwargs={
                "n_fft": Config.AUDIO_N_FFT,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "n_mels": Config.AUDIO_MEL_BANDS,  # 64 internal bands
                "center": False,  # To align better with frames
            },
        )

    def __len__(self):
        return len(self.df)

    def _get_cache_path(self, sample_id):
        return os.path.join(self.cache_dir, f"{sample_id}.npz")

    def _load_mat_file(self, mat_path):
        """Parses .mat file for Skeleton and Labels."""
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]

            # 1. Extract Skeleton
            # Cite {solution_lesson_node_00064}: Robust shape-based validation
            frames = video.Frames
            num_frames = len(frames)

            # Pre-allocate
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            for i, frame_obj in enumerate(frames):
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel = frame_obj.Skeleton
                # Handle multiple users (take first)
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]

                if not hasattr(skel, "WorldPosition"):
                    continue

                wp = skel.WorldPosition

                # Robust extraction logic
                # Case A: WorldPosition is a struct with X, Y, Z arrays (common in this dataset)
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    try:
                        # Ensure we are getting arrays of length 20
                        x = np.atleast_1d(wp.X)
                        y = np.atleast_1d(wp.Y)
                        z = np.atleast_1d(wp.Z)

                        if x.shape[0] == Config.NUM_JOINTS:
                            skeleton_data[i, :, 0] = x
                            skeleton_data[i, :, 1] = y
                            skeleton_data[i, :, 2] = z
                    except:
                        pass

                # Case B: WorldPosition is a direct matrix (N, 3) or (3, N)
                elif isinstance(wp, np.ndarray):
                    if wp.shape == (Config.NUM_JOINTS, 3):
                        skeleton_data[i] = wp
                    elif wp.shape == (3, Config.NUM_JOINTS):
                        skeleton_data[i] = wp.T

            # 2. Extract Labels
            # Convert segments to frame-wise labels
            frame_labels = np.zeros(
                num_frames, dtype=np.int64
            )  # Default 0 (background)

            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]

                for l in raw_labels:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        start = int(l.Begin) - 1  # 1-based to 0-based
                        end = int(
                            l.End
                        )  # inclusive in 1-based, so effectively end index in slice

                        if name in Config.LABEL_MAP:
                            lid = Config.LABEL_MAP[name]
                            # Clip to valid range
                            start = max(0, start)
                            end = min(num_frames, end)
                            if end > start:
                                frame_labels[start:end] = lid

            return skeleton_data, frame_labels

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return None, None

    def _load_audio(self, audio_path, target_num_frames):
        """Loads audio and extracts aligned MFCCs."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary (though config says 16k)
            if sample_rate != Config.AUDIO_SAMPLING_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLING_RATE
                )
                waveform = resampler(waveform)

            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Extract MFCC
            # Shape: (1, n_mfcc, time)
            mfcc = self.audio_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

            # Align with video frames
            # Simple truncation or padding
            curr_frames = mfcc.shape[0]
            if curr_frames > target_num_frames:
                mfcc = mfcc[:target_num_frames, :]
            elif curr_frames < target_num_frames:
                pad_amt = target_num_frames - curr_frames
                padding = torch.zeros((pad_amt, mfcc.shape[1]))
                mfcc = torch.cat([mfcc, padding], dim=0)

            return mfcc.numpy()

        except Exception as e:
            # print(f"Error loading audio {audio_path}: {e}")
            return np.zeros(
                (target_num_frames, Config.AUDIO_INPUT_DIM), dtype=np.float32
            )

    def _process_sample(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = self._get_cache_path(sample_id)

        # 1. Try Load Cache
        if os.path.exists(cache_path):
            try:
                # Use context manager to prevent file handle leaks (Cite {debug_lesson_12})
                with np.load(cache_path) as data:
                    skel = data["skeleton"]
                    audio = data["audio"]
                    lbl = data["labels"]

                    # FIX: Validate Cache Consistency (Cite {debug_lesson_1})
                    # Ensure label sequence length matches skeleton sequence length (Frame-wise)
                    if skel.shape[0] != lbl.shape[0]:
                        raise ValueError(
                            f"Cache mismatch: Skeleton len {skel.shape[0]} != Label len {lbl.shape[0]}"
                        )

                    return skel.copy(), audio.copy(), lbl.copy()
            except Exception:
                pass  # Corrupt or stale cache, recompute

        # 2. Compute from Scratch
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        skeleton, labels = self._load_mat_file(mat_path)

        if skeleton is None:
            # Fallback for broken files: create dummy data matching duration from metadata if possible
            # or just return a minimal valid sequence
            num_frames = row["num_frames"] if pd.notna(row["num_frames"]) else 100
            skeleton = np.zeros(
                (int(num_frames), Config.NUM_JOINTS, 3), dtype=np.float32
            )
            labels = np.zeros(int(num_frames), dtype=np.int64)

        # Normalize Skeleton: Root Relative
        # Joint 0 is HipCenter
        root = skeleton[:, 0:1, :]  # (T, 1, 3)
        skeleton = skeleton - root

        # Flatten Skeleton: (T, J*3)
        skeleton = skeleton.reshape(skeleton.shape[0], -1)

        # Load Audio
        audio = self._load_audio(audio_path, skeleton.shape[0])

        # Save Cache
        np.savez_compressed(cache_path, skeleton=skeleton, audio=audio, labels=labels)

        return skeleton, audio, labels

    def __getitem__(self, idx):
        skeleton, audio, labels = self._process_sample(idx)

        # Augmentation removed (Cite {solution_lesson_node_00090})

        # Normalization
        if self.stats:
            # Skeleton Z-score
            skeleton = (skeleton - self.stats["skel_mean"]) / (
                self.stats["skel_std"] + 1e-6
            )
            # Audio Z-score
            audio = (audio - self.stats["audio_mean"]) / (
                self.stats["audio_std"] + 1e-6
            )

        return (
            torch.tensor(skeleton, dtype=torch.float32),
            torch.tensor(audio, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch):
    """
    Pads sequences and sorts by length descending for packed_sequence.
    Returns: padded_skel, padded_audio, padded_labels, lengths, original_indices
    """
    # batch is list of tuples (skel, audio, label)

    # Sort by length descending
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    skeletons = [x[0] for x in batch]
    audios = [x[1] for x in batch]
    labels = [x[2] for x in batch]
    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad
    padded_skel = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0.0
    )
    padded_audio = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0.0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return padded_skel, padded_audio, padded_labels, lengths


def compute_global_stats(dataset_instance, save_path):
    """Computes mean/std of dataset for normalization."""
    print("Computing global stats...")
    skel_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    skel_sq_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    audio_sum = np.zeros(Config.AUDIO_INPUT_DIM)
    audio_sq_sum = np.zeros(Config.AUDIO_INPUT_DIM)
    total_frames = 0

    # Iterate without augmentation
    original_mode = dataset_instance.mode
    dataset_instance.mode = "test"  # Disable augmentation

    for i in range(len(dataset_instance)):
        skel, audio, _ = dataset_instance._process_sample(i)

        skel_sum += np.sum(skel, axis=0)
        skel_sq_sum += np.sum(skel**2, axis=0)

        audio_sum += np.sum(audio, axis=0)
        audio_sq_sum += np.sum(audio**2, axis=0)

        total_frames += skel.shape[0]

    dataset_instance.mode = original_mode

    skel_mean = skel_sum / total_frames
    skel_std = np.sqrt((skel_sq_sum / total_frames) - skel_mean**2)

    audio_mean = audio_sum / total_frames
    audio_std = np.sqrt((audio_sq_sum / total_frames) - audio_mean**2)

    stats = {
        "skel_mean": skel_mean,
        "skel_std": skel_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }
    np.savez(save_path, **stats)
    return stats


def get_dataloaders():
    """
    Factory function to create dataloaders.
    Handles stats computation and caching.
    """
    stats_path = os.path.join(Config.WORK_DIR, "stats.npz")

    # Initialize Train Dataset first to compute stats
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    train_ds = SkeletonAudioDataset(train_csv, mode="train")

    if os.path.exists(stats_path):
        loaded = np.load(stats_path)
        stats = {k: loaded[k] for k in loaded.files}
    else:
        stats = compute_global_stats(train_ds, stats_path)

    # Assign stats to train dataset
    train_ds.stats = stats

    # Val Dataset
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    val_ds = SkeletonAudioDataset(val_csv, mode="val", stats=stats)

    # Test Dataset
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")
    test_ds = SkeletonAudioDataset(test_csv, mode="test", stats=stats)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Sequential for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
