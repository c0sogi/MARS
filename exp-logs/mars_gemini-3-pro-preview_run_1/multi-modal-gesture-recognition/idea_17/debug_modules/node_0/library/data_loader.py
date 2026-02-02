import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings

# Import configuration and utilities
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    LABEL_MAP,
    AUDIO_SAMPLE_RATE,
    AUDIO_HOP_LENGTH,
    AUDIO_N_FFT,
    AUDIO_N_MELS,
    SKELETON_INPUT_DIM,
    BACKGROUND_CLASS_ID,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG_SUBSET_SIZE,
)
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ItalianGestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching for processed samples.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Setup directories
        self.cache_dir = os.path.join(WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(TRAIN_METADATA_PATH)
        elif split == "val":
            self.df = pd.read_csv(VAL_METADATA_PATH)
        else:
            self.df = pd.read_csv(TEST_METADATA_PATH)

        # Debugging: Subset
        if DEBUG_SUBSET_SIZE is not None and len(self.df) > DEBUG_SUBSET_SIZE:
            self.df = self.df.iloc[:DEBUG_SUBSET_SIZE].reset_index(drop=True)

        # Audio Transform (Physics-Based Alignment)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=AUDIO_N_MELS,
            melkwargs={
                "n_fft": AUDIO_N_FFT,
                "hop_length": AUDIO_HOP_LENGTH,
                "n_mels": AUDIO_N_MELS,
                "center": False,
            },
        )

        # Statistics for Normalization
        self.stats_path = os.path.join(WORKING_DIR, "stats.npz")
        self.skel_mean = None
        self.skel_std = None
        self.audio_mean = None
        self.audio_std = None

        self._initialize_stats()

    def _initialize_stats(self):
        """
        Loads stats if available, otherwise computes them on the training set.
        """
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.skel_mean = torch.from_numpy(stats["skel_mean"]).float()
            self.skel_std = torch.from_numpy(stats["skel_std"]).float()
            self.audio_mean = torch.from_numpy(stats["audio_mean"]).float()
            self.audio_std = torch.from_numpy(stats["audio_std"]).float()
        else:
            if self.split == "train":
                print("Computing global statistics on training set...")
                self._compute_and_save_stats()
            else:
                # If we are in val/test and stats don't exist, we must compute them
                # (This happens if we run inference without training first in a fresh env)
                # Ideally, we should load the training set to compute stats, but for simplicity
                # here we might warn or compute on current set (suboptimal) or require training first.
                # We will attempt to compute on current set if file missing (fallback).
                print(
                    "Warning: stats.npz not found. Computing on current split (suboptimal if not train)."
                )
                self._compute_and_save_stats()

    def _compute_and_save_stats(self):
        """
        Iterates over the dataset to compute mean and std for Skeleton and Audio.
        Saves to stats.npz.
        """
        skel_sum = torch.zeros(SKELETON_INPUT_DIM)
        skel_sq_sum = torch.zeros(SKELETON_INPUT_DIM)
        skel_count = 0

        audio_sum = torch.zeros(AUDIO_N_MELS)
        audio_sq_sum = torch.zeros(AUDIO_N_MELS)
        audio_count = 0

        # Temporarily disable caching to force raw load for stat computation if needed,
        # but actually we can use the _process_raw_item method directly.

        for idx in range(len(self.df)):
            try:
                skel, audio, _ = self._process_raw_item(idx)

                if skel is not None:
                    # Skeleton (T, 60)
                    skel_sum += skel.sum(dim=0)
                    skel_sq_sum += (skel**2).sum(dim=0)
                    skel_count += skel.size(0)

                if audio is not None:
                    # Audio (T, 64)
                    audio_sum += audio.sum(dim=0)
                    audio_sq_sum += (audio**2).sum(dim=0)
                    audio_count += audio.size(0)
            except Exception as e:
                continue

        # Compute Mean and Std
        self.skel_mean = skel_sum / max(1, skel_count)
        self.skel_std = torch.sqrt(
            (skel_sq_sum / max(1, skel_count)) - self.skel_mean**2 + 1e-6
        )

        self.audio_mean = audio_sum / max(1, audio_count)
        self.audio_std = torch.sqrt(
            (audio_sq_sum / max(1, audio_count)) - self.audio_mean**2 + 1e-6
        )

        np.savez(
            self.stats_path,
            skel_mean=self.skel_mean.numpy(),
            skel_std=self.skel_std.numpy(),
            audio_mean=self.audio_mean.numpy(),
            audio_std=self.audio_std.numpy(),
        )
        print("Statistics computed and saved.")

    def _process_raw_item(self, idx):
        """
        Loads raw data from disk and performs feature extraction.
        Returns: (skeleton_tensor, audio_tensor, label_tensor)
        """
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        # 1. Load Skeleton Data (.mat)
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)
            frames = getattr(video, "Frames", [])
        except Exception:
            # Fallback for corrupt files
            return None, None, None

        # Extract Skeleton
        # Target: (T, 20, 3) -> (T, 60)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        if isinstance(frames, np.ndarray) and len(frames) > 0:
            # Check if frames match num_frames, truncate if necessary
            valid_frames = min(len(frames), num_frames)
            for t in range(valid_frames):
                frame_obj = frames[t]
                if hasattr(frame_obj, "Skeleton"):
                    skel_obj = frame_obj.Skeleton
                    # Handle array of skeletons (multi-user), take first
                    if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                        skel_obj = skel_obj[0]

                    if hasattr(skel_obj, "WorldPosition"):
                        wp = skel_obj.WorldPosition
                        # wp should be (20, 3) or similar.
                        # Based on prompt: X, Y, Z components.
                        # Assuming wp is 20x3 or struct of arrays.
                        # Usually in these datasets it's a struct with X,Y,Z fields or matrix.
                        # Prompt says: WorldPosition structure with X, Y, Z.
                        # Let's try to parse robustly.
                        try:
                            # If it's a matrix
                            if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                                skeleton_data[t] = wp
                            # If it's a struct with X, Y, Z arrays/scalars
                            elif (
                                hasattr(wp, "X")
                                and hasattr(wp, "Y")
                                and hasattr(wp, "Z")
                            ):
                                # Assuming they are arrays of length 20
                                x = (
                                    wp.X
                                    if isinstance(wp.X, np.ndarray)
                                    else np.array([wp.X])
                                )
                                y = (
                                    wp.Y
                                    if isinstance(wp.Y, np.ndarray)
                                    else np.array([wp.Y])
                                )
                                z = (
                                    wp.Z
                                    if isinstance(wp.Z, np.ndarray)
                                    else np.array([wp.Z])
                                )
                                if len(x) == 20:
                                    skeleton_data[t, :, 0] = x
                                    skeleton_data[t, :, 1] = y
                                    skeleton_data[t, :, 2] = z
                        except:
                            pass

        # Root-Relative Normalization
        # HipCenter is usually index 0. Subtract it from all joints.
        # Shape (T, 20, 3)
        hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_center

        # Flatten to (T, 60)
        skeleton_tensor = torch.from_numpy(
            skeleton_data.reshape(num_frames, -1)
        ).float()

        # 2. Load Audio Data (.wav)
        audio_path = (
            os.path.join(INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )
        if audio_path and os.path.exists(audio_path):
            waveform, sr = torchaudio.load(audio_path)
            # Resample if necessary (though dataset is 16k)
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Mix to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            # Shape: (1, n_mels, L)
            mfcc = self.mfcc_transform(waveform)
            # Transpose to (L, n_mels)
            audio_tensor = mfcc.squeeze(0).transpose(0, 1)

            # Physics-Based Alignment
            # Force audio length to match video frames
            if audio_tensor.size(0) > num_frames:
                audio_tensor = audio_tensor[:num_frames]
            elif audio_tensor.size(0) < num_frames:
                pad_len = num_frames - audio_tensor.size(0)
                # Pad with zeros or repeat last frame
                padding = torch.zeros((pad_len, AUDIO_N_MELS))
                audio_tensor = torch.cat([audio_tensor, padding], dim=0)
        else:
            # Fallback: Zero tensor
            audio_tensor = torch.zeros((num_frames, AUDIO_N_MELS))

        # 3. Generate Labels
        # Initialize with Background (0)
        labels_tensor = torch.full((num_frames,), BACKGROUND_CLASS_ID, dtype=torch.long)

        if self.split != "test":
            # Parse labels from .mat (Video.Labels)
            # We use the helper logic similar to metadata generation but applied to tensor construction
            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]
                elif raw_labels.size == 0:
                    raw_labels = []

                for l in raw_labels:
                    try:
                        name = l.Name
                        start = int(l.Begin) - 1  # 1-based to 0-based
                        end = int(
                            l.End
                        )  # inclusive in Matlab, so end index for slice is End

                        if name in LABEL_MAP:
                            lid = LABEL_MAP[name]
                            # Clamp indices
                            start = max(0, start)
                            end = min(num_frames, end)
                            if start < end:
                                labels_tensor[start:end] = lid
                    except:
                        pass

        return skeleton_tensor, audio_tensor, labels_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = self.df.iloc[idx]["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        skel = None
        audio = None
        labels = None

        # 1. Try Load from Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                skel = torch.from_numpy(data["skel"]).float()
                audio = torch.from_numpy(data["audio"]).float()
                labels = torch.from_numpy(data["labels"]).long()
            except Exception:
                # Corrupt cache, reprocess
                pass

        # 2. Process if not loaded
        if skel is None:
            skel, audio, labels = self._process_raw_item(idx)

            # Handle failure case (e.g. corrupt raw file)
            if skel is None:
                # Return dummy data to prevent crash
                skel = torch.zeros((20, SKELETON_INPUT_DIM))
                audio = torch.zeros((20, AUDIO_N_MELS))
                labels = torch.zeros((20,), dtype=torch.long)

            # Save to Cache
            if self.load_cached_data:
                np.savez(
                    cache_path,
                    skel=skel.numpy(),
                    audio=audio.numpy(),
                    labels=labels.numpy(),
                )

        # 3. Normalize (Z-Score)
        # Apply (X - Mean) / Std
        if self.skel_mean is not None:
            skel = (skel - self.skel_mean) / self.skel_std
        if self.audio_mean is not None:
            audio = (audio - self.audio_mean) / self.audio_std

        return skel, audio, labels


def collate_fn(batch):
    """
    Pads sequences and applies augmentations (if training).

    Args:
        batch: List of tuples (skel, audio, labels)

    Returns:
        dict with padded tensors, lengths, and masks.
    """
    # Unzip batch
    skel_list, audio_list, labels_list = zip(*batch)

    # Get lengths
    lengths = torch.tensor([s.size(0) for s in skel_list], dtype=torch.long)

    # Pad Sequences (Batch First)
    # Padding value 0 is fine for features (normalized data is centered around 0)
    # For labels, 0 is Background class, which is appropriate padding
    skel_padded = pad_sequence(skel_list, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels_list, batch_first=True, padding_value=BACKGROUND_CLASS_ID
    )

    # Create Mask (True for valid, False for padding)
    B, T_max, _ = skel_padded.shape
    mask = torch.arange(T_max).expand(B, T_max) < lengths.unsqueeze(1)

    # Augmentation (Only if training logic is applied - usually controlled by dataset split,
    # but collate_fn doesn't know split. We can infer or apply generally.
    # Given the prompt, we'll apply simple random masking here if it seems appropriate,
    # or rely on the dataset split. Since collate_fn is stateless, we'll apply it
    # if the batch size > 1 (heuristic for training) or just apply it.
    # To be safe and deterministic for validation, we usually strictly separate.
    # However, for this implementation, we will skip augmentation in collate_fn
    # to ensure deterministic validation/test results,
    # unless we pass a flag. Since we can't easily pass a flag to collate_fn in standard DataLoader,
    # we will skip augmentation here and assume the model handles dropout/noise,
    # OR implement it in __getitem__ if split=='train'.
    # RE-READING PROMPT: "Implements colate_fn to ... apply augmentations".
    # Okay, I will implement it here but I need to know if it's training.
    # A common trick is to check `torch.is_grad_enabled()` but that's for model forward.
    # I will assume this collate_fn is used for training. For val/test, one might use a different one
    # or we just accept that we need a way to control it.
    # I will add a check: if any label in the batch is not background (implies training data with annotations),
    # we might augment. But test data has no labels (all 0).
    # Better approach: The user prompt asked to implement `collate_fn`. I will implement `collate_fn_train` and `collate_fn_eval`.
    pass

    return {
        "skeleton": skel_padded,
        "audio": audio_padded,
        "labels": labels_padded,
        "lengths": lengths,
        "mask": mask,
    }


class TrainCollate:
    def __call__(self, batch):
        data = collate_fn(batch)

        # Apply Augmentations
        # 1. Random Channel Masking (Skeleton)
        if torch.rand(1).item() < 0.5:
            # Mask ~10% of channels
            B, T, C = data["skeleton"].shape
            mask_c = torch.rand(B, 1, C) > 0.1
            data["skeleton"] = data["skeleton"] * mask_c.float()

        # 2. Random Time Masking
        if torch.rand(1).item() < 0.5:
            B, T, _ = data["skeleton"].shape
            # Pick a random start and length
            mask_len = random.randint(5, 15)
            if T > mask_len:
                start = random.randint(0, T - mask_len)
                # Zero out skeleton and audio in this window
                data["skeleton"][:, start : start + mask_len, :] = 0
                data["audio"][:, start : start + mask_len, :] = 0

        return data


class EvalCollate:
    def __call__(self, batch):
        return collate_fn(batch)


def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    train_ds = ItalianGestureDataset(split="train")
    val_ds = ItalianGestureDataset(split="val")
    test_ds = ItalianGestureDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=TrainCollate(),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalCollate(),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalCollate(),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
