import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io
import scipy.signal
import torchaudio
import warnings

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    LABEL_MAP,
    BACKGROUND_CLASS_ID,
    HYPERPARAMS,
    TRAIN_METADATA_PATH,
)
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, split="train", load_cached_data=True, cache_dir=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'val', or 'test'. Controls augmentation and stats usage.
            load_cached_data (bool): If True, attempts to load/save processed samples from cache.
            cache_dir (str): Directory to store cached .npz files.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Setup Cache Directory
        if cache_dir is None:
            self.cache_dir = os.path.join(WORKING_DIR, "cache")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Metadata
        self.metadata = pd.read_csv(metadata_path)

        # Filter out samples that don't have essential files (Color/Depth/Data)
        # We rely on 'data_path' (mat file) and 'audio_path' primarily for this architecture
        self.metadata = self.metadata[
            self.metadata["data_path"].notna() & self.metadata["audio_path"].notna()
        ].reset_index(drop=True)

        # Audio Transformation (MFCC)
        # We initialize it here but might use functional calls if needed,
        # but class-based is better for consistency.
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=HYPERPARAMS["audio_sample_rate"],
            n_mfcc=HYPERPARAMS["n_mfcc"],
            melkwargs={
                "n_fft": HYPERPARAMS["n_fft"],
                "hop_length": HYPERPARAMS["hop_length"],
                "center": False,
            },
        )

        # Load or Compute Global Stats for Normalization
        self.stats_path = os.path.join(WORKING_DIR, "stats.npz")
        self.stats = self._get_global_stats()

    def _get_global_stats(self):
        """
        Loads stats from file or computes them from the training set.
        """
        if os.path.exists(self.stats_path) and self.load_cached_data:
            with np.load(self.stats_path) as data:
                return {k: data[k] for k in data.files}

        # If not found or forced refresh, compute from training data
        # We need to identify training samples. We assume TRAIN_METADATA_PATH points to them.
        print("Computing global normalization statistics from training set...")
        train_df = pd.read_csv(TRAIN_METADATA_PATH)
        train_df = train_df[
            train_df["data_path"].notna() & train_df["audio_path"].notna()
        ]

        skeleton_accumulator = []
        audio_accumulator = []

        # We process a subset to save time if dataset is huge, but here it's small (~300)
        for _, row in train_df.iterrows():
            skel, aud, _ = self._process_raw_files(row)
            if skel is not None and aud is not None:
                # Randomly sample frames to avoid memory explosion
                if len(skel) > 50:
                    indices = np.linspace(0, len(skel) - 1, 50).astype(int)
                    skeleton_accumulator.append(skel[indices])
                    audio_accumulator.append(aud[indices])
                else:
                    skeleton_accumulator.append(skel)
                    audio_accumulator.append(aud)

        all_skel = np.concatenate(skeleton_accumulator, axis=0)
        all_audio = np.concatenate(audio_accumulator, axis=0)

        stats = {
            "skel_mean": np.mean(all_skel, axis=0),
            "skel_std": np.std(all_skel, axis=0) + 1e-6,  # Avoid div by zero
            "audio_mean": np.mean(all_audio, axis=0),
            "audio_std": np.std(all_audio, axis=0) + 1e-6,
        }

        np.savez(self.stats_path, **stats)
        print("Global statistics computed and saved.")
        return stats

    def _process_raw_files(self, row):
        """
        Reads raw .mat and .wav files and extracts features.
        Returns:
            skeleton (np.array): (T, Joints*3)
            audio (np.array): (T, n_mfcc)
            labels (np.array): (T,) dense frame-wise labels
        """
        sample_id = row["sample_id"]
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)

            # Extract Labels info first to build dense target
            labels_struct = getattr(video, "Labels", [])
            dense_labels = np.zeros(num_frames, dtype=int)  # Default to Background (0)

            if not isinstance(labels_struct, np.ndarray):
                labels_struct = [labels_struct] if labels_struct else []
            elif labels_struct.size == 1:
                labels_struct = [labels_struct.item()]

            for l in labels_struct:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in LABEL_MAP:
                        lid = LABEL_MAP[name]
                        # Matlab 1-based indexing
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))
                        dense_labels[start:end] = lid

            # Extract Skeleton
            # Video.Frames is array of structs
            frames = getattr(video, "Frames", [])
            if len(frames) != num_frames:
                # Fallback or mismatch handling
                num_frames = len(frames)
                dense_labels = dense_labels[:num_frames]

            skeleton_data = np.zeros((num_frames, 20 * 3), dtype=np.float32)

            for i, frame in enumerate(frames):
                if hasattr(frame, "Skeleton"):
                    skel = frame.Skeleton
                    # Handle multiple users: take first
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition usually (20, 3) or similar
                        # We need to flatten.
                        # Assuming joint order is consistent.
                        # Prompt says: HipCenter is usually first.
                        # We need to parse the struct.
                        # Since scipy.io with struct_as_record=False gives objects,
                        # we can't easily iterate unless it's an array.
                        # However, usually WorldPosition is a (20,3) matrix in these datasets.
                        # Let's check type.
                        wp = skel.WorldPosition
                        if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                            # Root Relative: Subtract HipCenter (Index 0)
                            root = wp[0, :]
                            rel_wp = wp - root
                            skeleton_data[i] = rel_wp.flatten()
                        else:
                            # Fallback if structure differs (e.g. separate fields)
                            pass

        except Exception as e:
            # print(f"Error processing MAT {sample_id}: {e}")
            return None, None, None

        # 2. Load Audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Resample if needed
            if sample_rate != HYPERPARAMS["audio_sample_rate"]:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, HYPERPARAMS["audio_sample_rate"]
                )
                waveform = resampler(waveform)

            # Extract MFCC
            # (Channel, n_mfcc, Time)
            mfcc = self.mfcc_transform(waveform)
            mfcc = mfcc.mean(dim=0)  # Average over channels if stereo
            mfcc = mfcc.transpose(0, 1).numpy()  # (Time, n_mfcc)

        except Exception as e:
            # print(f"Error processing Audio {sample_id}: {e}")
            return None, None, None

        # 3. Alignment
        # Truncate or Pad Audio to match Skeleton frames
        # We trust Skeleton frame count as the ground truth for sequence length
        target_len = len(skeleton_data)
        curr_audio_len = len(mfcc)

        if curr_audio_len > target_len:
            mfcc = mfcc[:target_len]
        elif curr_audio_len < target_len:
            pad_len = target_len - curr_audio_len
            # Pad with zeros
            padding = np.zeros((pad_len, mfcc.shape[1]), dtype=mfcc.dtype)
            mfcc = np.concatenate([mfcc, padding], axis=0)

        return skeleton_data, mfcc, dense_labels

    def _load_sample(self, index):
        row = self.metadata.iloc[index]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # Try Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data["skeleton"], data["audio"], data["labels"]
            except:
                pass  # Corrupt cache, recompute

        # Compute
        skel, aud, lbl = self._process_raw_files(row)

        # Save Cache
        if skel is not None and self.load_cached_data:
            np.savez(cache_path, skeleton=skel, audio=aud, labels=lbl)

        return skel, aud, lbl

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        skeleton, audio, labels = self._load_sample(idx)

        if skeleton is None:
            # Fallback for broken sample: return zero-length or next sample
            # Simple hack: return next sample (wrapping around)
            return self.__getitem__((idx + 1) % len(self))

        # Convert to Tensor
        skeleton = torch.tensor(skeleton, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Normalization
        skel_mean = torch.tensor(self.stats["skel_mean"], dtype=torch.float32)
        skel_std = torch.tensor(self.stats["skel_std"], dtype=torch.float32)
        audio_mean = torch.tensor(self.stats["audio_mean"], dtype=torch.float32)
        audio_std = torch.tensor(self.stats["audio_std"], dtype=torch.float32)

        skeleton = (skeleton - skel_mean) / skel_std
        audio = (audio - audio_mean) / audio_std

        # Augmentation (Train only)
        if self.split == "train":
            # 1. Global Temporal Resampling
            alpha = np.random.uniform(
                HYPERPARAMS["resample_alpha_min"], HYPERPARAMS["resample_alpha_max"]
            )
            new_len = int(skeleton.shape[0] * alpha)
            if new_len > 0:
                # Interpolate Skeleton (T, D) -> (1, D, T) for interpolate
                skel_t = skeleton.permute(1, 0).unsqueeze(0)
                skel_t = F.interpolate(
                    skel_t, size=new_len, mode="linear", align_corners=False
                )
                skeleton = skel_t.squeeze(0).permute(1, 0)

                # Interpolate Audio
                aud_t = audio.permute(1, 0).unsqueeze(0)
                aud_t = F.interpolate(
                    aud_t, size=new_len, mode="linear", align_corners=False
                )
                audio = aud_t.squeeze(0).permute(1, 0)

                # Interpolate Labels (Nearest)
                lbl_t = labels.float().view(1, 1, -1)
                lbl_t = F.interpolate(lbl_t, size=new_len, mode="nearest")
                labels = lbl_t.view(-1).long()

            # 2. Channel Masking
            if np.random.random() < HYPERPARAMS["mask_channel_prob"]:
                # Mask Skeleton Channels
                mask_idx = torch.randperm(skeleton.shape[1])[
                    : int(skeleton.shape[1] * 0.1)
                ]
                skeleton[:, mask_idx] = 0
                # Mask Audio Channels
                mask_idx_a = torch.randperm(audio.shape[1])[: int(audio.shape[1] * 0.1)]
                audio[:, mask_idx_a] = 0

        return skeleton, audio, labels


def collate_fn(batch):
    """
    Pads sequences to the max length in the batch.
    """
    skeletons, audios, labels = zip(*batch)

    # Get lengths
    lengths = torch.tensor([s.size(0) for s in skeletons])

    # Pad
    # pad_sequence expects (T, *, *)
    padded_skeletons = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0.0
    )
    padded_audios = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0.0
    )

    # Pad labels with BACKGROUND_CLASS_ID (0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=BACKGROUND_CLASS_ID
    )

    return padded_skeletons, padded_audios, lengths, padded_labels
