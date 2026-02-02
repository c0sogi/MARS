import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    SEED,
    FRAME_RATE,
    AUDIO_SAMPLE_RATE,
    AUDIO_HOP_LENGTH,
    AUDIO_N_FFT,
    AUDIO_N_MELS,
    SKELETON_JOINTS,
    SKELETON_CHANNELS,
    LABEL_MAP,
    TEMPORAL_SCALE_MIN,
    TEMPORAL_SCALE_MAX,
    WORK_DIR,
)
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(SEED)


class GestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug_subset_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, tries to load from cache.
            debug_subset_size (int, optional): If set, limits dataset size for debugging.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Load Metadata
        metadata_file = os.path.join(METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.metadata = pd.read_csv(metadata_file)

        # Filter out samples with missing essential files
        self.metadata = self.metadata[
            self.metadata["data_path"].notna() & self.metadata["audio_path"].notna()
        ].reset_index(drop=True)

        if debug_subset_size is not None:
            self.metadata = self.metadata.iloc[:debug_subset_size]

        # Initialize Audio Transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=AUDIO_N_MELS,
            melkwargs={
                "n_fft": AUDIO_N_FFT,
                "n_mels": AUDIO_N_MELS,
                "hop_length": AUDIO_HOP_LENGTH,
                "center": False,  # Align with frames
            },
        )

        # Global Statistics (Mean/Std)
        self.stats_path = os.path.join(WORK_DIR, "stats.npz")
        self.skel_mean = None
        self.skel_std = None
        self.audio_mean = None
        self.audio_std = None

        self._initialize_stats()

    def _initialize_stats(self):
        """Loads or computes global statistics for normalization."""
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.skel_mean = torch.from_numpy(stats["skel_mean"]).float()
            self.skel_std = torch.from_numpy(stats["skel_std"]).float()
            self.audio_mean = torch.from_numpy(stats["audio_mean"]).float()
            self.audio_std = torch.from_numpy(stats["audio_std"]).float()
        else:
            if self.split == "train":
                print("Computing global statistics on training set...")
                self._compute_stats()
            else:
                # If validating/testing before training, use identity (zeros/ones)
                # This is a fallback; typically training runs first.
                print("Warning: Stats file not found. Using identity normalization.")
                self.skel_mean = torch.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
                self.skel_std = torch.ones(SKELETON_JOINTS * SKELETON_CHANNELS)
                self.audio_mean = torch.zeros(AUDIO_N_MELS)
                self.audio_std = torch.ones(AUDIO_N_MELS)

    def _compute_stats(self):
        """Computes mean and std over the dataset and saves to disk."""
        skel_sum = torch.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
        skel_sq_sum = torch.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
        skel_count = 0

        audio_sum = torch.zeros(AUDIO_N_MELS)
        audio_sq_sum = torch.zeros(AUDIO_N_MELS)
        audio_count = 0

        # Temporarily disable caching/augmentation for stat computation
        prev_cache_setting = self.load_cached_data
        self.load_cached_data = True  # Use cache if available to speed up

        for i in range(len(self)):
            # Load raw data (no augmentation, no norm yet)
            skel, audio, _ = self._load_sample(i)

            # Update Skeleton Stats
            skel_sum += skel.sum(dim=0)
            skel_sq_sum += (skel**2).sum(dim=0)
            skel_count += skel.size(0)

            # Update Audio Stats
            audio_sum += audio.sum(dim=0)
            audio_sq_sum += (audio**2).sum(dim=0)
            audio_count += audio.size(0)

        self.load_cached_data = prev_cache_setting

        # Finalize
        self.skel_mean = skel_sum / skel_count
        self.skel_std = torch.sqrt(
            (skel_sq_sum / skel_count) - (self.skel_mean**2) + 1e-6
        )

        self.audio_mean = audio_sum / audio_count
        self.audio_std = torch.sqrt(
            (audio_sq_sum / audio_count) - (self.audio_mean**2) + 1e-6
        )

        # Save
        np.savez(
            self.stats_path,
            skel_mean=self.skel_mean.numpy(),
            skel_std=self.skel_std.numpy(),
            audio_mean=self.audio_mean.numpy(),
            audio_std=self.audio_std.numpy(),
        )
        print(f"Stats saved to {self.stats_path}")

    def _load_raw_data(self, idx):
        """Reads .mat and .wav files and extracts features."""
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]

        # 1. Load Skeleton (.mat)
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]

            # Extract Frames
            if not hasattr(video_struct, "Frames"):
                raise ValueError("Missing Frames in MAT file")

            frames = video_struct.Frames
            num_frames = len(frames)

            # Extract WorldPosition (N, 20, 3)
            # Assuming standard structure: Frames[i].Skeleton.WorldPosition
            # Handling potential single-frame or structure variations
            joints_list = []
            for f in frames:
                if hasattr(f, "Skeleton") and hasattr(f.Skeleton, "WorldPosition"):
                    # Handle multiple skeletons (take first) or single
                    skel = f.Skeleton
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition might be (20, 3) or similar
                        pos = skel.WorldPosition
                        # Ensure shape (20, 3)
                        if pos.shape != (SKELETON_JOINTS, 3):
                            # Fallback or reshape if data is weird, but usually it's correct
                            pos = np.zeros((SKELETON_JOINTS, 3))
                        joints_list.append(pos)
                    else:
                        joints_list.append(np.zeros((SKELETON_JOINTS, 3)))
                else:
                    joints_list.append(np.zeros((SKELETON_JOINTS, 3)))

            skeleton_data = np.array(joints_list)  # (T, 20, 3)

            # Root-Relative Coordinates
            # HipCenter is index 0
            root = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - root

            # Flatten: (T, 60)
            skeleton_tensor = (
                torch.from_numpy(skeleton_data).float().view(num_frames, -1)
            )

            # Extract Labels (Training Only)
            labels_tensor = torch.zeros(num_frames, dtype=torch.long)
            if self.split == "train" and hasattr(video_struct, "Labels"):
                labels_raw = video_struct.Labels
                if not isinstance(labels_raw, np.ndarray):
                    labels_raw = [labels_raw]
                elif labels_raw.size == 1:
                    labels_raw = [labels_raw.item()]

                for l in labels_raw:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in LABEL_MAP:
                            lid = LABEL_MAP[name]
                            # MATLAB is 1-based, Python 0-based
                            start = max(0, int(l.Begin) - 1)
                            end = min(num_frames, int(l.End))
                            labels_tensor[start:end] = lid

        except Exception as e:
            # Fallback for corrupted files
            # print(f"Error loading MAT {sample_id}: {e}")
            skeleton_tensor = torch.zeros((10, SKELETON_JOINTS * SKELETON_CHANNELS))
            labels_tensor = torch.zeros(10, dtype=torch.long)

        # 2. Load Audio (.wav)
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample if needed
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Mix to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            # Output: (1, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)
            # Remove channel dim and Transpose to (Time, n_mfcc)
            audio_tensor = mfcc.squeeze(0).transpose(0, 1)  # (T_audio, 64)

        except Exception as e:
            # Fallback
            # print(f"Error loading Audio {sample_id}: {e}")
            audio_tensor = torch.zeros((10, AUDIO_N_MELS))

        # 3. Alignment
        # Trim to min length to synchronize
        min_len = min(
            skeleton_tensor.size(0), audio_tensor.size(0), labels_tensor.size(0)
        )
        if min_len == 0:
            # Handle empty case
            min_len = 1
            skeleton_tensor = torch.zeros((1, SKELETON_JOINTS * SKELETON_CHANNELS))
            audio_tensor = torch.zeros((1, AUDIO_N_MELS))
            labels_tensor = torch.zeros(1, dtype=torch.long)
        else:
            skeleton_tensor = skeleton_tensor[:min_len]
            audio_tensor = audio_tensor[:min_len]
            labels_tensor = labels_tensor[:min_len]

        return skeleton_tensor, audio_tensor, labels_tensor

    def _load_sample(self, idx):
        """Handles caching logic."""
        sample_id = self.metadata.iloc[idx]["sample_id"]
        cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                skel = torch.from_numpy(data["skel"])
                audio = torch.from_numpy(data["audio"])
                labels = torch.from_numpy(data["labels"])
                return skel, audio, labels
            except Exception:
                pass  # Fallback to compute

        # Compute
        skel, audio, labels = self._load_raw_data(idx)

        # Save to cache
        np.savez_compressed(
            cache_path, skel=skel.numpy(), audio=audio.numpy(), labels=labels.numpy()
        )

        return skel, audio, labels

    def apply_augmentation(self, skel, audio, labels):
        """Applies Global Speed Jittering via interpolation."""
        # Random scale factor
        scale = np.random.uniform(TEMPORAL_SCALE_MIN, TEMPORAL_SCALE_MAX)

        # Prepare for interpolation: (Batch, Channels, Time)
        # Add batch dimension
        skel_in = skel.unsqueeze(0).transpose(1, 2)  # (1, 60, T)
        audio_in = audio.unsqueeze(0).transpose(1, 2)  # (1, 64, T)
        labels_in = labels.float().view(1, 1, -1)  # (1, 1, T)

        # Interpolate
        skel_out = F.interpolate(
            skel_in, scale_factor=scale, mode="linear", align_corners=False
        )
        audio_out = F.interpolate(
            audio_in, scale_factor=scale, mode="linear", align_corners=False
        )
        labels_out = F.interpolate(labels_in, scale_factor=scale, mode="nearest")

        # Reshape back
        skel = skel_out.squeeze(0).transpose(0, 1)  # (T', 60)
        audio = audio_out.squeeze(0).transpose(0, 1)  # (T', 64)
        labels = labels_out.view(-1).long()  # (T',)

        return skel, audio, labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        skel, audio, labels = self._load_sample(idx)

        # Augmentation (Train only)
        if self.split == "train":
            skel, audio, labels = self.apply_augmentation(skel, audio, labels)

        # Normalization
        # (X - Mean) / Std
        if self.skel_mean is not None:
            skel = (skel - self.skel_mean) / self.skel_std
        if self.audio_mean is not None:
            audio = (audio - self.audio_mean) / self.audio_std

        return skel, audio, labels


def collate_fn(batch):
    """
    Collates a batch of (skel, audio, labels).
    Pads sequences and sorts by length descending (required for pack_padded_sequence).
    """
    # Unzip batch
    skels, audios, labels = zip(*batch)

    # Get lengths
    lengths = torch.tensor([s.size(0) for s in skels])

    # Sort by length descending
    lengths, sort_idx = lengths.sort(descending=True)

    skels = [skels[i] for i in sort_idx]
    audios = [audios[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]

    # Pad Sequences
    # batch_first=True -> (Batch, Time, Channels)
    skels_padded = pad_sequence(skels, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    return skels_padded, audios_padded, labels_padded, lengths
