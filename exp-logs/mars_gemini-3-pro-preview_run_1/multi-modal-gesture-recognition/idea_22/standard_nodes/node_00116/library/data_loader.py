import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class GestureDataset(Dataset):
    """
    Dataset class for Multi-Modal Gesture Recognition (RGB-D + Audio).
    Handles loading, preprocessing, caching, normalization, and augmentation.
    """

    def __init__(self, split="train", max_samples=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            max_samples (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Load Metadata
        if split == "train":
            self.metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
            self.augment = True
        elif split == "val":
            self.metadata = pd.read_csv(Config.VAL_METADATA_PATH)
            self.augment = False
        else:
            self.metadata = pd.read_csv(Config.TEST_METADATA_PATH)
            self.augment = False

        if max_samples is not None:
            self.metadata = self.metadata.iloc[:max_samples]
        elif Config.MAX_SAMPLES is not None:
            self.metadata = self.metadata.iloc[: Config.MAX_SAMPLES]

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # MFCC Transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "center": False,
            },
        )

        # Load or Compute Global Stats (Mean/Std)
        self.stats_path = os.path.join(Config.WORKING_DIR, "stats.npz")

        # Validate existing stats file to prevent loading corrupted artifacts
        if os.path.exists(self.stats_path):
            try:
                with np.load(self.stats_path) as data:
                    for k in ["skel_mean", "skel_std", "audio_mean", "audio_std"]:
                        if k not in data:
                            raise KeyError(f"Missing key: {k}")
            except (KeyError, ValueError, OSError):
                print("Corrupted stats file detected. Deleting to regenerate.")
                os.remove(self.stats_path)

        if split == "train":
            self._compute_global_stats()

        # Load stats for normalization
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.skel_mean = torch.from_numpy(stats["skel_mean"]).float()
            self.skel_std = torch.from_numpy(stats["skel_std"]).float()
            self.audio_mean = torch.from_numpy(stats["audio_mean"]).float()
            self.audio_std = torch.from_numpy(stats["audio_std"]).float()
        else:
            # Fallback if stats missing (should not happen if train runs first)
            self.skel_mean = torch.zeros(Config.SKELETON_INPUT_CHANNELS)
            self.skel_std = torch.ones(Config.SKELETON_INPUT_CHANNELS)
            self.audio_mean = torch.zeros(Config.N_MFCC)
            self.audio_std = torch.ones(Config.N_MFCC)

    def _compute_global_stats(self):
        """
        Computes global mean and std for skeleton and audio features over the training set.
        Saves to stats.npz.
        """
        if os.path.exists(self.stats_path) and self.load_cached_data:
            return

        print("Computing global statistics for normalization...")
        skel_sum = torch.zeros(Config.SKELETON_INPUT_CHANNELS)
        skel_sq_sum = torch.zeros(Config.SKELETON_INPUT_CHANNELS)
        skel_count = 0

        audio_sum = torch.zeros(Config.N_MFCC)
        audio_sq_sum = torch.zeros(Config.N_MFCC)
        audio_count = 0

        # Iterate over training data (without augmentation/normalization)
        # We process raw samples here
        for idx in range(len(self.metadata)):
            sample = self._load_sample(idx, normalize=False)
            if sample is None:
                continue

            # Skeleton
            s = sample["skeleton"]  # (T, 60)
            skel_sum += s.sum(dim=0)
            skel_sq_sum += (s**2).sum(dim=0)
            skel_count += s.shape[0]

            # Audio
            a = sample["audio"]  # (T, 13)
            audio_sum += a.sum(dim=0)
            audio_sq_sum += (a**2).sum(dim=0)
            audio_count += a.shape[0]

        # Finalize
        skel_mean = skel_sum / max(skel_count, 1)
        skel_std = torch.sqrt((skel_sq_sum / max(skel_count, 1)) - skel_mean**2 + 1e-6)

        audio_mean = audio_sum / max(audio_count, 1)
        audio_std = torch.sqrt(
            (audio_sq_sum / max(audio_count, 1)) - audio_mean**2 + 1e-6
        )

        np.savez(
            self.stats_path,
            skel_mean=skel_mean.numpy(),
            skel_std=skel_std.numpy(),
            audio_mean=audio_mean.numpy(),
            audio_std=audio_std.numpy(),
        )
        print("Global statistics saved.")

    def _process_skeleton(self, mat_data, num_frames):
        """
        Extracts skeleton data from MAT structure.
        Returns tensor of shape (T, 60) with Root-Relative coordinates.
        """
        try:
            video = mat_data["Video"]
            frames = video.Frames

            # Pre-allocate
            # 20 joints * 3 coords
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            # Check if frames is array
            if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
                # Single frame case or malformed
                if hasattr(frames, "Skeleton"):
                    frames = [frames]
                else:
                    return None

            count = min(len(frames), num_frames)

            for i in range(count):
                frame = frames[i]
                if not hasattr(frame, "Skeleton"):
                    continue

                skel = frame.Skeleton
                # Handle multiple users (take first)
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]

                if hasattr(skel, "WorldPosition"):
                    wp = skel.WorldPosition
                    # WorldPosition might be a struct with X,Y,Z or a matrix
                    # If it's a matrix 20x3
                    if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                        skeleton_data[i] = wp
                    # If it's a struct with X,Y,Z arrays
                    elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # Assuming X, Y, Z are arrays of length 20
                        # Or single values if looped (unlikely for 20 joints)
                        # Let's try to stack
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

            # Root-Relative: Subtract HipCenter (Index 0)
            # skeleton_data shape: (T, 20, 3)
            root = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - root

            # Flatten to (T, 60)
            skeleton_data = skeleton_data.reshape(num_frames, -1)
            return torch.tensor(skeleton_data, dtype=torch.float32)

        except Exception as e:
            # print(f"Skeleton processing error: {e}")
            return None

    def _process_audio(self, audio_path, target_frames):
        """
        Loads audio, mixes to mono, extracts MFCCs, aligns to target_frames.
        Returns tensor of shape (T, 13).
        """
        if not os.path.exists(os.path.join(Config.INPUT_DIR, audio_path)):
            return torch.zeros((target_frames, Config.N_MFCC))

        try:
            waveform, sample_rate = torchaudio.load(
                os.path.join(Config.INPUT_DIR, audio_path)
            )

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Extract MFCC
            # Input: (1, samples), Output: (1, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

            # Align length
            curr_len = mfcc.shape[0]
            if curr_len < target_frames:
                # Pad
                pad_amt = target_frames - curr_len
                mfcc = F.pad(mfcc, (0, 0, 0, pad_amt))
            elif curr_len > target_frames:
                # Trim
                mfcc = mfcc[:target_frames, :]

            return mfcc

        except Exception as e:
            # print(f"Audio processing error: {e}")
            return torch.zeros((target_frames, Config.N_MFCC))

    def _process_labels(self, mat_data, num_frames):
        """
        Constructs frame-wise label array.
        """
        labels = torch.zeros(num_frames, dtype=torch.long)  # Default 0 (background)

        try:
            video = mat_data["Video"]
            if hasattr(video, "Labels"):
                raw_labels = video.Labels

                # Normalize to list
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]

                for l in raw_labels:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in Config.LABEL_MAP:
                            lid = Config.LABEL_MAP[name]
                            # MATLAB is 1-based, Python 0-based
                            start = max(0, int(l.Begin) - 1)
                            end = min(num_frames, int(l.End))
                            labels[start:end] = lid
        except:
            pass

        return labels

    def _load_sample(self, idx, normalize=True):
        """
        Loads a single sample, either from cache or by processing raw files.
        """
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        # 1. Try Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                skeleton = torch.from_numpy(data["skeleton"])
                audio = torch.from_numpy(data["audio"])
                labels = torch.from_numpy(data["labels"])

                if normalize:
                    skeleton = (skeleton - self.skel_mean) / self.skel_std
                    audio = (audio - self.audio_mean) / self.audio_std

                return {
                    "skeleton": skeleton,
                    "audio": audio,
                    "labels": labels,
                    "id": sample_id,
                }
            except:
                pass  # Failed to load, reprocess

        # 2. Process Raw
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            num_frames = getattr(mat["Video"], "NumFrames", 0)
            if num_frames == 0:
                return None

            # Skeleton
            skeleton = self._process_skeleton(mat, num_frames)
            if skeleton is None:
                # Fallback: zeros
                skeleton = torch.zeros((num_frames, Config.SKELETON_INPUT_CHANNELS))

            # Audio
            audio_path = row["audio_path"] if pd.notna(row["audio_path"]) else ""
            audio = self._process_audio(audio_path, num_frames)

            # Labels
            if self.split != "test":
                labels = self._process_labels(mat, num_frames)
            else:
                labels = torch.zeros(num_frames, dtype=torch.long)

            # Save Cache
            np.savez(
                cache_path,
                skeleton=skeleton.numpy(),
                audio=audio.numpy(),
                labels=labels.numpy(),
            )

            if normalize:
                skeleton = (skeleton - self.skel_mean) / self.skel_std
                audio = (audio - self.audio_mean) / self.audio_std

            return {
                "skeleton": skeleton,
                "audio": audio,
                "labels": labels,
                "id": sample_id,
            }

        except Exception as e:
            # print(f"Error loading {sample_id}: {e}")
            return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        if sample is None:
            # Return a dummy sample if loading fails to prevent crash
            # Only happens if raw file is totally broken
            dummy_len = 50
            return {
                "skeleton": torch.zeros((dummy_len, Config.SKELETON_INPUT_CHANNELS)),
                "audio": torch.zeros((dummy_len, Config.N_MFCC)),
                "labels": torch.zeros(dummy_len, dtype=torch.long),
                "length": dummy_len,
                "id": "dummy",
            }

        skeleton = sample["skeleton"]
        audio = sample["audio"]
        labels = sample["labels"]

        # Augmentation (Train only)
        if self.augment:
            # 1. Global Temporal Resampling
            scale = np.random.uniform(*Config.TEMPORAL_RESAMPLE_RANGE)
            new_len = int(skeleton.shape[0] * scale)
            if new_len > 0:
                # Interpolate Features (C, T) -> (C, T_new)
                # Skeleton
                skel_t = skeleton.transpose(0, 1).unsqueeze(0)  # (1, C, T)
                skel_new = F.interpolate(
                    skel_t, size=new_len, mode="linear", align_corners=False
                )
                skeleton = skel_new.squeeze(0).transpose(0, 1)  # (T_new, C)

                # Audio
                audio_t = audio.transpose(0, 1).unsqueeze(0)
                audio_new = F.interpolate(
                    audio_t, size=new_len, mode="linear", align_corners=False
                )
                audio = audio_new.squeeze(0).transpose(0, 1)

                # Labels (Nearest)
                lab_t = labels.float().view(1, 1, -1)
                lab_new = F.interpolate(lab_t, size=new_len, mode="nearest")
                labels = lab_new.view(-1).long()

            # 2. Random Channel Masking
            if np.random.random() < Config.CHANNEL_MASK_PROB:
                # Mask Skeleton channels
                mask_s = torch.rand(Config.SKELETON_INPUT_CHANNELS) > 0.1
                skeleton = skeleton * mask_s.float()
                # Mask Audio channels
                mask_a = torch.rand(Config.N_MFCC) > 0.1
                audio = audio * mask_a.float()

        return {
            "skeleton": skeleton,
            "audio": audio,
            "labels": labels,
            "length": skeleton.shape[0],
            "id": sample["id"],
        }


def collate_fn(batch):
    """
    Pads sequences to the max length in the batch.
    Returns packed tensors.
    """
    # Filter out None/Dummy if necessary (though __getitem__ handles it)
    batch = [b for b in batch if b["length"] > 0]
    if not batch:
        return None

    # Sort by length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    max_len = lengths[0].item()

    # Dimensions
    skel_dim = batch[0]["skeleton"].shape[1]
    audio_dim = batch[0]["audio"].shape[1]
    batch_size = len(batch)

    # Pre-allocate padded tensors
    padded_skeleton = torch.zeros(batch_size, max_len, skel_dim)
    padded_audio = torch.zeros(batch_size, max_len, audio_dim)
    padded_labels = torch.full(
        (batch_size, max_len), Config.BACKGROUND_CLASS_ID, dtype=torch.long
    )

    ids = []

    for i, item in enumerate(batch):
        l = item["length"]
        padded_skeleton[i, :l, :] = item["skeleton"]
        padded_audio[i, :l, :] = item["audio"]
        padded_labels[i, :l] = item["labels"]
        ids.append(item["id"])

    return {
        "skeleton": padded_skeleton,
        "audio": padded_audio,
        "labels": padded_labels,
        "lengths": lengths,
        "ids": ids,
    }


def get_dataloaders():
    """
    Factory function to create dataloaders for train, val, test.
    """
    train_ds = GestureDataset(split="train")
    val_ds = GestureDataset(split="val")
    test_ds = GestureDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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
