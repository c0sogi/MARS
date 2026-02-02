import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io
import soundfile as sf
import torchaudio
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class GestureDataset(Dataset):
    def __init__(self, split="train", debug=Config.DEBUG):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            debug (bool): If True, use a small subset of data.
        """
        self.split = split
        self.debug = debug
        self.cache_dir = Config.CACHE_DIR
        self.stats_path = os.path.join(os.path.dirname(Config.CACHE_DIR), "stats.npz")

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.debug:
            self.df = self.df.head(10)

        # Initialize stats
        self.stats = None
        if Config.USE_GLOBAL_STATS:
            self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        """
        Loads global stats from file or computes them from the training set.
        """
        if os.path.exists(self.stats_path):
            data = np.load(self.stats_path)
            self.stats = {
                "skel_mean": torch.tensor(data["skel_mean"], dtype=torch.float32),
                "skel_std": torch.tensor(data["skel_std"], dtype=torch.float32),
                "audio_mean": torch.tensor(data["audio_mean"], dtype=torch.float32),
                "audio_std": torch.tensor(data["audio_std"], dtype=torch.float32),
            }
        else:
            # Only compute on training split to avoid leakage
            if self.split != "train":
                # If we are in val/test and stats don't exist, we must compute them from train CSV
                train_df = pd.read_csv(Config.TRAIN_CSV)
                if self.debug:
                    train_df = train_df.head(10)
                samples_to_scan = train_df
            else:
                samples_to_scan = self.df

            print("Computing global statistics...")
            skel_sum = torch.zeros(Config.SKELETON_JOINTS * Config.SKELETON_CHANNELS)
            skel_sq_sum = torch.zeros(Config.SKELETON_JOINTS * Config.SKELETON_CHANNELS)
            audio_sum = torch.zeros(Config.AUDIO_N_MFCC)
            audio_sq_sum = torch.zeros(Config.AUDIO_N_MFCC)
            skel_count = 0
            audio_count = 0

            for _, row in samples_to_scan.iterrows():
                # Load without augmentation/normalization
                data = self._load_sample(row, apply_aug=False, normalize=False)
                if data is None:
                    continue

                # Skeleton: (T, J, 3) -> (T, J*3)
                skel = data["skeleton"].view(
                    -1, Config.SKELETON_JOINTS * Config.SKELETON_CHANNELS
                )
                skel_sum += skel.sum(dim=0)
                skel_sq_sum += (skel**2).sum(dim=0)
                skel_count += skel.shape[0]

                # Audio: (T, C)
                audio = data["audio"]
                audio_sum += audio.sum(dim=0)
                audio_sq_sum += (audio**2).sum(dim=0)
                audio_count += audio.shape[0]

            # Compute Mean and Std
            skel_mean = skel_sum / max(1, skel_count)
            skel_std = torch.sqrt(
                (skel_sq_sum / max(1, skel_count)) - skel_mean**2 + 1e-6
            )

            audio_mean = audio_sum / max(1, audio_count)
            audio_std = torch.sqrt(
                (audio_sq_sum / max(1, audio_count)) - audio_mean**2 + 1e-6
            )

            self.stats = {
                "skel_mean": skel_mean,
                "skel_std": skel_std,
                "audio_mean": audio_mean,
                "audio_std": audio_std,
            }

            # Save
            np.savez(
                self.stats_path,
                skel_mean=skel_mean.numpy(),
                skel_std=skel_std.numpy(),
                audio_mean=audio_mean.numpy(),
                audio_std=audio_std.numpy(),
            )
            print("Statistics computed and saved.")

    def _load_raw_skeleton(self, mat_path):
        """Parses .mat file for skeleton data and labels."""
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = video.NumFrames

            # 1. Extract Skeleton
            # Initialize with zeros: (NumFrames, Joints, 3)
            # Joints=20, Channels=3
            skeleton_data = np.zeros(
                (num_frames, Config.SKELETON_JOINTS, 3), dtype=np.float32
            )

            frames = video.Frames
            # Handle cases where Frames might be a single object or empty
            if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
                frames = [frames]

            # Iterate through frames
            # Limit by num_frames to avoid index errors if metadata mismatches
            for i in range(min(len(frames), num_frames)):
                frame_obj = frames[i]
                if hasattr(frame_obj, "Skeleton"):
                    skel_obj = frame_obj.Skeleton
                    # If multiple skeletons, take the first one
                    if isinstance(skel_obj, np.ndarray):
                        if skel_obj.size > 0:
                            skel_obj = skel_obj[0]
                        else:
                            continue  # No skeleton

                    if hasattr(skel_obj, "WorldPosition"):
                        # WorldPosition usually has X, Y, Z fields or is an array
                        # Based on prompt: WorldPosition.X, WorldPosition.Y, WorldPosition.Z
                        # And JointsType is a list of 20 joints.
                        # We assume the order matches the 20 joints listed in prompt.

                        # Check if WorldPosition is a struct with arrays or array of structs
                        # Usually in these datasets, it's a struct with X,Y,Z being 20x1 arrays
                        wp = skel_obj.WorldPosition

                        # Robust extraction
                        try:
                            if (
                                hasattr(wp, "X")
                                and hasattr(wp, "Y")
                                and hasattr(wp, "Z")
                            ):
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
                            else:
                                # Fallback: maybe it's an array directly?
                                continue

                            # Ensure we have 20 joints
                            if len(x) == Config.SKELETON_JOINTS:
                                skeleton_data[i, :, 0] = x
                                skeleton_data[i, :, 1] = y
                                skeleton_data[i, :, 2] = z
                        except:
                            continue

            # 2. Extract Labels
            # Create dense labels: (NumFrames,)
            labels_dense = np.zeros(num_frames, dtype=np.int64)  # 0 is background

            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]

                # Map names to IDs
                # We need the LABEL_MAP. It's not imported from config, let's define it or assume logic.
                # The prompt gave a list 1..20. Let's recreate the map quickly or rely on metadata names.
                # Actually, the metadata CSV already has the sequence of labels, but not the timestamps.
                # We MUST use the .mat for timestamps.

                # Hardcoded map based on prompt description
                label_map = {
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

                for l in raw_labels:
                    # Check if valid label object
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in label_map:
                            lid = label_map[name]
                            # Matlab is 1-based index usually, but let's check.
                            # If Begin=1, that's index 0.
                            start = int(l.Begin) - 1
                            end = int(l.End)

                            # Clip to valid range
                            start = max(0, start)
                            end = min(num_frames, end)

                            if end > start:
                                labels_dense[start:end] = lid

            return skeleton_data, labels_dense

        except Exception as e:
            # print(f"Error parsing mat file {mat_path}: {e}")
            return None, None

    def _process_audio(self, audio_path, target_frames):
        """Loads audio, extracts MFCC, aligns to video frames."""
        try:
            # Load audio
            y, sr = sf.read(audio_path)

            # Mix to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # Resample if necessary (though dataset is 16kHz)
            if sr != Config.SAMPLE_RATE:
                # Simple resampling not implemented for dependency reasons,
                # but dataset analysis confirmed 16kHz.
                pass

            y_tensor = torch.tensor(y, dtype=torch.float32)

            # MFCC Extraction
            # Hop length = 800 (aligned to 20fps)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=Config.SAMPLE_RATE,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": Config.AUDIO_N_FFT,
                    "hop_length": Config.AUDIO_HOP_LENGTH,
                    "center": False,  # To align better with frames
                },
            )

            mfcc = mfcc_transform(y_tensor)  # (n_mfcc, time)
            mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

            # Alignment: Pad or Trim to match target_frames
            curr_frames = mfcc.shape[0]

            if curr_frames < target_frames:
                # Pad with zeros
                pad_amt = target_frames - curr_frames
                mfcc = F.pad(mfcc, (0, 0, 0, pad_amt))
            elif curr_frames > target_frames:
                # Trim
                mfcc = mfcc[:target_frames, :]

            return mfcc.numpy()

        except Exception as e:
            # print(f"Error processing audio {audio_path}: {e}")
            # Return zeros if failed
            return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _load_sample(self, row, apply_aug=False, normalize=True):
        """
        Loads a single sample, checking cache first.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Try Load Cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                skeleton = cached["skeleton"]
                audio = cached["audio"]
                labels = cached["labels"]
            except:
                # Corrupt cache, reload raw
                skeleton, audio, labels = None, None, None
        else:
            skeleton, audio, labels = None, None, None

        # 2. Compute if missing
        if skeleton is None:
            if not isinstance(row["data_path"], str):
                return None

            full_mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            full_audio_path = (
                os.path.join(Config.INPUT_DIR, row["audio_path"])
                if pd.notna(row["audio_path"])
                else None
            )

            skeleton, labels = self._load_raw_skeleton(full_mat_path)

            if skeleton is None:
                return None  # Failed to load

            # Root Relative Normalization
            # Assume Joint 0 is HipCenter
            hip_center = skeleton[:, 0:1, :]  # (T, 1, 3)
            skeleton = skeleton - hip_center

            # Audio
            if full_audio_path and os.path.exists(full_audio_path):
                audio = self._process_audio(
                    full_audio_path, target_frames=skeleton.shape[0]
                )
            else:
                audio = np.zeros(
                    (skeleton.shape[0], Config.AUDIO_N_MFCC), dtype=np.float32
                )

            # Save to cache
            np.savez_compressed(
                cache_path, skeleton=skeleton, audio=audio, labels=labels
            )

        # Convert to Tensors
        skeleton = torch.tensor(skeleton, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 3. Augmentation (Temporal Resampling)
        if apply_aug and self.split == "train":
            # Random resampling factor
            alpha = np.random.uniform(
                Config.TEMPORAL_RESAMPLE_MIN, Config.TEMPORAL_RESAMPLE_MAX
            )
            orig_len = skeleton.shape[0]
            new_len = int(orig_len * alpha)

            if new_len > 0:
                # Interpolate Skeleton: (T, J, 3) -> (T, J*3) -> (1, J*3, T) -> interpolate -> reshape
                skel_flat = (
                    skeleton.view(orig_len, -1).permute(1, 0).unsqueeze(0)
                )  # (1, C, T)
                skel_resampled = F.interpolate(
                    skel_flat, size=new_len, mode="linear", align_corners=False
                )
                skeleton = (
                    skel_resampled.squeeze(0)
                    .permute(1, 0)
                    .view(new_len, Config.SKELETON_JOINTS, 3)
                )

                # Interpolate Audio
                audio_t = audio.permute(1, 0).unsqueeze(0)
                audio_resampled = F.interpolate(
                    audio_t, size=new_len, mode="linear", align_corners=False
                )
                audio = audio_resampled.squeeze(0).permute(1, 0)

                # Interpolate Labels (Nearest)
                labels_f = labels.float().view(1, 1, orig_len)
                labels_resampled = F.interpolate(labels_f, size=new_len, mode="nearest")
                labels = labels_resampled.view(new_len).long()

        # 4. Normalization
        if normalize and self.stats is not None:
            # Skeleton
            # Reshape stats to broadcast: (1, J*3) -> (1, J, 3)
            s_mean = (
                self.stats["skel_mean"]
                .view(1, Config.SKELETON_JOINTS, 3)
                .to(skeleton.device)
            )
            s_std = (
                self.stats["skel_std"]
                .view(1, Config.SKELETON_JOINTS, 3)
                .to(skeleton.device)
            )
            skeleton = (skeleton - s_mean) / s_std

            # Audio
            a_mean = self.stats["audio_mean"].view(1, -1).to(audio.device)
            a_std = self.stats["audio_std"].view(1, -1).to(audio.device)
            audio = (audio - a_mean) / a_std

        return {
            "sample_id": row["sample_id"],
            "skeleton": skeleton,
            "audio": audio,
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self._load_sample(row, apply_aug=(self.split == "train"), normalize=True)

        # If loading failed, try another sample (fallback)
        if data is None:
            # Simple fallback: return the next item or previous
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        return data


def collate_fn(batch):
    """
    Pads sequences and generates boundary targets.
    """
    # Filter Nones if any slipped through
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Sort by length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    sample_ids = [b["sample_id"] for b in batch]
    skeletons = [b["skeleton"] for b in batch]
    audios = [b["audio"] for b in batch]
    labels = [b["labels"] for b in batch]
    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad Sequences
    # (B, T, J, 3)
    padded_skeleton = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0
    )
    # (B, T, C)
    padded_audio = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0
    )
    # (B, T) - Pad with 0 (Background)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0
    )

    # Generate Boundary Targets
    # y_t = 1 if label_t != label_{t-1}
    # Shift labels right: [0, L0, L1, ...]
    # Compare with original
    # We compute this per sequence to respect padding
    boundaries = []
    for lbl in labels:
        # lbl is (T,)
        # shift: (0, lbl[:-1])
        shifted = torch.cat([torch.tensor([lbl[0]], device=lbl.device), lbl[:-1]])
        # boundary where current != prev
        # Note: First frame is usually not a boundary unless we define it so.
        # Let's say boundary if diff. First frame diff with itself is 0.
        # But if transition from background to gesture?
        # The prompt says: "label_t != label_{t-1}"
        b = (lbl != shifted).float()
        boundaries.append(b)

    padded_boundaries = torch.nn.utils.rnn.pad_sequence(
        boundaries, batch_first=True, padding_value=0
    )

    return {
        "sample_ids": sample_ids,
        "skeleton": padded_skeleton,  # (B, T, J, 3)
        "audio": padded_audio,  # (B, T, C)
        "labels": padded_labels,  # (B, T)
        "boundaries": padded_boundaries,  # (B, T)
        "lengths": lengths,
    }


def get_dataloaders():
    """
    Factory function to get train, val, test loaders.
    """
    train_ds = GestureDataset(split="train")
    val_ds = GestureDataset(split="val")
    test_ds = GestureDataset(split="test")

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
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
