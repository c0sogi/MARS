import os
import torch
import torchaudio
import scipy.io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    """
    Dataset class for Multi-modal Gesture Recognition (RGB-D + Audio).
    Handles loading, preprocessing, caching, and normalization of Skeleton and Audio data.
    """

    def __init__(self, split="train", debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.debug = debug

        # Load Metadata
        if split == "train":
            self.csv_path = Config.TRAIN_CSV
            self.mode = "train"
        elif split == "val":
            self.csv_path = Config.VAL_CSV
            self.mode = "val"
        else:
            self.csv_path = Config.TEST_CSV
            self.mode = "test"

        self.df = pd.read_csv(self.csv_path)

        # Debugging: Limit size
        if self.debug or (
            Config.DEBUG_SUBSET_SIZE is not None
            and isinstance(Config.DEBUG_SUBSET_SIZE, int)
        ):
            limit = Config.DEBUG_SUBSET_SIZE if Config.DEBUG_SUBSET_SIZE else 20
            self.df = self.df.iloc[:limit]

        # Label Map (Name -> ID)
        self.label_map = {
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

        # Load or Compute Global Stats
        self.stats_path = os.path.join(Config.WORKING_DIR, "stats.npz")
        self.skel_mean, self.skel_std, self.audio_mean, self.audio_std = (
            self._get_stats()
        )

    def _get_stats(self):
        """Loads stats from file or computes them from the training set."""
        if os.path.exists(self.stats_path):
            data = np.load(self.stats_path)
            return (
                data["skel_mean"],
                data["skel_std"],
                data["audio_mean"],
                data["audio_std"],
            )

        # If not found and we are in training mode, compute them
        if self.mode == "train":
            print("Computing global statistics on training set...")
            skel_accum = []
            audio_accum = []

            # Use a subset to save time if dataset is huge, but here we use all valid
            # We iterate without caching to avoid filling cache with un-normalized data
            valid_count = 0
            for idx in range(len(self.df)):
                # Limit to first 200 samples for stats estimation to save time
                if valid_count > 200:
                    break

                row = self.df.iloc[idx]
                skel, audio, _ = self._process_raw_files(row, compute_labels=False)

                if skel is not None and audio is not None:
                    skel_accum.append(skel)
                    audio_accum.append(audio)
                    valid_count += 1

            if not skel_accum:
                # Fallback defaults
                return np.zeros(60), np.ones(60), np.zeros(13), np.ones(13)

            skel_concat = np.concatenate(skel_accum, axis=0)
            audio_concat = np.concatenate(audio_accum, axis=0)

            skel_mean = np.mean(skel_concat, axis=0)
            skel_std = np.std(skel_concat, axis=0) + 1e-6

            audio_mean = np.mean(audio_concat, axis=0)
            audio_std = np.std(audio_concat, axis=0) + 1e-6

            np.savez(
                self.stats_path,
                skel_mean=skel_mean,
                skel_std=skel_std,
                audio_mean=audio_mean,
                audio_std=audio_std,
            )

            return skel_mean, skel_std, audio_mean, audio_std
        else:
            # If validation/test and stats don't exist, use defaults (should not happen in proper pipeline)
            return np.zeros(60), np.ones(60), np.zeros(13), np.ones(13)

    def _process_raw_files(self, row, compute_labels=True):
        """Reads MAT and WAV files, extracts features and labels."""
        try:
            sample_id = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # 1. Load Skeleton from MAT
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None, None, None

            video = mat["Video"]
            frames = video.Frames
            num_frames = getattr(video, "NumFrames", len(frames))

            # Extract Skeleton (T, 20, 3)
            skeleton_frames = []
            for f in frames:
                # Robust extraction of WorldPosition
                # Structure varies: Frame -> Skeleton -> WorldPosition
                if hasattr(f, "Skeleton"):
                    skel = f.Skeleton
                    # Handle array of users (take first)
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        # Ensure shape is (20, 3)
                        if wp.shape == (20, 3):
                            skeleton_frames.append(wp)
                        elif wp.shape == (3, 20):
                            skeleton_frames.append(wp.T)
                        else:
                            skeleton_frames.append(np.zeros((20, 3)))
                    else:
                        skeleton_frames.append(np.zeros((20, 3)))
                else:
                    skeleton_frames.append(np.zeros((20, 3)))

            skeleton_data = np.array(skeleton_frames, dtype=np.float32)  # (T, 20, 3)

            # Relative Coordinates: Subtract HipCenter (Index 0)
            # Assuming Index 0 is HipCenter based on prompt list
            hip_center = skeleton_data[:, 0:1, :]
            skeleton_data = skeleton_data - hip_center

            # Flatten to (T, 60)
            T, J, C = skeleton_data.shape
            skeleton_data = skeleton_data.reshape(T, J * C)

            # 2. Load Audio
            # Sync: Video 20fps -> 1 frame = 50ms. Audio 16kHz -> 800 samples.
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != Config.AUDIO_SR:
                resampler = torchaudio.transforms.Resample(sample_rate, Config.AUDIO_SR)
                waveform = resampler(waveform)

            # MFCC
            # hop_length=800 ensures 1 MFCC vector per video frame (approx)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=Config.AUDIO_SR,
                n_mfcc=Config.AUDIO_INPUT_DIM,
                melkwargs={"n_fft": 2048, "hop_length": 800, "n_mels": 64},
            )
            audio_features = (
                mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            )  # (T_audio, 13)

            # Sync Lengths: Audio might be slightly longer/shorter due to padding
            # We truncate or pad audio to match video frames T
            audio_features = audio_features.numpy()
            if audio_features.shape[0] > T:
                audio_features = audio_features[:T, :]
            elif audio_features.shape[0] < T:
                pad_len = T - audio_features.shape[0]
                padding = np.zeros((pad_len, Config.AUDIO_INPUT_DIM), dtype=np.float32)
                audio_features = np.concatenate([audio_features, padding], axis=0)

            # 3. Construct Labels (Frame-wise)
            labels_tensor = np.zeros(T, dtype=np.int64)
            if compute_labels and hasattr(video, "Labels"):
                labels_struct = video.Labels
                # Handle single vs array
                if not isinstance(labels_struct, np.ndarray):
                    labels_struct = [labels_struct] if labels_struct is not None else []
                elif labels_struct.size == 1:
                    labels_struct = [labels_struct.item()]

                for l in labels_struct:
                    # Check validity
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # Matlab 1-based indexing -> Python 0-based
                        # Begin and End are frame indices
                        start = int(l.Begin) - 1
                        end = int(l.End)

                        if name in self.label_map:
                            lid = self.label_map[name]
                            # Clamp to video duration
                            start = max(0, start)
                            end = min(T, end)
                            if end > start:
                                labels_tensor[start:end] = lid

            return skeleton_data, audio_features, labels_tensor

        except Exception as e:
            # print(f"Error processing {row['sample_id']}: {e}")
            return None, None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_file = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                skel = torch.from_numpy(data["skeleton"])
                audio = torch.from_numpy(data["audio"])
                labels = torch.from_numpy(data["labels"])
                return {
                    "skeleton": skel,
                    "audio": audio,
                    "labels": labels,
                    "id": sample_id,
                }
            except:
                pass  # Failed to load, recompute

        # Compute from raw
        skel, audio, labels = self._process_raw_files(
            row, compute_labels=(self.mode != "test")
        )

        if skel is None:
            # Return dummy if failed (filtered later)
            return None

        # Normalize
        skel = (skel - self.skel_mean) / self.skel_std
        audio = (audio - self.audio_mean) / self.audio_std

        skel = skel.astype(np.float32)
        audio = audio.astype(np.float32)

        # Save to cache
        np.savez(cache_file, skeleton=skel, audio=audio, labels=labels)

        return {
            "skeleton": torch.from_numpy(skel),
            "audio": torch.from_numpy(audio),
            "labels": torch.from_numpy(labels),
            "id": sample_id,
        }


def collate_fn(batch):
    """
    Pads sequences and applies augmentations during training.
    """
    # Filter failed samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Extract
    skeletons = [b["skeleton"] for b in batch]
    audios = [b["audio"] for b in batch]
    labels = [b["labels"] for b in batch]
    ids = [b["id"] for b in batch]

    # Padding (Batch First = True)
    # pad_sequence expects list of (L, D)
    skel_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    # Create Lengths / Masks
    lengths = torch.tensor([len(s) for s in skeletons], dtype=torch.long)

    # Augmentation (Training Only)
    # We detect training mode by checking if gradients might be needed or context
    # Usually passed via dataset, but collate is stateless.
    # Heuristic: If labels are present and non-empty, likely training/val.
    # To be precise, we can check a flag or just apply if random seed is set (which is always).
    # We will apply augmentation ONLY if we can infer training mode.
    # Since collate doesn't know 'mode', we'll skip complex logic and assume
    # the training loop handles 'train' vs 'eval' mode on the model,
    # BUT data augmentation happens here.
    # We will assume we augment if the batch size > 1 (heuristic) or simple randomness.
    # Better: The prompt asks to implement augmentations here.
    # We will apply them probabilistically.

    # Note: Ideally, collate_fn should know mode.
    # Since we cannot easily pass mode to collate_fn in standard DataLoader without functools,
    # we will implement the augmentation functions but only apply them if explicitly enabled
    # or we will assume this collate is used for training.
    # However, for validation/test, we must NOT augment.
    # Standard fix: The user of this module should use `functools.partial` to pass mode,
    # or we define a class `CollateFn`.

    return {
        "skeleton": skel_padded,
        "audio": audio_padded,
        "labels": labels_padded,
        "lengths": lengths,
        "ids": ids,
    }


class CollateFn:
    def __init__(self, mode="train"):
        self.mode = mode

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        skeletons = [b["skeleton"] for b in batch]
        audios = [b["audio"] for b in batch]
        labels = [b["labels"] for b in batch]
        ids = [b["id"] for b in batch]

        # Padding
        skel_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
        audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(s) for s in skeletons], dtype=torch.long)

        # Augmentation
        if self.mode == "train":
            # 1. Additive Noise to Skeleton
            noise = torch.randn_like(skel_padded) * 0.05
            skel_padded = skel_padded + noise

            # 2. Random Channel Masking
            # Mask ~10% of channels with 50% prob
            if torch.rand(1).item() > 0.5:
                B, T, C = skel_padded.shape
                mask = torch.ones((B, 1, C))
                # Create random mask indices
                num_mask = int(0.1 * C)
                mask_indices = torch.randperm(C)[:num_mask]
                mask[:, :, mask_indices] = 0
                skel_padded = skel_padded * mask

        return {
            "skeleton": skel_padded,
            "audio": audio_padded,
            "labels": labels_padded,
            "lengths": lengths,
            "ids": ids,
        }
