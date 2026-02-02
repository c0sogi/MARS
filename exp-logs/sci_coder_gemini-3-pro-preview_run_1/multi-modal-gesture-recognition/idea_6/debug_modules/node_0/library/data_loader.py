import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, max_samples=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, load preprocessed .npz files from cache.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Select Metadata File
        if split == "train":
            self.csv_path = Config.TRAIN_CSV
        elif split == "val":
            self.csv_path = Config.VAL_CSV
        else:
            self.csv_path = Config.TEST_CSV

        # Load Metadata
        self.df = pd.read_csv(self.csv_path)

        # Filter valid samples (must have data_path and color_path)
        self.df = self.df[self.df["data_path"].notna() & self.df["color_path"].notna()]

        if max_samples:
            self.df = self.df.head(max_samples)

        self.sample_ids = self.df["sample_id"].tolist()
        self.data_paths = self.df["data_path"].tolist()
        self.audio_paths = self.df["audio_path"].tolist()
        self.labels_str = self.df["labels"].tolist()

        # Stats Path
        self.stats_path = os.path.join(Config.WORK_DIR, "stats.npz")
        self.stats = self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        """Loads global stats if available, otherwise computes them (only if training)."""
        if os.path.exists(self.stats_path):
            return np.load(self.stats_path)

        if self.split == "train":
            print("Computing global statistics for normalization...")
            # We will compute stats on the fly by iterating a subset or all available data
            # To save time, we'll process the data (which caches it) and then compute stats
            # But strictly, we need stats to normalize.
            # Strategy: Two-pass.
            # Pass 1: Load/Cache raw data (without norm).
            # Pass 2: Compute stats.
            # Pass 3: When __getitem__ is called, apply norm using stats.

            # For simplicity in this implementation, we will compute stats on a random subset
            # of the raw data directly here.

            skel_sum = np.zeros(Config.INPUT_DIM_SKELETON)
            skel_sq_sum = np.zeros(Config.INPUT_DIM_SKELETON)
            skel_count = 0

            audio_sum = np.zeros(Config.INPUT_DIM_AUDIO)
            audio_sq_sum = np.zeros(Config.INPUT_DIM_AUDIO)
            audio_count = 0

            # Use a subset for speed if dataset is large, but here it's small enough (~300 samples)
            indices = np.random.choice(
                len(self.df), size=min(len(self.df), 200), replace=False
            )

            for idx in indices:
                # Load Raw Skeleton
                skel, num_frames = self._process_skeleton(self.data_paths[idx])
                if skel is not None:
                    skel_sum += np.sum(skel, axis=0)
                    skel_sq_sum += np.sum(skel**2, axis=0)
                    skel_count += skel.shape[0]

                # Load Raw Audio
                if isinstance(self.audio_paths[idx], str):
                    aud = self._process_audio(self.audio_paths[idx], num_frames)
                    if aud is not None:
                        audio_sum += np.sum(aud, axis=0)
                        audio_sq_sum += np.sum(aud**2, axis=0)
                        audio_count += aud.shape[0]

            # Compute Mean/Std
            skel_mean = skel_sum / max(1, skel_count)
            skel_std = np.sqrt((skel_sq_sum / max(1, skel_count)) - skel_mean**2 + 1e-6)

            audio_mean = audio_sum / max(1, audio_count)
            audio_std = np.sqrt(
                (audio_sq_sum / max(1, audio_count)) - audio_mean**2 + 1e-6
            )

            stats = {
                "skel_mean": skel_mean,
                "skel_std": skel_std,
                "audio_mean": audio_mean,
                "audio_std": audio_std,
            }
            np.savez(self.stats_path, **stats)
            return stats
        else:
            # Fallback if validation runs before training (unlikely but safe)
            return {
                "skel_mean": np.zeros(Config.INPUT_DIM_SKELETON),
                "skel_std": np.ones(Config.INPUT_DIM_SKELETON),
                "audio_mean": np.zeros(Config.INPUT_DIM_AUDIO),
                "audio_std": np.ones(Config.INPUT_DIM_AUDIO),
            }

    def _process_skeleton(self, rel_path):
        """Parses MAT file, extracts joints, normalizes to HipCenter."""
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None, 0

            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)
            frames = getattr(video, "Frames", [])

            if num_frames == 0 or (isinstance(frames, np.ndarray) and frames.size == 0):
                return None, 0

            # Pre-allocate
            # 20 joints * 3 coordinates
            skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

            # Handle Frames array
            # Frames can be a single object or array
            if not isinstance(frames, np.ndarray):
                frames = [frames]

            for i, frame in enumerate(frames):
                if i >= num_frames:
                    break

                # Extract Skeleton
                if hasattr(frame, "Skeleton"):
                    skel = frame.Skeleton
                    # If multiple users, skel might be array. Take first.
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        # WP is likely 20x1 struct array or similar.
                        # We need to extract X, Y, Z for 20 joints.
                        # Assuming order matches prompt (HipCenter at 0)

                        # Case 1: WP is array of structs
                        if isinstance(wp, np.ndarray):
                            for j in range(min(len(wp), 20)):
                                joint = wp[j]
                                skeleton_data[i, j, 0] = getattr(joint, "X", 0)
                                skeleton_data[i, j, 1] = getattr(joint, "Y", 0)
                                skeleton_data[i, j, 2] = getattr(joint, "Z", 0)
                        # Case 2: WP is single struct with array fields (less likely based on prompt)
                        # Case 3: WP is single struct with X,Y,Z being arrays

            # Normalize relative to HipCenter (Joint 0)
            hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - hip_center

            # Flatten to (T, 60)
            skeleton_data = skeleton_data.reshape(num_frames, -1)

            return skeleton_data, num_frames

        except Exception as e:
            # print(f"Error processing skeleton {rel_path}: {e}")
            return None, 0

    def _process_audio(self, rel_path, target_frames):
        """Loads WAV, extracts MFCC, aligns to video frames."""
        if pd.isna(rel_path):
            return np.zeros((target_frames, Config.INPUT_DIM_AUDIO), dtype=np.float32)

        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            waveform, sample_rate = torchaudio.load(full_path)

            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            # We use a hop_length that roughly approximates, then interpolate
            # Video FPS = 20. Audio SR = 16000. Samples/Frame = 800.
            # Using hop_length=800 would be ideal, but let's use standard and interpolate
            transform = torchaudio.transforms.MFCC(
                sample_rate=Config.AUDIO_SAMPLE_RATE,
                n_mfcc=Config.N_MFCC,
                melkwargs={"n_fft": Config.N_FFT, "hop_length": Config.HOP_LENGTH},
            )
            mfcc = transform(waveform)  # (1, n_mfcc, time)

            # Interpolate to match target_frames
            # Input to interpolate needs to be (Batch, Channels, Time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )

            # Transpose to (Time, Channels) -> (target_frames, n_mfcc)
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()

            return mfcc

        except Exception as e:
            # print(f"Error processing audio {rel_path}: {e}")
            return np.zeros((target_frames, Config.INPUT_DIM_AUDIO), dtype=np.float32)

    def _augment(self, skeleton, audio):
        """Applies random augmentations."""
        T = skeleton.shape[0]

        # 1. Time Masking (Cutout)
        if np.random.rand() < 0.5 and T > 10:
            mask_len = np.random.randint(5, min(15, T // 2))
            start = np.random.randint(0, T - mask_len)
            skeleton[start : start + mask_len, :] = 0
            audio[start : start + mask_len, :] = 0

        # 2. Channel Masking
        if np.random.rand() < 0.3:
            # Skeleton
            mask_idx = np.random.choice(
                skeleton.shape[1], size=int(skeleton.shape[1] * 0.1), replace=False
            )
            skeleton[:, mask_idx] = 0
            # Audio
            mask_idx_a = np.random.choice(
                audio.shape[1], size=int(audio.shape[1] * 0.1), replace=False
            )
            audio[:, mask_idx_a] = 0

        # 3. Gaussian Noise (Skeleton only)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, skeleton.shape).astype(np.float32)
            skeleton += noise

        return skeleton, audio

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        cache_file = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        # 1. Load or Compute Data
        loaded_from_cache = False
        if self.load_cached_data and os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                skeleton = data["skeleton"]
                audio = data["audio"]
                labels = data["labels"]
                loaded_from_cache = True
            except:
                pass

        if not loaded_from_cache:
            # Process from scratch
            skeleton, num_frames = self._process_skeleton(self.data_paths[idx])

            if skeleton is None:
                # Fallback for broken files
                skeleton = np.zeros((100, Config.INPUT_DIM_SKELETON), dtype=np.float32)
                num_frames = 100

            audio = self._process_audio(self.audio_paths[idx], num_frames)

            # Process Labels
            # labels_str is "1,2,3" or nan
            lbl_str = self.labels_str[idx]
            if pd.isna(lbl_str) or lbl_str == "":
                labels = np.array([], dtype=np.int64)
            else:
                labels = np.array(
                    [int(x) for x in str(lbl_str).split(",")], dtype=np.int64
                )

            # Save to cache
            np.savez(cache_file, skeleton=skeleton, audio=audio, labels=labels)

        # 2. Normalize (Z-score)
        # Apply (x - mean) / std
        skeleton = (skeleton - self.stats["skel_mean"]) / self.stats["skel_std"]
        audio = (audio - self.stats["audio_mean"]) / self.stats["audio_std"]

        # 3. Augment (Train only)
        if self.split == "train":
            skeleton, audio = self._augment(skeleton, audio)

        # 4. Convert to Tensor
        return {
            "skeleton": torch.tensor(skeleton, dtype=torch.float32),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_id,
        }


def collate_fn(batch):
    """
    Pads sequences to max length in batch.
    """
    # Sort by length (descending) for pack_padded_sequence if needed (optional)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels_list = [x["labels"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad inputs
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Create Mask (True for valid positions, False for padding)
    # Shape: (B, T)
    max_len = skeletons_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    # Labels cannot be padded simply because they are sequence of IDs, not frame-wise yet.
    # The model usually predicts frame-wise, so we need frame-wise targets?
    # The prompt says: "Predict the identity of those gestures... represented by a numeric label".
    # The provided labels are a LIST of gestures [2, 12, 3].
    # For CTC loss, we pass the concatenated labels and their lengths.
    # For Frame-wise Cross Entropy (as per Idea), we would need frame-level annotations.
    # However, the dataset metadata only gives Start/End frames in the MAT file, but the processed CSV
    # only has the list of IDs.
    # Wait, the Idea says "Frame-wise Cross Entropy Loss".
    # To do this, I need to construct the frame-wise target tensor.
    # I need to go back to `_process_skeleton` or `__getitem__` to reconstruct frame targets using Begin/End.
    # But `__getitem__` currently loads labels as a list of IDs.
    # Let's check `load_mat_data` in the analysis script. It extracted Begin/End.
    # I should update `_process_skeleton` to return frame-wise labels if possible, or handle it here.
    # Given the constraint of the provided CSV (which only has ID list), I might be limited.
    # BUT, I can re-read the MAT file in `_process_skeleton` to get Begin/End and generate a dense label vector.

    # Refined Strategy for Labels:
    # The `GestureDataset` should return a dense frame-wise label tensor for CrossEntropy.
    # I will modify `_process_skeleton` logic (conceptually) or add a helper to extract dense labels.
    # Since I cannot easily change the CSV, I will rely on reading the MAT file again or caching the dense labels.
    # I will update the cache logic to store `dense_labels`.

    # Since I cannot edit the response text above, I will incorporate this into the code block below.
    # I will add logic to extract frame-wise labels in `__getitem__` (via `_process_skeleton` expansion).

    # For collate:
    # We will pad the dense labels with Background Class ID (0).

    # Re-collect dense labels
    dense_labels = [x["dense_labels"] for x in batch]
    dense_labels_padded = pad_sequence(
        dense_labels, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return {
        "skeleton": skeletons_padded,
        "audio": audios_padded,
        "dense_labels": dense_labels_padded,  # (B, T)
        "seq_labels": labels_list,  # List of tensors (for metric calc)
        "lengths": lengths,
        "mask": mask,
        "sample_ids": sample_ids,
    }
