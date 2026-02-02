import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Gesture Recognition (Skeleton + Audio).
    Handles data loading, caching, preprocessing, normalization, and augmentation.
    """

    def __init__(self, split="train", limit=None, force_compute_stats=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            limit (int, optional): Limit dataset size for debugging.
            force_compute_stats (bool): If True, recompute global stats even if file exists.
        """
        self.split = split
        self.limit = limit

        # Select Metadata File and Cache Directory
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.cache_dir = Config.CACHE_TRAIN_DIR
            self.augment = True
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.cache_dir = Config.CACHE_VAL_DIR
            self.augment = False
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.cache_dir = Config.CACHE_TEST_DIR
            self.augment = False
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Metadata
        self.df = pd.read_csv(self.metadata_path)
        if self.limit:
            self.df = self.df.head(self.limit)

        # Audio Transform (MFCC)
        # Hop length = Sample Rate / FPS = 16000 / 20 = 800
        self.hop_length = int(Config.AUDIO_SAMPLE_RATE / Config.VIDEO_FPS)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": 2048,
                "hop_length": self.hop_length,
                "n_mels": 64,
                "center": False,  # To align strictly with frames
            },
        )

        # Global Statistics (Mean/Std)
        self.stats = self._load_or_compute_stats(force_compute_stats)

    def _load_or_compute_stats(self, force_compute):
        """
        Loads global stats from disk or computes them from the training cache.
        """
        # Only compute stats if we are the training set loader and stats don't exist (or forced)
        if self.split == "train" and (
            force_compute or not os.path.exists(Config.STATS_PATH)
        ):
            print("Computing global statistics from training data...")

            # We need to process all training samples first to compute stats
            # This ensures the cache is populated
            all_skeletons = []
            all_audios = []

            for idx in range(len(self.df)):
                data = self._process_sample(idx, load_cached_data=True)
                if data is not None:
                    all_skeletons.append(data["skeleton"])
                    all_audios.append(data["audio"])

            if not all_skeletons:
                raise RuntimeError("No valid training data found to compute stats.")

            # Concatenate all frames
            skel_concat = np.concatenate(all_skeletons, axis=0)  # (TotalFrames, 60)
            audio_concat = np.concatenate(all_audios, axis=0)  # (TotalFrames, 13)

            stats = {
                "skel_mean": np.mean(skel_concat, axis=0),
                "skel_std": np.std(skel_concat, axis=0) + 1e-6,  # Avoid div by zero
                "audio_mean": np.mean(audio_concat, axis=0),
                "audio_std": np.std(audio_concat, axis=0) + 1e-6,
            }

            np.savez(Config.STATS_PATH, **stats)
            print(f"Statistics saved to {Config.STATS_PATH}")
            return stats

        elif os.path.exists(Config.STATS_PATH):
            # Load existing stats
            loaded = np.load(Config.STATS_PATH)
            return {k: loaded[k] for k in loaded.files}

        else:
            # If we are val/test and stats don't exist, we can't proceed properly.
            # However, for the very first run, train loader should be initialized first.
            # If not, we return identity stats (zeros/ones) and warn.
            print("Warning: Statistics file not found. Using identity normalization.")
            return {
                "skel_mean": np.zeros(Config.SKELETON_INPUT_SIZE),
                "skel_std": np.ones(Config.SKELETON_INPUT_SIZE),
                "audio_mean": np.zeros(Config.AUDIO_INPUT_SIZE),
                "audio_std": np.ones(Config.AUDIO_INPUT_SIZE),
            }

    def _process_sample(self, idx, load_cached_data=True):
        """
        Loads, processes, and caches a single sample.
        Returns dictionary with numpy arrays: 'skeleton', 'audio', 'labels'.
        """
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return {
                    "skeleton": data["skeleton"],
                    "audio": data["audio"],
                    "labels": data["labels"],
                }
            except Exception as e:
                print(f"Error loading cache for {sample_id}: {e}. Recomputing.")

        # 2. Compute from Raw Data
        try:
            # --- Load Skeleton (.mat) ---
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]

            num_frames = getattr(video_struct, "NumFrames", 0)
            if num_frames == 0:
                return None

            # Extract Skeleton Frames
            # Frames is an array of structs. Each struct has a 'Skeleton' field.
            # 'Skeleton' usually has 'WorldPosition'.
            frames_data = video_struct.Frames

            # Pre-allocate skeleton array: (T, 20, 3)
            # 20 joints, 3 coords
            skeleton_frames = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            # Iterate frames
            # Handle case where frames_data is a single object or array
            if not isinstance(frames_data, np.ndarray):
                frames_data = [frames_data]

            for f_idx, frame_obj in enumerate(frames_data):
                if f_idx >= num_frames:
                    break

                if hasattr(frame_obj, "Skeleton"):
                    skel = frame_obj.Skeleton
                    # If multiple users, skel might be an array. Take first valid.
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        # wp should be (20, 3) or similar.
                        # If it's a struct of arrays, we might need different parsing.
                        # Assuming standard matrix form based on typical datasets.
                        if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                            skeleton_frames[f_idx] = wp
                        elif isinstance(wp, np.ndarray) and wp.size == 60:
                            skeleton_frames[f_idx] = wp.reshape(20, 3)

            # Normalize Skeleton: Relative to HipCenter (Joint 0)
            # Assuming Joint 0 is HipCenter.
            hip_centers = skeleton_frames[:, 0:1, :]  # (T, 1, 3)
            skeleton_frames = skeleton_frames - hip_centers

            # Flatten to (T, 60)
            skeleton_features = skeleton_frames.reshape(num_frames, -1)

            # --- Load Audio (.wav) ---
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            if os.path.exists(audio_path):
                waveform, sr = sf.read(audio_path)
                # Convert to torch for transform
                waveform = torch.tensor(waveform, dtype=torch.float32)

                # If stereo, mean to mono
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=1)

                # Resample if needed
                if sr != Config.AUDIO_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        sr, Config.AUDIO_SAMPLE_RATE
                    )
                    waveform = resampler(waveform)

                # Extract MFCC
                # (n_mfcc, time)
                mfcc = self.mfcc_transform(waveform)
                # Transpose to (time, n_mfcc)
                audio_features = mfcc.transpose(0, 1).numpy()
            else:
                # Fallback silent audio
                audio_features = np.zeros(
                    (num_frames, Config.AUDIO_INPUT_SIZE), dtype=np.float32
                )

            # --- Align Modalities ---
            # Truncate or Pad Audio to match Video Frames
            curr_audio_len = audio_features.shape[0]
            if curr_audio_len > num_frames:
                audio_features = audio_features[:num_frames]
            elif curr_audio_len < num_frames:
                pad_len = num_frames - curr_audio_len
                padding = np.zeros((pad_len, Config.AUDIO_INPUT_SIZE), dtype=np.float32)
                audio_features = np.concatenate([audio_features, padding], axis=0)

            # --- Construct Frame-wise Labels ---
            # Default background (0)
            labels_seq = np.zeros(num_frames, dtype=np.int64)

            if hasattr(video_struct, "Labels"):
                raw_labels = video_struct.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]

                # Map from prompt:
                # Label names are: 'vattene': 1, ...
                # We use the Config.NUM_CLASSES (21) implicitly via the map

                # Re-defining locally to ensure self-containment as requested
                LOCAL_LABEL_MAP = {
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
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # Matlab 1-based indexing
                        start_frame = int(l.Begin) - 1
                        end_frame = int(l.End)

                        if name in LOCAL_LABEL_MAP:
                            lbl_idx = LOCAL_LABEL_MAP[name]
                            # Clip to valid range
                            start_frame = max(0, start_frame)
                            end_frame = min(num_frames, end_frame)
                            if end_frame > start_frame:
                                labels_seq[start_frame:end_frame] = lbl_idx

            # --- Save to Cache ---
            np.savez(
                cache_path,
                skeleton=skeleton_features,
                audio=audio_features,
                labels=labels_seq,
            )

            return {
                "skeleton": skeleton_features,
                "audio": audio_features,
                "labels": labels_seq,
            }

        except Exception as e:
            print(f"Failed to process {sample_id}: {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Data
        data = self._process_sample(idx, load_cached_data=True)

        # Handle corruption or failure
        if data is None:
            # Return a dummy sample (zeros) to avoid crashing the loader
            # In practice, we should fix the data, but this is robust for runtime
            T = 100
            data = {
                "skeleton": np.zeros((T, Config.SKELETON_INPUT_SIZE), dtype=np.float32),
                "audio": np.zeros((T, Config.AUDIO_INPUT_SIZE), dtype=np.float32),
                "labels": np.zeros((T,), dtype=np.int64),
            }

        skel = data["skeleton"]
        audio = data["audio"]
        labels = data["labels"]

        # 1. Normalization (Z-Score)
        skel = (skel - self.stats["skel_mean"]) / self.stats["skel_std"]
        audio = (audio - self.stats["audio_mean"]) / self.stats["audio_std"]

        # 2. Augmentation (Train Only)
        if self.augment:
            # A. Additive Gaussian Noise (Skeleton only)
            noise = np.random.normal(0, Config.AUG_NOISE_SIGMA, skel.shape)
            skel = skel + noise

            # B. Random Channel Masking
            if np.random.rand() < Config.AUG_MASK_PROB:
                # Mask Skeleton Channels
                num_skel_channels = skel.shape[1]
                mask_indices = np.random.choice(
                    num_skel_channels,
                    size=int(num_skel_channels * Config.AUG_MASK_RATIO),
                    replace=False,
                )
                skel[:, mask_indices] = 0.0

                # Mask Audio Channels
                num_audio_channels = audio.shape[1]
                mask_indices_audio = np.random.choice(
                    num_audio_channels,
                    size=int(num_audio_channels * Config.AUG_MASK_RATIO),
                    replace=False,
                )
                audio[:, mask_indices_audio] = 0.0

        # Convert to Tensor
        return {
            "skeleton": torch.tensor(skel, dtype=torch.float32),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": self.df.iloc[idx]["sample_id"],
        }


def collate_fn(batch):
    """
    Collates a batch of variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Filter out None samples if any
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Find max length
    lengths = [b["skeleton"].shape[0] for b in batch]
    max_len = max(lengths)

    # Dimensions
    skel_dim = batch[0]["skeleton"].shape[1]
    audio_dim = batch[0]["audio"].shape[1]
    batch_size = len(batch)

    # Prepare padded tensors
    padded_skel = torch.zeros(batch_size, max_len, skel_dim, dtype=torch.float32)
    padded_audio = torch.zeros(batch_size, max_len, audio_dim, dtype=torch.float32)
    padded_labels = torch.zeros(
        batch_size, max_len, dtype=torch.long
    )  # Default 0 (background)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    sample_ids = []

    for i, b in enumerate(batch):
        length = lengths[i]
        padded_skel[i, :length] = b["skeleton"]
        padded_audio[i, :length] = b["audio"]
        padded_labels[i, :length] = b["labels"]
        mask[i, :length] = True
        sample_ids.append(b["sample_id"])

    return {
        "skeleton": padded_skel,
        "audio": padded_audio,
        "labels": padded_labels,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "sample_ids": sample_ids,
    }
