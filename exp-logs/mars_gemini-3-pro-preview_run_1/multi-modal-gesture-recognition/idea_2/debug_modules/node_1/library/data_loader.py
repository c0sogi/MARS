import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import Config
from library.utils import save_npz, load_npz


class GestureDataset(Dataset):
    def __init__(self, split="train", augment=False, limit=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation (only for training).
            limit (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.augment = augment

        # Select Metadata File and Cache Directory
        if split == "train":
            self.csv_path = Config.TRAIN_CSV
            self.cache_dir = Config.CACHE_TRAIN_DIR
        elif split == "val":
            self.csv_path = Config.VAL_CSV
            self.cache_dir = Config.CACHE_VAL_DIR
        else:
            self.csv_path = Config.TEST_CSV
            self.cache_dir = Config.CACHE_TEST_DIR

        # Load Metadata
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if limit:
            self.df = self.df.head(limit)

        self.sample_ids = self.df["sample_id"].tolist()

        # Normalization Statistics
        self.stats_path = os.path.join(Config.CACHE_DIR, "stats.npz")
        self.mean_skel = None
        self.std_skel = None
        self.mean_audio = None
        self.std_audio = None

        # If training, ensure stats are computed. If inference, load them.
        if split == "train":
            # We need to ensure all data is cached to compute stats
            # However, to save time on first init, we might compute stats lazily or
            # assume they exist. For robustness in this task, we'll try to load,
            # and if missing, we will compute them by iterating the dataset once.
            if not os.path.exists(self.stats_path):
                print("Computing normalization statistics (this may take a while)...")
                self._compute_global_stats()

        # Load stats if they exist
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.mean_skel = torch.from_numpy(stats["mean_skel"]).float()
            self.std_skel = torch.from_numpy(stats["std_skel"]).float()
            self.mean_audio = torch.from_numpy(stats["mean_audio"]).float()
            self.std_audio = torch.from_numpy(stats["std_audio"]).float()
            # Avoid division by zero
            self.std_skel[self.std_skel == 0] = 1.0
            self.std_audio[self.std_audio == 0] = 1.0

    def _compute_global_stats(self):
        """Computes global mean and std for skeleton and audio from training data."""
        # Temporary lists to hold all features would be too large.
        # We use Welford's online algorithm or simple sum/sq_sum accumulation.

        sum_skel = torch.zeros(Config.INPUT_DIM_SKELETON)
        sq_sum_skel = torch.zeros(Config.INPUT_DIM_SKELETON)
        count_skel = 0

        sum_audio = torch.zeros(Config.INPUT_DIM_AUDIO)
        sq_sum_audio = torch.zeros(Config.INPUT_DIM_AUDIO)
        count_audio = 0

        # Iterate over all samples in the dataframe
        for idx in range(len(self.df)):
            # Force load/process
            data = self._get_item_data(idx)
            if data is None:
                continue

            skel = torch.from_numpy(data["skeleton"]).float()  # (T, D)
            audio = torch.from_numpy(data["audio"]).float()  # (T, D)

            # Accumulate Skeleton
            sum_skel += skel.sum(dim=0)
            sq_sum_skel += (skel**2).sum(dim=0)
            count_skel += skel.shape[0]

            # Accumulate Audio
            sum_audio += audio.sum(dim=0)
            sq_sum_audio += (audio**2).sum(dim=0)
            count_audio += audio.shape[0]

        # Compute Mean and Std
        mean_skel = sum_skel / count_skel
        std_skel = torch.sqrt((sq_sum_skel / count_skel) - (mean_skel**2))

        mean_audio = sum_audio / count_audio
        std_audio = torch.sqrt((sq_sum_audio / count_audio) - (mean_audio**2))

        # Save
        np.savez(
            self.stats_path,
            mean_skel=mean_skel.numpy(),
            std_skel=std_skel.numpy(),
            mean_audio=mean_audio.numpy(),
            std_audio=std_audio.numpy(),
        )

        self.mean_skel = mean_skel
        self.std_skel = std_skel
        self.mean_audio = mean_audio
        self.std_audio = std_audio

    def _process_raw_data(self, row):
        """
        Reads raw .mat and .wav files and extracts features.
        Returns dict with keys: 'skeleton', 'audio', 'labels'.
        """
        try:
            sample_id = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # 1. Load MAT file
            # struct_as_record=False makes structs objects, squeeze_me=True simplifies arrays
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)

            if num_frames == 0:
                return None

            # --- SKELETON EXTRACTION ---
            # Initialize (Time, Joints, 3)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            frames = getattr(video, "Frames", [])
            # Handle case where Frames is a single object or empty
            if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
                frames = [frames] if frames else []

            # If frames count doesn't match num_frames, take min
            actual_frames = len(frames)
            process_len = min(num_frames, actual_frames)

            for t in range(process_len):
                frame_obj = frames[t]
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel_obj = frame_obj.Skeleton
                # Check if skeleton exists and is not empty
                # Sometimes skel_obj might be an array (multiple users), take first
                if isinstance(skel_obj, np.ndarray):
                    if skel_obj.size == 0:
                        continue
                    skel_obj = skel_obj[0]

                # Now skel_obj should be the Skeleton structure
                # It should contain WorldPosition or be an array of joints
                # Based on prompt: "Skeleton Frame: An array of Skeleton structures... JointsType... WorldPosition"
                # This implies skel_obj is likely an array of joints.

                joints_array = None
                if isinstance(skel_obj, np.ndarray):
                    joints_array = skel_obj
                elif hasattr(skel_obj, "WorldPosition"):
                    # It might be a single struct with arrays inside, or just one joint?
                    # If it has JointsType, it's likely a single joint struct, which is wrong for a whole skeleton.
                    # Let's assume it's an array if accessible.
                    pass

                # If we can't iterate as array, try to access fields directly if it's a struct of arrays
                # But standard Kinect MAT format usually has an array of structs for joints.

                if joints_array is not None and len(joints_array) >= Config.NUM_JOINTS:
                    for j_idx in range(Config.NUM_JOINTS):
                        joint = joints_array[j_idx]
                        if hasattr(joint, "WorldPosition"):
                            wp = joint.WorldPosition
                            # wp should have X, Y, Z
                            if hasattr(wp, "X"):
                                skeleton_data[t, j_idx, 0] = wp.X
                                skeleton_data[t, j_idx, 1] = wp.Y
                                skeleton_data[t, j_idx, 2] = wp.Z

            # Normalize Skeleton (Relative to HipCenter)
            # HipCenter is index 0
            hip_centers = skeleton_data[:, Config.HIP_CENTER_INDEX, :].copy()  # (T, 3)
            # Broadcast subtraction: (T, J, 3) - (T, 1, 3)
            skeleton_data = skeleton_data - hip_centers[:, np.newaxis, :]

            # Flatten: (T, J*3)
            skeleton_flat = skeleton_data.reshape(num_frames, -1)

            # --- AUDIO EXTRACTION ---
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            # waveform: (Channels, Samples)

            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Mix to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            # n_mfcc=13, n_fft=2048, hop_length=800
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=Config.AUDIO_SAMPLE_RATE,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": Config.AUDIO_N_FFT,
                    "hop_length": Config.AUDIO_HOP_LENGTH,
                    "n_mels": 40,  # Standard default
                    "center": False,  # To align better with frames usually
                },
            )
            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, Time_audio)
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (Time_audio, n_mfcc)

            # Align Audio and Video
            # We want audio length to match num_frames
            if mfcc.shape[0] < num_frames:
                # Pad
                pad_len = num_frames - mfcc.shape[0]
                padding = np.zeros((pad_len, mfcc.shape[1]), dtype=np.float32)
                mfcc = np.concatenate([mfcc, padding], axis=0)
            elif mfcc.shape[0] > num_frames:
                # Trim
                mfcc = mfcc[:num_frames, :]

            # --- LABELS EXTRACTION ---
            labels_vec = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                # Standardize to list
                if not isinstance(raw_labels, np.ndarray) and not isinstance(
                    raw_labels, list
                ):
                    raw_labels = [raw_labels]
                elif isinstance(raw_labels, np.ndarray) and raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]
                elif isinstance(raw_labels, np.ndarray):
                    raw_labels = raw_labels.tolist()

                for l in raw_labels:
                    # Check validity
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in Config.LABEL_MAP:
                            lid = Config.LABEL_MAP[name]
                            # MATLAB is 1-based, Python 0-based
                            # Begin and End are frame indices
                            start = int(l.Begin) - 1
                            end = int(l.End)

                            # Clip to valid range
                            start = max(0, start)
                            end = min(num_frames, end)

                            if end > start:
                                labels_vec[start:end] = lid

            return {
                "skeleton": skeleton_flat.astype(np.float32),
                "audio": mfcc.astype(np.float32),
                "labels": labels_vec.astype(np.int64),
            }

        except Exception as e:
            print(f"Error processing {row['sample_id']}: {e}")
            return None

    def _get_item_data(self, idx):
        """Retrieves data for index, using cache if available."""
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Try Load Cache
        if os.path.exists(cache_path):
            data = load_npz(cache_path)
            # Validate keys to ensure cache consistency
            required_keys = ["skeleton", "audio", "labels"]
            if data is not None and all(k in data for k in required_keys):
                return data
            else:
                print(
                    f"Warning: Cache for {sample_id} is invalid or missing keys. Re-processing."
                )

        # 2. Process Raw
        data = self._process_raw_data(row)
        if data is None:
            # Create dummy data to avoid crashing, but print error
            # This shouldn't happen with clean data
            print(f"Warning: Failed to load {sample_id}, returning dummy.")
            return {
                "skeleton": np.zeros((10, Config.INPUT_DIM_SKELETON), dtype=np.float32),
                "audio": np.zeros((10, Config.INPUT_DIM_AUDIO), dtype=np.float32),
                "labels": np.zeros(10, dtype=np.int64),
            }

        # 3. Save Cache
        save_npz(cache_path, data)
        return data

    def _apply_augmentations(self, skeleton, audio):
        """Applies random augmentations to training data."""
        # 1. Additive Gaussian Noise to Skeleton
        if Config.AUG_NOISE_SIGMA > 0:
            noise = torch.randn_like(skeleton) * Config.AUG_NOISE_SIGMA
            skeleton = skeleton + noise

        # 2. Random Channel Masking
        # Mask Skeleton Channels
        if np.random.rand() < Config.AUG_MASK_CHANNEL_PROB:
            mask_indices = np.random.choice(
                skeleton.shape[1],
                size=int(skeleton.shape[1] * Config.AUG_MASK_CHANNEL_RATIO),
                replace=False,
            )
            skeleton[:, mask_indices] = 0.0

        # Mask Audio Channels (MFCC coefficients)
        if np.random.rand() < Config.AUG_MASK_CHANNEL_PROB:
            mask_indices = np.random.choice(
                audio.shape[1],
                size=int(audio.shape[1] * Config.AUG_MASK_CHANNEL_RATIO),
                replace=False,
            )
            audio[:, mask_indices] = 0.0

        return skeleton, audio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Data
        data_dict = self._get_item_data(idx)

        skeleton = torch.from_numpy(data_dict["skeleton"])
        audio = torch.from_numpy(data_dict["audio"])
        labels = torch.from_numpy(data_dict["labels"])

        # Apply Normalization
        if self.mean_skel is not None:
            skeleton = (skeleton - self.mean_skel) / self.std_skel
            audio = (audio - self.mean_audio) / self.std_audio

        # Apply Augmentation (only in training)
        if self.split == "train" and self.augment:
            skeleton, audio = self._apply_augmentations(skeleton, audio)

        return skeleton, audio, labels


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    Returns:
        padded_skeleton: (B, T_max, D_skel)
        padded_audio: (B, T_max, D_audio)
        padded_labels: (B, T_max)
        lengths: (B,)
    """
    skeletons, audios, labels = zip(*batch)

    lengths = torch.tensor([s.shape[0] for s in skeletons])
    max_len = lengths.max().item()

    batch_size = len(batch)
    dim_skel = skeletons[0].shape[1]
    dim_audio = audios[0].shape[1]

    # Initialize padded tensors
    padded_skel = torch.zeros(batch_size, max_len, dim_skel)
    padded_audio = torch.zeros(batch_size, max_len, dim_audio)
    # Fill with -100 (ignore index) for labels
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i in range(batch_size):
        l = lengths[i]
        padded_skel[i, :l, :] = skeletons[i]
        padded_audio[i, :l, :] = audios[i]
        padded_labels[i, :l] = labels[i]

    return padded_skel, padded_audio, padded_labels, lengths
