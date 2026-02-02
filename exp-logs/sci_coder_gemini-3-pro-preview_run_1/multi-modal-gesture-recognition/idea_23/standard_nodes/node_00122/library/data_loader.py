import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class GestureDataset(Dataset):
    """
    Multimodal dataset for gesture recognition (Skeleton + Audio).
    Handles loading, preprocessing, caching, augmentation, and normalization.
    """

    def __init__(
        self, metadata_path, mode="train", mean=None, std=None, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            mean (tuple): (skeleton_mean, audio_mean) tensors for normalization.
            std (tuple): (skeleton_std, audio_std) tensors for normalization.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Normalization stats
        self.skel_mean = mean[0] if mean is not None else None
        self.skel_std = std[0] if std is not None else None
        self.audio_mean = mean[1] if mean is not None else None
        self.audio_std = std[1] if std is not None else None

        # Audio Transform (Physics-based MFCC)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.HOP_LENGTH,
                "n_mels": 40,
                "center": False,  # To align strictly with windows
            },
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]

        # Load data (from cache or raw)
        skel, audio, labels = self._load_sample(row)

        # Convert to torch tensors
        skel = torch.from_numpy(skel).float()
        audio = torch.from_numpy(audio).float()
        labels = torch.from_numpy(labels).long()

        # Augmentation (Train only)
        if self.mode == "train":
            skel, audio, labels = self._augment(skel, audio, labels)

        # Normalization
        if self.skel_mean is not None and self.skel_std is not None:
            skel = (skel - self.skel_mean) / (self.skel_std + 1e-6)

        if self.audio_mean is not None and self.audio_std is not None:
            audio = (audio - self.audio_mean) / (self.audio_std + 1e-6)

        return {
            "skeleton": skel,
            "audio": audio,
            "labels": labels,
            "sample_id": sample_id,
        }

    def _load_sample(self, row):
        """
        Loads sample data. Uses caching mechanism.
        Returns:
            skel (np.ndarray): (T, 60) - Flattened root-relative joints.
            audio (np.ndarray): (T, 13) - MFCCs.
            labels (np.ndarray): (T,) - Frame-wise labels.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{row['sample_id']}.npz")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data["skeleton"], data["audio"], data["labels"]
            except Exception:
                pass  # Fallback to raw load if cache is corrupt

        # --- Raw Processing ---

        # 1. Load MAT (Skeleton & Labels)
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = video.NumFrames

        # Extract Skeleton
        # video.Frames is array of structs. Each has Skeleton.
        # We need to robustly extract 20 joints x 3 (X,Y,Z).
        # Joint order is implicit in prompt list.

        # Pre-allocate skeleton array: (NumFrames, 20, 3)
        skeleton_data = np.zeros(
            (num_frames, Config.SKELETON_NUM_JOINTS, 3), dtype=np.float32
        )

        frames = video.Frames
        # Handle case where Frames is a single object or array
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        # Iterate frames
        for i, frame in enumerate(frames):
            if i >= num_frames:
                break

            if hasattr(frame, "Skeleton"):
                skel_obj = frame.Skeleton
                # If multiple users, take first
                if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                    skel_obj = skel_obj[0]

                if hasattr(skel_obj, "WorldPosition"):
                    wp = skel_obj.WorldPosition
                    # wp should be 20x1 struct array or similar based on prompt description
                    # Prompt: "WorldPosition... X value... Y value... Z value"
                    # Usually in these datasets, WorldPosition is a struct array of size 20
                    # or a single struct with arrays.
                    # Let's assume WorldPosition is an array of 20 structs (Joints).

                    if (
                        isinstance(wp, np.ndarray)
                        and wp.size == Config.SKELETON_NUM_JOINTS
                    ):
                        for j in range(Config.SKELETON_NUM_JOINTS):
                            joint = wp[j]
                            skeleton_data[i, j, 0] = joint.X
                            skeleton_data[i, j, 1] = joint.Y
                            skeleton_data[i, j, 2] = joint.Z

        # Root-Relative Normalization
        # HipCenter is usually index 0 (based on prompt list: HipCenter, Spine, ...)
        hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_center

        # Flatten: (T, 60)
        skeleton_data = skeleton_data.reshape(num_frames, -1)

        # 2. Load Audio & Extract MFCC
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        waveform, sample_rate = torchaudio.load(audio_path)

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary (though analysis said 16k is consistent)
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Extract MFCC: (1, n_mfcc, n_frames_audio)
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (T_audio, n_mfcc)

        # Align Audio to Video Frames
        # Truncate or Pad
        if mfcc.shape[0] > num_frames:
            mfcc = mfcc[:num_frames, :]
        elif mfcc.shape[0] < num_frames:
            pad_len = num_frames - mfcc.shape[0]
            padding = np.zeros((pad_len, mfcc.shape[1]), dtype=mfcc.dtype)
            mfcc = np.concatenate([mfcc, padding], axis=0)

        # 3. Construct Labels
        labels = np.full(num_frames, Config.BACKGROUND_LABEL, dtype=np.int64)

        # Labels are only present in Train/Val usually. Test might be empty.
        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]
            elif raw_labels.size == 0:
                raw_labels = []

            for l in raw_labels:
                # Check validity
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in Config.LABEL_MAP:
                        lid = Config.LABEL_MAP[name]
                        # Matlab 1-based indexing -> Python 0-based
                        # Begin and End are frame indices
                        start = int(l.Begin) - 1
                        end = int(l.End) - 1

                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames - 1, end)

                        if end >= start:
                            labels[start : end + 1] = lid

        # Save to cache
        np.savez_compressed(
            cache_path, skeleton=skeleton_data, audio=mfcc, labels=labels
        )

        return skeleton_data, mfcc, labels

    def _augment(self, skel, audio, labels):
        """
        Applies Global Temporal Resampling and Channel Masking.
        """
        # 1. Global Temporal Resampling
        # Factor alpha ~ U(0.8, 1.2)
        alpha = np.random.uniform(0.8, 1.2)
        new_len = int(skel.shape[0] * alpha)

        # Ensure min length
        new_len = max(Config.MIN_GESTURE_LENGTH, new_len)

        # Interpolate Features (Linear)
        # Input to interpolate: (Batch, Channels, Time)
        skel_t = skel.unsqueeze(0).permute(0, 2, 1)  # (1, 60, T)
        audio_t = audio.unsqueeze(0).permute(0, 2, 1)  # (1, 13, T)

        skel_res = F.interpolate(
            skel_t, size=new_len, mode="linear", align_corners=False
        )
        audio_res = F.interpolate(
            audio_t, size=new_len, mode="linear", align_corners=False
        )

        skel = skel_res.squeeze(0).permute(1, 0)  # (T_new, 60)
        audio = audio_res.squeeze(0).permute(1, 0)  # (T_new, 13)

        # Interpolate Labels (Nearest)
        labels_t = labels.unsqueeze(0).unsqueeze(0).float()  # (1, 1, T)
        labels_res = F.interpolate(labels_t, size=new_len, mode="nearest")
        labels = labels_res.squeeze().long()

        # 2. Random Channel Masking (Feature Dropout)
        # Probability 0.1
        if np.random.random() < 0.1:
            # Mask Skeleton Channels
            mask_skel = torch.bernoulli(torch.full((skel.shape[1],), 0.9)).to(
                skel.device
            )
            skel = skel * mask_skel

            # Mask Audio Channels
            mask_audio = torch.bernoulli(torch.full((audio.shape[1],), 0.9)).to(
                audio.device
            )
            audio = audio * mask_audio

        return skel, audio, labels


def collate_fn(batch):
    """
    Collate function for padding variable length sequences.
    """
    # Sort by length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels = [x["labels"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons])

    # Pad sequences
    # pad_sequence pads dim 0. batch_first=True -> (B, T, C)
    padded_skel = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0
    )
    padded_audio = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=Config.BACKGROUND_LABEL
    )

    return {
        "skeleton": padded_skel,
        "audio": padded_audio,
        "labels": padded_labels,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_dataloaders():
    """
    Factory function to create dataloaders.
    Computes/Loads global stats from training data.
    """
    # 1. Initialize Train Dataset (without normalization first to compute stats)
    # We set load_cached_data=True. If cache is empty, it will process and save.
    print("Initializing Training Dataset...")
    train_ds_raw = GestureDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )

    # 2. Compute or Load Stats
    stats_path = os.path.join(Config.WORK_DIR, "stats.npz")

    if os.path.exists(stats_path):
        print(f"Loading global stats from {stats_path}")
        stats = np.load(stats_path)
        skel_mean = torch.from_numpy(stats["skel_mean"])
        skel_std = torch.from_numpy(stats["skel_std"])
        audio_mean = torch.from_numpy(stats["audio_mean"])
        audio_std = torch.from_numpy(stats["audio_std"])
    else:
        print("Computing global stats from training data (this may take a moment)...")
        # We iterate over the dataset to compute running mean/std
        # Using Welford's algorithm or simple accumulation if dataset fits in memory (it doesn't easily)
        # We'll use simple accumulation for simplicity as N is manageable (~300 videos * ~500 frames)

        all_skel = []
        all_audio = []

        # Iterate raw dataset
        for i in range(len(train_ds_raw)):
            # Force load (it will cache)
            s, a, _ = train_ds_raw._load_sample(train_ds_raw.metadata.iloc[i])
            all_skel.append(s)
            all_audio.append(a)

        all_skel = np.concatenate(all_skel, axis=0)  # (TotalFrames, 60)
        all_audio = np.concatenate(all_audio, axis=0)  # (TotalFrames, 13)

        skel_mean = np.mean(all_skel, axis=0)
        skel_std = np.std(all_skel, axis=0)
        audio_mean = np.mean(all_audio, axis=0)
        audio_std = np.std(all_audio, axis=0)

        np.savez(
            stats_path,
            skel_mean=skel_mean,
            skel_std=skel_std,
            audio_mean=audio_mean,
            audio_std=audio_std,
        )

        skel_mean = torch.from_numpy(skel_mean)
        skel_std = torch.from_numpy(skel_std)
        audio_mean = torch.from_numpy(audio_mean)
        audio_std = torch.from_numpy(audio_std)
        print("Stats computed and saved.")

    # 3. Create Datasets with Normalization
    train_ds = GestureDataset(
        Config.TRAIN_METADATA_PATH,
        mode="train",
        mean=(skel_mean, audio_mean),
        std=(skel_std, audio_std),
    )

    val_ds = GestureDataset(
        Config.VAL_METADATA_PATH,
        mode="val",
        mean=(skel_mean, audio_mean),
        std=(skel_std, audio_std),
    )

    test_ds = GestureDataset(
        Config.TEST_METADATA_PATH,
        mode="test",
        mean=(skel_mean, audio_mean),
        std=(skel_std, audio_std),
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
