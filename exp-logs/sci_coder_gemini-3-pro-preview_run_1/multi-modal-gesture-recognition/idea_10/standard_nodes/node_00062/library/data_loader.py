import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    LABEL_MAP,
    SKELETON_JOINTS,
    SKELETON_CHANNELS,
    AUDIO_N_MFCC,
    BATCH_SIZE,
    SEED,
    BACKGROUND_CLASS_ID,
)
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split="train",
        transform=None,
        load_cached_data=True,
        stats_path=None,
    ):
        """
        Args:
            metadata_path (str): Path to the CSV metadata file.
            split (str): 'train', 'val', or 'test'.
            transform (bool): Whether to apply augmentations (usually True for train).
            load_cached_data (bool): Whether to use cached .npz files.
            stats_path (str): Path to pre-computed mean/std stats.
        """
        self.split = split
        self.transform = transform
        self.load_cached_data = load_cached_data
        self.metadata = pd.read_csv(metadata_path)

        # Filter out entries without essential data if necessary
        # The metadata script already does some filtering, but we ensure safety here
        self.metadata = self.metadata[self.metadata["data_path"].notna()]

        # Load or Compute Global Statistics for Normalization
        self.stats = None
        loaded = False
        if stats_path and os.path.exists(stats_path):
            self.stats = np.load(stats_path)
            # Validate keys exist before accessing them
            required_keys = ["skel_mean", "skel_std", "audio_mean", "audio_std"]
            if all(key in self.stats for key in required_keys):
                self.skel_mean = self.stats["skel_mean"]
                self.skel_std = self.stats["skel_std"]
                self.audio_mean = self.stats["audio_mean"]
                self.audio_std = self.stats["audio_std"]
                loaded = True

        if not loaded:
            if split == "train":
                # Compute stats if training and not provided
                self._compute_global_stats(stats_path)
            else:
                # For val/test, if stats not provided, we warn or use defaults (zeros/ones)
                # Ideally, stats should be passed from training set
                self.skel_mean = np.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
                self.skel_std = np.ones(SKELETON_JOINTS * SKELETON_CHANNELS)
                self.audio_mean = np.zeros(AUDIO_N_MFCC)
                self.audio_std = np.ones(AUDIO_N_MFCC)

    def _compute_global_stats(self, save_path):
        print("Computing global statistics on training data...")
        skel_sum = np.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
        skel_sq_sum = np.zeros(SKELETON_JOINTS * SKELETON_CHANNELS)
        skel_count = 0

        audio_sum = np.zeros(AUDIO_N_MFCC)
        audio_sq_sum = np.zeros(AUDIO_N_MFCC)
        audio_count = 0

        # Use a subset to speed up if dataset is huge, but here it's small enough
        for idx in range(len(self.metadata)):
            data = self._load_sample(idx)
            if data is None:
                continue

            skel = data["skeleton"]
            audio = data["audio"]

            skel_sum += np.sum(skel, axis=0)
            skel_sq_sum += np.sum(skel**2, axis=0)
            skel_count += skel.shape[0]

            audio_sum += np.sum(audio, axis=0)
            audio_sq_sum += np.sum(audio**2, axis=0)
            audio_count += audio.shape[0]

        self.skel_mean = skel_sum / skel_count
        self.skel_std = np.sqrt((skel_sq_sum / skel_count) - (self.skel_mean**2)) + 1e-6

        self.audio_mean = audio_sum / audio_count
        self.audio_std = (
            np.sqrt((audio_sq_sum / audio_count) - (self.audio_mean**2)) + 1e-6
        )

        if save_path:
            np.savez(
                save_path,
                skel_mean=self.skel_mean,
                skel_std=self.skel_std,
                audio_mean=self.audio_mean,
                audio_std=self.audio_std,
            )
            print(f"Statistics saved to {save_path}")

    def _load_sample(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")

        # 1. Try Loading from Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return dict(data)
            except Exception as e:
                print(
                    f"Warning: Corrupt cache for {sample_id}, reprocessing. Error: {e}"
                )

        # 2. Process from Raw Files
        try:
            # --- Skeleton Processing ---
            mat_path = os.path.join(INPUT_DIR, row["data_path"])
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = video.NumFrames
            fps = getattr(video, "FrameRate", 20.0)

            # Extract Skeleton
            # Structure: Video.Frames is array of structs. Each has Skeleton.
            # We assume single user or take first user.
            skeletons = []
            frames_data = video.Frames

            # Handle case where Frames is a single object or array
            if not isinstance(frames_data, np.ndarray):
                frames_data = [frames_data]

            for f_idx, frame_obj in enumerate(frames_data):
                if f_idx >= num_frames:
                    break

                # Default zero pose
                pose = np.zeros((SKELETON_JOINTS, 3))

                if hasattr(frame_obj, "Skeleton"):
                    skel_obj = frame_obj.Skeleton
                    # If multiple skeletons, take first
                    if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                        skel_obj = skel_obj[0]

                    if hasattr(skel_obj, "WorldPosition"):
                        # WorldPosition might be array of structs or struct of arrays
                        # Based on prompt: WorldPosition has X, Y, Z
                        # Assuming JointsType order matches standard Kinect (HipCenter at 0)
                        # We need to iterate 20 joints.
                        # Since parsing complex mat structures blindly is hard, we rely on
                        # the fact that usually WorldPosition is 20x1 struct array or similar.
                        # Let's try to extract coordinates directly if possible.
                        # If WorldPosition is a single struct with X,Y,Z, it's one joint.
                        # If it's an array of 20 structs, we iterate.

                        # Robust extraction strategy:
                        # Assuming skel_obj has fields like 'JointsType' and 'WorldPosition'
                        # which are arrays of length 20.
                        if hasattr(skel_obj, "WorldPosition") and isinstance(
                            skel_obj.WorldPosition, np.ndarray
                        ):
                            w_pos = skel_obj.WorldPosition
                            if len(w_pos) == SKELETON_JOINTS:
                                for j in range(SKELETON_JOINTS):
                                    joint = w_pos[j]
                                    pose[j] = [joint.X, joint.Y, joint.Z]

                skeletons.append(pose)

            # Pad or Trim to NumFrames
            if len(skeletons) < num_frames:
                # Pad with last frame
                last_pose = (
                    skeletons[-1]
                    if len(skeletons) > 0
                    else np.zeros((SKELETON_JOINTS, 3))
                )
                skeletons.extend([last_pose] * (num_frames - len(skeletons)))
            elif len(skeletons) > num_frames:
                skeletons = skeletons[:num_frames]

            skeletons = np.array(skeletons)  # (T, 20, 3)

            # Relative Coordinates: Subtract HipCenter (Index 0)
            hip_center = skeletons[:, 0:1, :]  # (T, 1, 3)
            skeletons = skeletons - hip_center

            # Flatten: (T, 60)
            skeleton_features = skeletons.reshape(num_frames, -1)

            # --- Audio Processing ---
            audio_features = np.zeros((num_frames, AUDIO_N_MFCC))
            if pd.notna(row["audio_path"]):
                audio_path = os.path.join(INPUT_DIR, row["audio_path"])

                # Load with torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Resample to 16000Hz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    )
                    waveform = resampler(waveform)
                    sample_rate = 16000

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Physics-based alignment
                hop_length = int(sample_rate / fps)
                if hop_length < 1:
                    hop_length = 1

                # Compute MFCC
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=AUDIO_N_MFCC,
                    melkwargs={
                        "n_fft": 2048,
                        "hop_length": hop_length,
                        "n_mels": 128,
                        "center": True,
                    },
                )

                mfcc = mfcc_transform(waveform)
                mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (T_audio, n_mfcc)

                # Align lengths
                if mfcc.shape[0] > num_frames:
                    mfcc = mfcc[:num_frames, :]
                elif mfcc.shape[0] < num_frames:
                    pad_len = num_frames - mfcc.shape[0]
                    padding = np.zeros((pad_len, AUDIO_N_MFCC))
                    mfcc = np.vstack([mfcc, padding])

                audio_features = mfcc

            # --- Label Processing ---
            # Create dense frame-wise labels
            dense_labels = np.zeros(num_frames, dtype=int)  # Default 0 (background)

            # Only extract labels if they exist in the mat file (Training data)
            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]
                elif raw_labels.size == 0:
                    raw_labels = []

                for l in raw_labels:
                    # Check if valid label object
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in LABEL_MAP:
                            lid = LABEL_MAP[name]
                            # MATLAB is 1-based, convert to 0-based
                            start_f = max(0, int(l.Begin) - 1)
                            end_f = min(num_frames, int(l.End))
                            dense_labels[start_f:end_f] = lid

            # --- Save to Cache ---
            data = {
                "skeleton": skeleton_features.astype(np.float32),
                "audio": audio_features.astype(np.float32),
                "labels": dense_labels.astype(np.int64),
            }
            np.savez(cache_path, **data)
            return data

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self._load_sample(idx)

        # Handle corruption or error by returning a zero-sample (or could throw error)
        if data is None:
            # Return dummy data of length 1
            return {
                "skeleton": torch.zeros(
                    (1, SKELETON_JOINTS * SKELETON_CHANNELS), dtype=torch.float32
                ),
                "audio": torch.zeros((1, AUDIO_N_MFCC), dtype=torch.float32),
                "labels": torch.zeros((1,), dtype=torch.long),
                "mask": torch.ones((1,), dtype=torch.bool),
            }

        skel = data["skeleton"]
        audio = data["audio"]
        labels = data["labels"]

        # 1. Normalization
        skel = (skel - self.skel_mean) / self.skel_std
        audio = (audio - self.audio_mean) / self.audio_std

        # Convert to Tensor
        skel = torch.from_numpy(skel).float()
        audio = torch.from_numpy(audio).float()
        labels = torch.from_numpy(labels).long()

        # 2. Augmentation (Training Only)
        if self.transform:
            # A. Gaussian Noise on Skeleton
            if np.random.rand() < 0.5:
                noise = torch.randn_like(skel) * 0.05
                skel = skel + noise

            # B. Channel Masking
            if np.random.rand() < 0.3:
                # Mask Skeleton Channels
                mask_idx = np.random.choice(
                    skel.shape[1], size=int(skel.shape[1] * 0.1), replace=False
                )
                skel[:, mask_idx] = 0
                # Mask Audio Channels
                mask_idx_a = np.random.choice(
                    audio.shape[1], size=int(audio.shape[1] * 0.1), replace=False
                )
                audio[:, mask_idx_a] = 0

            # C. Temporal Cutout
            if np.random.rand() < 0.3:
                seq_len = skel.shape[0]
                if seq_len > 20:
                    cut_len = np.random.randint(5, 15)
                    start = np.random.randint(0, seq_len - cut_len)
                    skel[start : start + cut_len, :] = 0
                    audio[start : start + cut_len, :] = 0

        return {"skeleton": skel, "audio": audio, "labels": labels}


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    """
    skeletons = [item["skeleton"] for item in batch]
    audios = [item["audio"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences (Batch, Time, Feat)
    # batch_first=True -> (B, T, C)
    skel_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=BACKGROUND_CLASS_ID
    )

    # Create Padding Mask (Batch, Time)
    # 1 for valid data, 0 for padding
    lengths = torch.tensor([len(s) for s in skeletons])
    max_len = skel_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return {
        "skeleton": skel_padded,
        "audio": audio_padded,
        "labels": labels_padded,
        "mask": mask,
        "lengths": lengths,
    }


def get_dataloaders(stats_path=None):
    """
    Factory function to create dataloaders.
    """
    set_seed(SEED)

    # Paths
    train_meta = os.path.join(os.path.dirname(INPUT_DIR), "metadata/train.csv")
    val_meta = os.path.join(os.path.dirname(INPUT_DIR), "metadata/val.csv")
    test_meta = os.path.join(os.path.dirname(INPUT_DIR), "metadata/test.csv")

    # Stats path
    if stats_path is None:
        stats_path = os.path.join(os.path.dirname(CACHE_DIR), "stats.npz")

    # Datasets
    # Train computes stats if missing
    train_dataset = GestureDataset(
        metadata_path=train_meta, split="train", transform=True, stats_path=stats_path
    )

    # Val/Test use computed stats
    val_dataset = GestureDataset(
        metadata_path=val_meta, split="val", transform=False, stats_path=stats_path
    )

    test_dataset = GestureDataset(
        metadata_path=test_meta, split="test", transform=False, stats_path=stats_path
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
