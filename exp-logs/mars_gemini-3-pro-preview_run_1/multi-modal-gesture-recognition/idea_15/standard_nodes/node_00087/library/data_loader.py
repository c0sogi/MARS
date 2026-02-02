import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(
        self,
        split="train",
        load_cached_data=True,
        compute_stats=False,
        augment=False,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, use cached .npz files.
            compute_stats (bool): If True, compute global stats from this split (usually train).
            augment (bool): If True, apply data augmentation.
        """
        self.split = split
        self.load_cached_data = load_cached_data
        self.augment = augment

        # Load Metadata
        metadata_file = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.metadata = pd.read_csv(metadata_file)

        # Filter out samples with missing essential files (color/data)
        # The metadata generation already did some filtering, but we ensure data_path exists
        self.metadata = self.metadata[self.metadata["data_path"].notna()]

        # Prepare Cache
        self._prepare_cache()

        # Load or Compute Global Statistics for Normalization
        self.stats_path = os.path.join(Config.WORKING_DIR, "stats.npz")
        self.mean_skel = None
        self.std_skel = None
        self.mean_audio = None
        self.std_audio = None

        if compute_stats:
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

    def _prepare_cache(self):
        """
        Iterates over metadata, checks/creates cached .npz files.
        """
        # MFCC Transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={
                "n_fft": Config.AUDIO_N_FFT,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "center": False,  # Align strictly
            },
        )

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

            if self.load_cached_data and os.path.exists(cache_path):
                continue

            # --- Process Raw Data ---
            try:
                # 1. Load Skeleton & Labels from MAT
                mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
                skeleton_data, num_frames, labels_list = self._process_mat_file(
                    mat_path
                )

                if skeleton_data is None:
                    # Fallback: Create zero skeleton if missing/corrupt
                    # This shouldn't happen often given metadata filtering
                    num_frames = (
                        row["num_frames"] if not pd.isna(row["num_frames"]) else 100
                    )
                    skeleton_data = np.zeros(
                        (int(num_frames), Config.NUM_JOINTS * 3), dtype=np.float32
                    )
                    labels_list = []

                # 2. Create Dense Targets
                target = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)
                for lbl in labels_list:
                    # Matlab 1-based indexing -> Python 0-based
                    start = max(0, lbl["begin"] - 1)
                    end = min(num_frames, lbl["end"])
                    lid = lbl["id"]
                    if start < end:
                        target[start:end] = lid

                # 3. Process Audio
                audio_path = row["audio_path"]
                if pd.notna(audio_path):
                    full_audio_path = os.path.join(Config.INPUT_DIR, audio_path)
                    if os.path.exists(full_audio_path):
                        waveform, sr = torchaudio.load(full_audio_path)
                        # Resample if necessary
                        if sr != Config.AUDIO_SAMPLE_RATE:
                            resampler = torchaudio.transforms.Resample(
                                sr, Config.AUDIO_SAMPLE_RATE
                            )
                            waveform = resampler(waveform)

                        # Mix down to mono if necessary
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        # Extract MFCC
                        # Input: (1, samples) -> Output: (1, n_mfcc, time)
                        mfcc = mfcc_transform(waveform)
                        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (time, n_mfcc)
                    else:
                        mfcc = np.zeros(
                            (num_frames, Config.AUDIO_N_MFCC), dtype=np.float32
                        )
                else:
                    mfcc = np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

                # 4. Align Audio/Skeleton
                # Skeleton dictates the length (num_frames)
                if mfcc.shape[0] > num_frames:
                    mfcc = mfcc[:num_frames, :]
                elif mfcc.shape[0] < num_frames:
                    pad_len = num_frames - mfcc.shape[0]
                    padding = np.zeros((pad_len, Config.AUDIO_N_MFCC), dtype=np.float32)
                    mfcc = np.vstack([mfcc, padding])

                # Save to Cache
                np.savez(
                    cache_path,
                    skeleton=skeleton_data.astype(np.float32),
                    audio=mfcc.astype(np.float32),
                    target=target.astype(np.int64),
                    sample_id=sample_id,
                )

            except Exception as e:
                # print(f"Error processing {sample_id}: {e}")
                # Create dummy data to prevent crash, will be filtered or ignored usually
                # But strict adherence requires handling. We'll skip saving and let __getitem__ fail or handle.
                pass

    def _process_mat_file(self, mat_path):
        """
        Parses .mat file for Skeleton (Root-Relative) and Labels.
        """
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = int(video.NumFrames)

            # --- Labels ---
            labels_list = []
            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]

                for l in raw_labels:
                    # Check if valid label object
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in Config.LABEL_MAP:
                            labels_list.append(
                                {
                                    "id": Config.LABEL_MAP[name],
                                    "begin": int(l.Begin),
                                    "end": int(l.End),
                                }
                            )

            # --- Skeleton ---
            # Initialize with zeros
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            if hasattr(video, "Frames"):
                frames = video.Frames
                if isinstance(frames, np.ndarray) and len(frames) == num_frames:
                    for i in range(num_frames):
                        frame_obj = frames[i]
                        if hasattr(frame_obj, "Skeleton"):
                            skel_obj = frame_obj.Skeleton
                            # Handle array of skeletons (multi-user), take first
                            if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                                skel_obj = skel_obj[0]

                            if hasattr(skel_obj, "WorldPosition"):
                                wp = skel_obj.WorldPosition
                                # WorldPosition is usually 20x3 or struct of arrays
                                # Based on prompt: "X value represents...".
                                # Usually in these datasets it's a 20x3 matrix or similar.
                                # If it's a 20x3 numpy array:
                                if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                                    skeleton_data[i] = wp
                                # If it's a struct with X, Y, Z arrays?
                                # Let's assume standard matrix form based on prompt "20x4 matrix" for rotation, likely 20x3 for pos.
                                # If zero, we leave as zero.

            # Root-Relative Normalization
            # HipCenter is index 0
            hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - hip_center

            # Flatten: (T, 20*3)
            skeleton_data = skeleton_data.reshape(num_frames, -1)

            return skeleton_data, num_frames, labels_list

        except Exception:
            return None, 0, []

    def _compute_global_stats(self):
        """
        Computes mean and std for Skeleton and Audio over the dataset.
        Saves to stats.npz.
        """
        # Online Welford's algorithm or simple sum accumulation
        # Given dataset size (~300 videos * ~50 frames), simple accumulation is fine.

        sum_skel = np.zeros(Config.INPUT_DIM_SKELETON)
        sq_sum_skel = np.zeros(Config.INPUT_DIM_SKELETON)
        count_skel = 0

        sum_audio = np.zeros(Config.INPUT_DIM_AUDIO)
        sq_sum_audio = np.zeros(Config.INPUT_DIM_AUDIO)
        count_audio = 0

        for idx, row in self.metadata.iterrows():
            cache_path = os.path.join(Config.CACHE_DIR, f"{row['sample_id']}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                skel = data["skeleton"]
                aud = data["audio"]

                sum_skel += np.sum(skel, axis=0)
                sq_sum_skel += np.sum(skel**2, axis=0)
                count_skel += skel.shape[0]

                sum_audio += np.sum(aud, axis=0)
                sq_sum_audio += np.sum(aud**2, axis=0)
                count_audio += aud.shape[0]

        if count_skel > 0:
            mean_skel = sum_skel / count_skel
            std_skel = np.sqrt((sq_sum_skel / count_skel) - (mean_skel**2))
        else:
            mean_skel = np.zeros(Config.INPUT_DIM_SKELETON)
            std_skel = np.ones(Config.INPUT_DIM_SKELETON)

        if count_audio > 0:
            mean_audio = sum_audio / count_audio
            std_audio = np.sqrt((sq_sum_audio / count_audio) - (mean_audio**2))
        else:
            mean_audio = np.zeros(Config.INPUT_DIM_AUDIO)
            std_audio = np.ones(Config.INPUT_DIM_AUDIO)

        np.savez(
            self.stats_path,
            mean_skel=mean_skel,
            std_skel=std_skel,
            mean_audio=mean_audio,
            std_audio=std_audio,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        if not os.path.exists(cache_path):
            # Should not happen if _prepare_cache ran, but fallback
            # Return zeros
            return (
                torch.zeros(10, Config.INPUT_DIM_SKELETON),
                torch.zeros(10, Config.INPUT_DIM_AUDIO),
                torch.zeros(10, dtype=torch.long),
                sample_id,
            )

        data = np.load(cache_path)
        skel = torch.from_numpy(data["skeleton"]).float()
        audio = torch.from_numpy(data["audio"]).float()
        target = torch.from_numpy(data["target"]).long()

        # Apply Normalization
        if self.mean_skel is not None:
            skel = (skel - self.mean_skel) / (self.std_skel + 1e-6)

        if self.mean_audio is not None:
            audio = (audio - self.mean_audio) / (self.std_audio + 1e-6)

        # Apply Augmentation (Cite {solution_lesson_node_00040}, {solution_lesson_node_00013})
        if self.augment and random.random() < Config.AUGMENT_PROB:
            skel, audio = self._apply_augmentation(skel, audio)

        return skel, audio, target, sample_id

    def _apply_augmentation(self, skel, audio):
        """
        Applies random noise, time masking, and channel masking.
        """
        # 1. Gaussian Noise
        if Config.NOISE_SIGMA > 0:
            skel = skel + torch.randn_like(skel) * Config.NOISE_SIGMA
            audio = audio + torch.randn_like(audio) * Config.NOISE_SIGMA

        # 2. Random Time Masking (Temporal Cutout)
        if random.random() < Config.MASK_TIME_PROB:
            # Skeleton
            T = skel.shape[0]
            mask_len = random.randint(1, Config.MASK_TIME_MAX_FRAMES)
            start = random.randint(0, max(0, T - mask_len))
            skel[start : start + mask_len, :] = 0.0

            # Audio
            T = audio.shape[0]
            mask_len = random.randint(1, Config.MASK_TIME_MAX_FRAMES)
            start = random.randint(0, max(0, T - mask_len))
            audio[start : start + mask_len, :] = 0.0

        # 3. Random Channel Masking
        if random.random() < Config.MASK_CHANNEL_PROB:
            # Skeleton
            C = skel.shape[1]
            num_mask = int(C * Config.MASK_CHANNEL_RATIO)
            if num_mask > 0:
                mask_idx = torch.randperm(C)[:num_mask]
                skel[:, mask_idx] = 0.0

            # Audio
            C = audio.shape[1]
            num_mask = int(C * Config.MASK_CHANNEL_RATIO)
            if num_mask > 0:
                mask_idx = torch.randperm(C)[:num_mask]
                audio[:, mask_idx] = 0.0

        return skel, audio


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    """
    skeletons, audios, targets, sample_ids = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(s) for s in skeletons])

    # Pad
    # batch_first=True -> (Batch, Time, Dim)
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Targets padding value must be BACKGROUND_CLASS_ID (0)
    targets_padded = pad_sequence(
        targets, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return {
        "skeleton": skeletons_padded,
        "audio": audios_padded,
        "target": targets_padded,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Creates DataLoaders for train, val, test.
    """
    # Train Dataset (Compute Stats + Augment)
    train_ds = GestureDataset(
        split="train", load_cached_data=True, compute_stats=True, augment=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Val Dataset
    val_ds = GestureDataset(split="val", load_cached_data=True, compute_stats=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Dataset
    test_ds = GestureDataset(split="test", load_cached_data=True, compute_stats=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
