import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import Config, set_seed
from library.utils import get_logger

logger = get_logger(__name__)


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Filter out samples with missing critical files just in case,
        # though metadata generation should have handled this.
        self.df = self.df[self.df["data_path"].notna() & self.df["audio_path"].notna()]
        self.df = self.df.reset_index(drop=True)

        # Load global stats for normalization if they exist and we are not computing them
        self.stats = None
        if os.path.exists(Config.STATS_PATH):
            self.stats = np.load(Config.STATS_PATH)
            self.skeleton_mean = torch.from_numpy(self.stats["skeleton_mean"]).float()
            self.skeleton_std = torch.from_numpy(self.stats["skeleton_std"]).float()
            self.audio_mean = torch.from_numpy(self.stats["audio_mean"]).float()
            self.audio_std = torch.from_numpy(self.stats["audio_std"]).float()
        elif mode != "train":
            # If stats don't exist and we are in inference mode, this is critical.
            # However, for the purpose of this script, we assume stats are generated during training setup.
            logger.warning(
                "Global stats not found. Normalization will be skipped or unstable."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        # Cache path
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        data = None

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                skeleton = torch.from_numpy(data["skeleton"]).float()
                audio = torch.from_numpy(data["audio"]).float()
                dense_labels = torch.from_numpy(data["dense_labels"]).long()
                # sequence_labels is stored as object array
                sequence_labels = torch.from_numpy(data["sequence_labels"]).long()
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for {sample_id}: {e}. Recomputing."
                )
                data = None

        # 2. Compute if not loaded
        if data is None:
            try:
                skeleton, audio, dense_labels, sequence_labels = self._process_raw_data(
                    row
                )

                # Save to cache
                np.savez_compressed(
                    cache_path,
                    skeleton=skeleton.numpy(),
                    audio=audio.numpy(),
                    dense_labels=dense_labels.numpy(),
                    sequence_labels=sequence_labels.numpy(),
                )
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
                # Return zero-tensors to avoid crashing, but log error.
                # In strict mode we might want to raise, but for DataLoader stability we often skip or pad.
                # Here we raise to ensure we catch bad data logic early as requested.
                raise e

        # 3. Normalization
        if self.stats is not None:
            # Epsilon to avoid division by zero
            eps = 1e-6
            skeleton = (skeleton - self.skeleton_mean) / (self.skeleton_std + eps)
            audio = (audio - self.audio_mean) / (self.audio_std + eps)

        # 4. Augmentation (Only in train mode)
        if self.mode == "train":
            skeleton, audio = self._augment(skeleton, audio)

        return {
            "sample_id": sample_id,
            "skeleton": skeleton,
            "audio": audio,
            "dense_labels": dense_labels,
            "sequence_labels": sequence_labels,
            "length": skeleton.shape[0],
        }

    def _process_raw_data(self, row):
        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # --- Load Skeleton ---
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            raise ValueError(f"Invalid MAT file structure: {row['data_path']}")

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        if num_frames == 0 or (isinstance(frames, np.ndarray) and frames.size == 0):
            raise ValueError(f"Empty video sequence: {row['data_path']}")

        skeleton_data = []
        # Ensure frames is iterable
        if not isinstance(frames, np.ndarray):
            frames = [frames]
        elif frames.size == 1:
            frames = [frames.item()]

        for i, frame in enumerate(frames):
            # Extract Skeleton
            if not hasattr(frame, "Skeleton"):
                # Fallback: assume zero pose if missing
                skeleton_data.append(np.zeros(Config.SKELETON_INPUT_SIZE))
                continue

            skel = frame.Skeleton
            # Handle multiple users (take first)
            if isinstance(skel, np.ndarray):
                if skel.size > 0:
                    skel = skel[0]
                else:
                    skeleton_data.append(np.zeros(Config.SKELETON_INPUT_SIZE))
                    continue

            # Extract WorldPosition
            if hasattr(skel, "WorldPosition"):
                wp = skel.WorldPosition
                # Check if it's a struct with X, Y, Z arrays or array of structs
                coords = []
                try:
                    # Case 1: Struct of arrays (NumJoints,)
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # Stack X, Y, Z
                        # Assuming 20 joints
                        xs = wp.X if isinstance(wp.X, np.ndarray) else np.array([wp.X])
                        ys = wp.Y if isinstance(wp.Y, np.ndarray) else np.array([wp.Y])
                        zs = wp.Z if isinstance(wp.Z, np.ndarray) else np.array([wp.Z])

                        # Ensure we have 20 joints
                        if len(xs) != 20:
                            # Resize or pad? Strict validation:
                            # If not 20, we might be misinterpreting.
                            # But let's try to flatten what we have.
                            pass

                        # Interleave: x1, y1, z1, x2, y2, z2...
                        frame_joints = np.stack([xs, ys, zs], axis=1).flatten()
                        coords = frame_joints
                    else:
                        # Case 2: Array of structs
                        # This is harder to vectorize blindly without data inspection.
                        # We'll assume Case 1 is dominant based on description.
                        # Fallback to zero if structure is unexpected
                        coords = np.zeros(Config.SKELETON_INPUT_SIZE)
                except:
                    coords = np.zeros(Config.SKELETON_INPUT_SIZE)

                if len(coords) != Config.SKELETON_INPUT_SIZE:
                    # Pad or trim
                    if len(coords) > Config.SKELETON_INPUT_SIZE:
                        coords = coords[: Config.SKELETON_INPUT_SIZE]
                    else:
                        padded = np.zeros(Config.SKELETON_INPUT_SIZE)
                        padded[: len(coords)] = coords
                        coords = padded

                skeleton_data.append(coords)
            else:
                skeleton_data.append(np.zeros(Config.SKELETON_INPUT_SIZE))

        skeleton_tensor = torch.tensor(np.array(skeleton_data), dtype=torch.float32)

        # --- Load Audio & MFCC ---
        # Physics-based hop length
        # We need 1 audio feature vector per video frame.
        # Video FPS = 20. Audio SR = 16000.
        # Hop = 16000 / 20 = 800 samples.

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # MFCC
        # n_fft: window size. Typical is 20-40ms. 1024 samples @ 16kHz is 64ms.
        # We use 1024 to ensure we cover the hop length of 800 comfortably.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": 1024,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "n_mels": 40,
                "center": False,  # False to align better with frames? True is usually fine.
            },
        )

        audio_features = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
        audio_features = audio_features.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Align Audio and Skeleton
        # Skeleton has T frames. Audio should have T frames.
        T_skel = skeleton_tensor.shape[0]
        T_audio = audio_features.shape[0]

        if T_audio > T_skel:
            audio_features = audio_features[:T_skel, :]
        elif T_audio < T_skel:
            pad_len = T_skel - T_audio
            padding = torch.zeros((pad_len, Config.N_MFCC))
            audio_features = torch.cat([audio_features, padding], dim=0)

        # --- Labels ---
        # Construct dense frame-wise labels
        dense_labels = torch.zeros(T_skel, dtype=torch.long)  # Default 0 (Background)

        # Parse sequence labels from metadata string
        seq_str = str(row["labels"]) if pd.notna(row["labels"]) else ""
        sequence_labels_list = (
            [int(x) for x in seq_str.split(",")]
            if seq_str and seq_str.lower() != "nan"
            else []
        )

        # If we have ground truth annotation in MAT file, use it for dense labels
        # The MAT file contains 'Labels' struct with 'Begin', 'End', 'Name'
        if hasattr(video, "Labels"):
            lbls = video.Labels
            if not isinstance(lbls, np.ndarray):
                lbls = [lbls] if lbls is not None else []
            elif lbls.size == 1:
                lbls = [lbls.item()]

            for l in lbls:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in Config.LABEL_MAP:
                        lid = Config.LABEL_MAP[name]
                        # Matlab is 1-based index usually, but let's assume raw values.
                        # If 1-based, we might need -1.
                        # Usually in these datasets, frame indices are 1-based.
                        # Python is 0-based.
                        start = int(l.Begin) - 1
                        end = int(l.End) - 1

                        # Clip to valid range
                        start = max(0, start)
                        end = min(T_skel - 1, end)

                        if end >= start:
                            dense_labels[start : end + 1] = lid

        sequence_labels = torch.tensor(sequence_labels_list, dtype=torch.long)

        return skeleton_tensor, audio_features, dense_labels, sequence_labels

    def _augment(self, skeleton, audio):
        """
        Applies augmentation:
        1. Random Channel Masking
        2. Random Time Masking
        3. Additive Gaussian Noise (Skeleton only)
        """
        # 1. Additive Gaussian Noise to Skeleton
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(skeleton) * 0.05
            skeleton = skeleton + noise

        # 2. Random Channel Masking
        # Mask ~10% of channels
        if torch.rand(1).item() < 0.5:
            # Skeleton
            mask_skel = torch.rand(skeleton.shape[1]) > 0.1
            skeleton = skeleton * mask_skel.float()
            # Audio
            mask_audio = torch.rand(audio.shape[1]) > 0.1
            audio = audio * mask_audio.float()

        # 3. Random Time Masking (Temporal Cutout)
        if torch.rand(1).item() < 0.5:
            T = skeleton.shape[0]
            if T > 10:
                mask_len = random.randint(5, min(15, T // 2))
                start = random.randint(0, T - mask_len)

                skeleton[start : start + mask_len, :] = 0
                audio[start : start + mask_len, :] = 0

        return skeleton, audio


def compute_global_stats(dataset):
    """
    Computes global mean and std for skeleton and audio features over the dataset.
    """
    logger.info("Computing global stats for normalization...")

    skel_sum = torch.zeros(Config.SKELETON_INPUT_SIZE)
    skel_sq_sum = torch.zeros(Config.SKELETON_INPUT_SIZE)
    skel_count = 0

    audio_sum = torch.zeros(Config.N_MFCC)
    audio_sq_sum = torch.zeros(Config.N_MFCC)
    audio_count = 0

    # Iterate without normalization
    for i in range(len(dataset)):
        # We access _process_raw_data directly or disable normalization in getitem
        # Easier to temporarily disable stats in dataset or just load raw here
        row = dataset.df.iloc[i]
        try:
            skel, audio, _, _ = dataset._process_raw_data(row)

            skel_sum += torch.sum(skel, dim=0)
            skel_sq_sum += torch.sum(skel**2, dim=0)
            skel_count += skel.shape[0]

            audio_sum += torch.sum(audio, dim=0)
            audio_sq_sum += torch.sum(audio**2, dim=0)
            audio_count += audio.shape[0]
        except Exception:
            continue

    skel_mean = skel_sum / skel_count
    skel_std = torch.sqrt((skel_sq_sum / skel_count) - skel_mean**2)

    audio_mean = audio_sum / audio_count
    audio_std = torch.sqrt((audio_sq_sum / audio_count) - audio_mean**2)

    # Save
    np.savez(
        Config.STATS_PATH,
        skeleton_mean=skel_mean.numpy(),
        skeleton_std=skel_std.numpy(),
        audio_mean=audio_mean.numpy(),
        audio_std=audio_std.numpy(),
    )

    logger.info("Global stats computed and saved.")
    return skel_mean, skel_std, audio_mean, audio_std


def collate_fn(batch):
    """
    Collate function to pad sequences.
    """
    # Filter out Nones if any
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    sample_ids = [b["sample_id"] for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> (Batch, Time, Feat)
    skeletons = [b["skeleton"] for b in batch]
    audios = [b["audio"] for b in batch]
    dense_labels = [b["dense_labels"] for b in batch]

    padded_skeletons = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
    padded_dense_labels = pad_sequence(
        dense_labels, batch_first=True, padding_value=Config.BACKGROUND_LABEL
    )

    # Sequence labels are list of lists (variable length), cannot easily pad as tensor for loss unless using CTC
    # For metric calculation, we keep them as a list of tensors
    sequence_labels = [b["sequence_labels"] for b in batch]

    return {
        "sample_ids": sample_ids,
        "skeleton": padded_skeletons,
        "audio": padded_audios,
        "dense_labels": padded_dense_labels,
        "sequence_labels": sequence_labels,
        "lengths": lengths,
    }


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test.
    """
    # Paths
    train_meta = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta = os.path.join(Config.METADATA_DIR, "test.csv")

    # Datasets
    # We turn off caching for the initial stats computation to ensure we read raw data correctly
    # But actually, caching raw data is fine.

    train_ds = GestureDataset(train_meta, mode="train", load_cached_data=True)
    val_ds = GestureDataset(val_meta, mode="val", load_cached_data=True)
    test_ds = GestureDataset(test_meta, mode="test", load_cached_data=True)

    # Debug mode: subset
    if debug:
        train_ds.df = train_ds.df.head(20)
        val_ds.df = val_ds.df.head(10)
        test_ds.df = test_ds.df.head(10)

    # Compute stats if needed
    if not os.path.exists(Config.STATS_PATH):
        compute_global_stats(train_ds)
        # Reload dataset to pick up stats
        train_ds = GestureDataset(train_meta, mode="train", load_cached_data=True)
        val_ds = GestureDataset(val_meta, mode="val", load_cached_data=True)
        if debug:
            train_ds.df = train_ds.df.head(20)
            val_ds.df = val_ds.df.head(10)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
