import os
import glob
import numpy as np
import pandas as pd
import torch
import scipy.io
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from library.config import Config


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split="train",
        load_cached_data=True,
        stats=None,
        debug_limit=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching.
            stats (dict): Dictionary containing 'mean' and 'std' for normalization.
            debug_limit (int): Limit dataset size for debugging.
        """
        self.split = split
        self.load_cached_data = load_cached_data
        self.stats = stats

        # Load metadata
        self.df = pd.read_csv(metadata_path)
        if debug_limit:
            self.df = self.df.iloc[:debug_limit]

        # Ensure cache directory exists
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-instantiate audio transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.HOP_LENGTH,
                "center": False,
            },
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Load Data (Cache or Compute)
        data = None
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                skeleton = data["skeleton"]
                audio = data["audio"]
                labels = data["labels"]
            except Exception:
                data = None

        if data is None:
            skeleton, audio, labels = self._process_sample(row)
            if self.load_cached_data:
                np.savez_compressed(
                    cache_path, skeleton=skeleton, audio=audio, labels=labels
                )

        # 2. Augmentation (Train only)
        if self.split == "train":
            skeleton, audio, labels = self._augment(skeleton, audio, labels)

        # 3. Normalization
        if self.stats:
            skeleton = (skeleton - self.stats["skel_mean"]) / (
                self.stats["skel_std"] + 1e-6
            )
            audio = (audio - self.stats["audio_mean"]) / (
                self.stats["audio_std"] + 1e-6
            )

        # Convert to tensors
        return {
            "skeleton": torch.FloatTensor(skeleton),
            "audio": torch.FloatTensor(audio),
            "labels": torch.LongTensor(labels),
            "length": torch.LongTensor([len(labels)]),
        }

    def _process_sample(self, row):
        # Paths
        if pd.isna(row["data_path"]):
            num_frames = int(row["num_frames"]) if pd.notna(row["num_frames"]) else 100
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS * 3), dtype=np.float32
            )
            audio_features = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)
            frame_labels = np.zeros(num_frames, dtype=np.int64)
            return skeleton_data, audio_features, frame_labels

        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(Config.INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )

        # --- Process Skeleton & Labels ---
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = video.NumFrames

            # Extract Skeleton
            # Shape: (NumFrames, NumJoints, 3) -> Flatten to (NumFrames, 60)
            # We need to handle the nested structure carefully
            raw_frames = video.Frames

            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS * 3), dtype=np.float32
            )

            # Check if Frames is populated
            if isinstance(raw_frames, np.ndarray) and len(raw_frames) > 0:
                # We iterate through frames. This can be slow, but it's cached.
                # Optimization: Vectorized extraction is hard due to object array structure in scipy.io loadmat
                for i, frame in enumerate(raw_frames):
                    if i >= num_frames:
                        break
                    if hasattr(frame, "Skeleton"):
                        skel = frame.Skeleton
                        # Handle multiple users: take the first one or the one with UserIndex
                        if isinstance(skel, np.ndarray):
                            if skel.size > 0:
                                skel = skel[0]
                            else:
                                continue  # Empty skeleton

                        if hasattr(skel, "WorldPosition"):
                            # WorldPosition might be an array of structs or struct of arrays
                            # Based on prompt: "JointsType... WorldPosition... X, Y, Z"
                            # Assuming standard Kinect order where HipCenter is index 0

                            # We need to extract 20 joints.
                            # Let's assume WorldPosition is a struct with x,y,z or an array of such structs
                            # If skel has 'WorldPosition', let's inspect it.
                            # Usually in these datasets, WorldPosition is 20x1 struct array or similar.

                            # Fallback: If we can't parse easily, we return zeros.
                            # But we must try.
                            # Let's assume we can get a (20, 3) array.

                            # Construct joint array
                            joints = np.zeros((20, 3), dtype=np.float32)

                            # If WorldPosition is an array of objects
                            if hasattr(skel, "WorldPosition") and isinstance(
                                skel.WorldPosition, np.ndarray
                            ):
                                wps = skel.WorldPosition
                                if len(wps) == 20:
                                    for j in range(20):
                                        joints[j, 0] = wps[j].X
                                        joints[j, 1] = wps[j].Y
                                        joints[j, 2] = wps[j].Z

                            # Root Relative: Subtract HipCenter (Index 0)
                            root = joints[0].copy()
                            joints -= root

                            skeleton_data[i] = joints.flatten()

        except Exception as e:
            # Fallback for broken files
            # print(f"Error parsing MAT {mat_path}: {e}")
            num_frames = row["num_frames"] if pd.notna(row["num_frames"]) else 100
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS * 3), dtype=np.float32
            )

        # Extract Labels (Frame-wise)
        frame_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

        # Only process labels if they exist (Train/Val)
        if hasattr(video, "Labels"):
            lbls = video.Labels
            if not isinstance(lbls, np.ndarray):
                lbls = [lbls] if lbls else []
            elif lbls.size == 1:
                lbls = [lbls.item()]

            for l in lbls:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in Config.LABEL_MAP:
                        lid = Config.LABEL_MAP[name]
                        # MATLAB is 1-based, Python 0-based
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))
                        frame_labels[start:end] = lid

        # --- Process Audio ---
        audio_features = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)
        if audio_path and os.path.exists(audio_path):
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Mix to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample if necessary
                if sample_rate != Config.AUDIO_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        sample_rate, Config.AUDIO_SAMPLE_RATE
                    )
                    waveform = resampler(waveform)

                # Extract MFCC
                # Output shape: (1, n_mfcc, time)
                mfcc = self.mfcc_transform(waveform)
                mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (time, n_mfcc)

                # Align to Video Frames
                # We interpolate MFCC time axis to match num_frames
                if mfcc.shape[0] != num_frames:
                    if mfcc.shape[0] > 1:
                        x_old = np.linspace(0, 1, mfcc.shape[0])
                        x_new = np.linspace(0, 1, num_frames)
                        f = interp1d(
                            x_old, mfcc, axis=0, kind="linear", fill_value="extrapolate"
                        )
                        audio_features = f(x_new).astype(np.float32)
                    else:
                        # If audio is too short/empty, pad zeros
                        audio_features = np.zeros(
                            (num_frames, Config.N_MFCC), dtype=np.float32
                        )
                else:
                    audio_features = mfcc

            except Exception as e:
                # print(f"Error processing audio {audio_path}: {e}")
                pass

        return skeleton_data, audio_features, frame_labels

    def _augment(self, skeleton, audio, labels):
        # 1. Global Temporal Resampling
        # Scale length by alpha ~ U(0.8, 1.2)
        alpha = np.random.uniform(
            Config.TEMPORAL_RESAMPLE_MIN, Config.TEMPORAL_RESAMPLE_MAX
        )
        orig_len = len(labels)
        new_len = int(orig_len * alpha)

        if new_len != orig_len and new_len > 0:
            x_old = np.linspace(0, 1, orig_len)
            x_new = np.linspace(0, 1, new_len)

            # Interpolate Skeleton
            f_skel = interp1d(
                x_old, skeleton, axis=0, kind="linear", fill_value="extrapolate"
            )
            skeleton = f_skel(x_new).astype(np.float32)

            # Interpolate Audio
            f_audio = interp1d(
                x_old, audio, axis=0, kind="linear", fill_value="extrapolate"
            )
            audio = f_audio(x_new).astype(np.float32)

            # Interpolate Labels (Nearest Neighbor)
            f_lbl = interp1d(
                x_old, labels, axis=0, kind="nearest", fill_value="extrapolate"
            )
            labels = f_lbl(x_new).astype(np.int64)

        # 2. Random Channel Masking
        # Mask Skeleton channels
        if np.random.rand() < Config.CHANNEL_MASK_PROB:
            mask = np.random.rand(skeleton.shape[1]) > 0.1  # Drop 10%
            skeleton = skeleton * mask

        # Mask Audio channels
        if np.random.rand() < Config.CHANNEL_MASK_PROB:
            mask = np.random.rand(audio.shape[1]) > 0.1
            audio = audio * mask

        return skeleton, audio, labels


def collate_fn(batch):
    # Sort by length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x["length"], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels = [x["labels"] for x in batch]
    lengths = torch.cat([x["length"] for x in batch])

    # Pad sequences
    # Pad with 0.0 for features
    padded_skeletons = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0.0
    )
    padded_audios = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0.0
    )

    # Pad labels with Background Class ID (0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return {
        "skeleton": padded_skeletons,
        "audio": padded_audios,
        "labels": padded_labels,
        "length": lengths,
    }


def compute_stats(dataset):
    """Computes mean and std for normalization."""
    print("Computing dataset statistics...")
    skel_sum = np.zeros(Config.SKELETON_CHANNELS)
    skel_sq_sum = np.zeros(Config.SKELETON_CHANNELS)
    audio_sum = np.zeros(Config.N_MFCC)
    audio_sq_sum = np.zeros(Config.N_MFCC)
    count = 0

    # Iterate through dataset (without augmentation/normalization)
    # We temporarily disable stats to get raw data
    original_stats = dataset.stats
    dataset.stats = None

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    for batch in loader:
        # Shapes: (1, T, C)
        s = batch["skeleton"].squeeze(0).numpy()
        a = batch["audio"].squeeze(0).numpy()

        skel_sum += np.sum(s, axis=0)
        skel_sq_sum += np.sum(s**2, axis=0)

        audio_sum += np.sum(a, axis=0)
        audio_sq_sum += np.sum(a**2, axis=0)

        count += s.shape[0]

    dataset.stats = original_stats

    skel_mean = skel_sum / count
    skel_std = np.sqrt((skel_sq_sum / count) - (skel_mean**2))

    audio_mean = audio_sum / count
    audio_std = np.sqrt((audio_sq_sum / count) - (audio_mean**2))

    return {
        "skel_mean": skel_mean,
        "skel_std": skel_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }


def get_dataloaders(debug_subset_size=None):
    """
    Creates DataLoaders for train, val, and test.
    Handles stats computation and caching.
    """
    stats_path = os.path.join(Config.WORKING_DIR, "stats.npz")

    # 1. Initialize Train Dataset to compute stats (if needed)
    train_ds_raw = GestureDataset(
        Config.TRAIN_METADATA_PATH,
        split="train",
        load_cached_data=True,
        stats=None,
        debug_limit=debug_subset_size,
    )

    stats = None
    if os.path.exists(stats_path):
        print(f"Loading stats from {stats_path}")
        loaded = np.load(stats_path)
        temp_stats = {k: loaded[k] for k in loaded}

        # Validate shapes to ensure compatibility with current Config (Cite debug_lesson_3)
        valid_skel = "skel_mean" in temp_stats and temp_stats["skel_mean"].shape == (
            Config.SKELETON_CHANNELS,
        )
        valid_audio = "audio_mean" in temp_stats and temp_stats["audio_mean"].shape == (
            Config.N_MFCC,
        )

        if valid_skel and valid_audio:
            stats = temp_stats
        else:
            print("Cached stats shape mismatch. Recomputing...")

    if stats is None:
        stats = compute_stats(train_ds_raw)
        np.savez(stats_path, **stats)
        print(f"Stats computed and saved to {stats_path}")

    # 2. Create Datasets with Stats
    train_ds = GestureDataset(
        Config.TRAIN_METADATA_PATH,
        split="train",
        load_cached_data=True,
        stats=stats,
        debug_limit=debug_subset_size,
    )
    val_ds = GestureDataset(
        Config.VAL_METADATA_PATH,
        split="val",
        load_cached_data=True,
        stats=stats,
        debug_limit=debug_subset_size,
    )
    test_ds = GestureDataset(
        Config.TEST_METADATA_PATH,
        split="test",
        load_cached_data=True,
        stats=stats,
        debug_limit=debug_subset_size,
    )

    # 3. Create Loaders
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
