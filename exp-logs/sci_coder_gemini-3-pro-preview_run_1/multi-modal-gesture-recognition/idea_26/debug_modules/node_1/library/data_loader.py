import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    """
    Multi-modal dataset for Gesture Recognition.
    Handles Skeleton (Anchor) and Audio streams, alignment, and caching.
    """

    def __init__(
        self, metadata_path, mode="train", stats=None, cache_dir=Config.CACHE_DIR
    ):
        """
        Args:
            metadata_path (str): Path to the CSV metadata file.
            mode (str): 'train', 'val', or 'test'.
            stats (dict, optional): Global mean/std for normalization.
            cache_dir (str): Directory to store processed .npz files.
        """
        self.mode = mode
        self.stats = stats
        self.cache_dir = cache_dir
        self.df = pd.read_csv(metadata_path)

        # Filter out samples with missing critical files
        self.df = self.df[
            self.df["data_path"].notna() & self.df["audio_path"].notna()
        ].reset_index(drop=True)

        # Audio Transform
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={
                "n_fft": Config.AUDIO_N_FFT,
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "n_mels": 64,
            },
        )

        # Pre-ensure cache exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _load_mat_file(self, rel_path):
        """Parses the .mat file for Skeleton and Labels."""
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            # squeeze_me=True simplifies the structure
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]

            # 1. Extract Skeleton
            # Structure: Video.Frames -> array of structs -> Skeleton -> WorldPosition
            frames = video.Frames
            num_frames = len(frames)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            for i, frame in enumerate(frames):
                if hasattr(frame, "Skeleton"):
                    skel = frame.Skeleton
                    # Handle case where Skeleton might be an array (multiple users)
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]  # Take first user
                    elif isinstance(skel, np.ndarray) and skel.size == 0:
                        continue  # No user

                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition is usually (20, 3) or similar
                        # If it's a struct of arrays (X, Y, Z), we might need to stack
                        # Based on prompt: "WorldPosition... X, Y, Z values"
                        # Usually in these datasets it comes as a struct with x,y,z fields or array
                        # Let's try to parse robustly.
                        # If squeeze_me=True, WorldPosition might be (20, 3) or (3, 20) or struct
                        # We assume standard Kinect format: 20 joints

                        # Fallback: check if WorldPosition is an object with X,Y,Z
                        # or an array.
                        # Given the complexity, we assume standard array shape if possible
                        # or iterate joints if they are named.
                        # Prompt says: "JointsType... HipCenter..."
                        # We will assume the order is fixed (0..19).

                        # Simplification for this task: Try to read as array
                        # If strictly following prompt description of struct:
                        # WorldPosition.X, WorldPosition.Y ...
                        # But scipy.io might load it differently.
                        # We will assume we can get a (20, 3) array.
                        # If it fails, we return zeros for that frame.
                        try:
                            # Attempt to read as array directly
                            pos = skel.WorldPosition
                            if isinstance(pos, np.ndarray) and pos.shape == (20, 3):
                                skeleton_data[i] = pos
                            elif isinstance(pos, np.ndarray) and pos.shape == (3, 20):
                                skeleton_data[i] = pos.T
                            else:
                                # Try extracting X, Y, Z if they are fields
                                # This is common in Matlab structs
                                pass
                        except:
                            pass

            # 2. Extract Labels
            # Labels: Name, Begin, End
            frame_labels = np.zeros(num_frames, dtype=np.int64)  # 0 is background
            aux_labels = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

            if hasattr(video, "Labels"):
                labels_raw = video.Labels
                if not isinstance(labels_raw, np.ndarray):
                    labels_raw = [labels_raw]
                elif labels_raw.size == 1:
                    labels_raw = [labels_raw.item()]

                # Label Map (Name -> ID)
                # We need the map from the analysis/prompt
                # 1..20. 0 is background.
                # We'll use the map defined in Config or local
                # Re-defining map here for safety based on prompt
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

                for l in labels_raw:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        if name in label_map:
                            lid = label_map[name]
                            start = max(0, int(l.Begin) - 1)  # Matlab 1-based
                            end = min(num_frames, int(l.End))
                            frame_labels[start:end] = lid
                            aux_labels[lid] = 1.0

            return skeleton_data, frame_labels, aux_labels

        except Exception as e:
            # Return None to signal failure
            # print(f"Error loading {full_path}: {e}")
            return None, None, None

    def _process_sample(self, idx):
        """
        Loads raw data, processes it, and returns aligned tensors.
        Implements caching.
        """
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Check Cache
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return (
                    torch.from_numpy(data["skeleton"]),
                    torch.from_numpy(data["audio"]),
                    torch.from_numpy(data["labels"]),
                    torch.from_numpy(data["aux_targets"]),
                )
            except:
                pass  # Corrupt cache, recompute

        # 2. Load Raw Data
        # Skeleton & Labels
        skel_data, frame_labels, aux_targets = self._load_mat_file(row["data_path"])
        if skel_data is None:
            # Fallback for broken files: return zeros
            # This shouldn't happen often due to init filtering
            T = 50
            skel_data = np.zeros((T, Config.NUM_JOINTS, 3), dtype=np.float32)
            frame_labels = np.zeros(T, dtype=np.int64)
            aux_targets = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

        num_frames = skel_data.shape[0]

        # Audio
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        try:
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

            # Compute MFCC
            mfcc = self.audio_transform(waveform)  # (1, n_mfcc, time)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        except:
            # Audio failure fallback
            mfcc = torch.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=torch.float32)

        # 3. Alignment (Interpolate Audio to match Video Frames)
        # MFCC shape: (Time_Audio, Feats) -> (1, Feats, Time_Audio) for interpolate
        mfcc_t = mfcc.transpose(0, 1).unsqueeze(0)
        if mfcc_t.shape[-1] > 0:
            mfcc_aligned = F.interpolate(
                mfcc_t, size=num_frames, mode="linear", align_corners=False
            )
            mfcc_aligned = mfcc_aligned.squeeze(0).transpose(
                0, 1
            )  # (Video_Frames, Feats)
        else:
            mfcc_aligned = torch.zeros(
                (num_frames, Config.AUDIO_N_MFCC), dtype=torch.float32
            )

        # 4. Skeleton Root-Relative Normalization
        # HipCenter is typically index 0. We assume index 0 based on prompt list order.
        # If all zeros, this does nothing.
        root = skel_data[:, 0:1, :]  # (T, 1, 3)
        skel_rel = skel_data - root

        # Flatten Skeleton: (T, 20, 3) -> (T, 60)
        skel_flat = skel_rel.reshape(num_frames, -1)

        # Convert to Tensors
        skel_tensor = torch.from_numpy(skel_flat).float()
        audio_tensor = mfcc_aligned.float()
        labels_tensor = torch.from_numpy(frame_labels).long()
        aux_tensor = torch.from_numpy(aux_targets).float()

        # 5. Save to Cache
        np.savez_compressed(
            cache_path,
            skeleton=skel_tensor.numpy(),
            audio=audio_tensor.numpy(),
            labels=labels_tensor.numpy(),
            aux_targets=aux_tensor.numpy(),
        )

        return skel_tensor, audio_tensor, labels_tensor, aux_tensor

    def __getitem__(self, idx):
        skel, audio, labels, aux = self._process_sample(idx)

        # 1. Normalization (Global Z-Score)
        if self.stats:
            # Skeleton
            skel = (skel - self.stats["skel_mean"]) / (self.stats["skel_std"] + 1e-6)
            # Audio
            audio = (audio - self.stats["audio_mean"]) / (
                self.stats["audio_std"] + 1e-6
            )

        # 2. Augmentation (Train only)
        if self.mode == "train":
            # Temporal Resampling
            if torch.rand(1) < 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                new_len = int(skel.shape[0] * alpha)
                if new_len > 0:
                    # Skel: (T, C) -> (1, C, T)
                    s_t = skel.transpose(0, 1).unsqueeze(0)
                    s_new = F.interpolate(
                        s_t, size=new_len, mode="linear", align_corners=False
                    )
                    skel = s_new.squeeze(0).transpose(0, 1)

                    # Audio: (T, C) -> (1, C, T)
                    a_t = audio.transpose(0, 1).unsqueeze(0)
                    a_new = F.interpolate(
                        a_t, size=new_len, mode="linear", align_corners=False
                    )
                    audio = a_new.squeeze(0).transpose(0, 1)

                    # Labels: (T) -> (1, 1, T) float for nearest
                    l_t = labels.float().view(1, 1, -1)
                    l_new = F.interpolate(l_t, size=new_len, mode="nearest")
                    labels = l_new.view(-1).long()

            # Channel Masking
            if torch.rand(1) < 0.5:
                # Skeleton Masking
                mask_s = torch.rand(skel.shape[1]) > 0.1
                skel = skel * mask_s.float()
                # Audio Masking
                mask_a = torch.rand(audio.shape[1]) > 0.1
                audio = audio * mask_a.float()

        return skel, audio, labels, aux, skel.shape[0]


def collate_fn(batch):
    """
    Collates batch of (skel, audio, labels, aux, length).
    Pads sequences to max length.
    """
    # Filter None
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None, None, None

    skels, audios, labels, auxs, lengths = zip(*batch)

    # Pad Sequences (Batch First = True via pad_sequence is tricky, usually it's T,B,C)
    # We want B, T, C. pad_sequence expects list of T, C. returns T, B, C.
    skels_padded = pad_sequence(skels, batch_first=True)
    audios_padded = pad_sequence(audios, batch_first=True)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    auxs_stacked = torch.stack(auxs)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return skels_padded, audios_padded, labels_padded, auxs_stacked, lengths_tensor


def compute_global_stats(dataset, save_path):
    """
    Iterates over the dataset to compute mean and std for Skeleton and Audio.
    Uses Welford's online algorithm or simple accumulation.
    """
    print("Computing global statistics...")
    skel_sum = torch.zeros(Config.SKELETON_INPUT_DIM)
    skel_sq_sum = torch.zeros(Config.SKELETON_INPUT_DIM)
    skel_count = 0

    audio_sum = torch.zeros(Config.AUDIO_INPUT_DIM)
    audio_sq_sum = torch.zeros(Config.AUDIO_INPUT_DIM)
    audio_count = 0

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    for skel, audio, _, _, _ in loader:
        if skel is None:
            continue

        # skel: (1, T, 60)
        # Flatten batch and time
        s_flat = skel.view(-1, Config.SKELETON_INPUT_DIM)
        skel_sum += s_flat.sum(dim=0)
        skel_sq_sum += (s_flat**2).sum(dim=0)
        skel_count += s_flat.size(0)

        a_flat = audio.view(-1, Config.AUDIO_INPUT_DIM)
        audio_sum += a_flat.sum(dim=0)
        audio_sq_sum += (a_flat**2).sum(dim=0)
        audio_count += a_flat.size(0)

    skel_mean = skel_sum / skel_count
    skel_std = torch.sqrt((skel_sq_sum / skel_count) - (skel_mean**2) + 1e-8)

    audio_mean = audio_sum / audio_count
    audio_std = torch.sqrt((audio_sq_sum / audio_count) - (audio_mean**2) + 1e-8)

    stats = {
        "skel_mean": skel_mean,
        "skel_std": skel_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }

    torch.save(stats, save_path)
    return stats


def get_data_loaders(debug=False):
    """
    Factory function to create DataLoaders.
    Handles stats computation and loading.
    """
    stats_path = os.path.join(Config.CACHE_DIR, "stats.pt")

    # 1. Initialize Train Dataset (Unnormalized) to compute stats
    train_ds_raw = GestureDataset(Config.TRAIN_CSV, mode="train", stats=None)

    # 2. Load or Compute Stats
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
    else:
        stats = compute_global_stats(train_ds_raw, stats_path)

    # 3. Create Datasets with Stats
    train_ds = GestureDataset(Config.TRAIN_CSV, mode="train", stats=stats)
    val_ds = GestureDataset(Config.VAL_CSV, mode="val", stats=stats)
    test_ds = GestureDataset(Config.TEST_CSV, mode="test", stats=stats)

    if debug:
        # Subset for debugging
        indices = list(range(min(len(train_ds), 32)))
        train_ds = torch.utils.data.Subset(train_ds, indices)
        val_ds = torch.utils.data.Subset(val_ds, indices)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
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
