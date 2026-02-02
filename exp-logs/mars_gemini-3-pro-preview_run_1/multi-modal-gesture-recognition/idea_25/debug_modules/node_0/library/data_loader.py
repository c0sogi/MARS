import os
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


def load_and_process_sample(sample_id, data_path, audio_path, load_cached_data=True):
    """
    Loads and processes a single sample (Skeleton + Audio).
    Implements caching mechanism using .npz files.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return torch.from_numpy(data["skeleton"]), torch.from_numpy(data["audio"])
        except Exception:
            pass  # Fallback to computing from scratch

    # 2. Process from Scratch
    # 2.a Load Skeleton
    full_data_path = os.path.join(Config.INPUT_DIR, data_path)
    skeleton_frames = []

    try:
        mat = scipy.io.loadmat(full_data_path, struct_as_record=False, squeeze_me=True)
        if hasattr(mat["Video"], "Frames"):
            frames = mat["Video"].Frames
            # Handle case where Frames is a single object
            if not isinstance(frames, np.ndarray):
                frames = [frames]

            for f in frames:
                # Extract Skeleton
                if hasattr(f, "Skeleton"):
                    skel = f.Skeleton
                    # Handle multiple users (take first)
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    # Extract WorldPosition
                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition is usually an array of joints
                        joints = skel.WorldPosition
                        frame_joints = []

                        # Ensure joints is iterable
                        if not isinstance(joints, np.ndarray):
                            joints = [joints]

                        for j in joints:
                            # Extract X, Y, Z
                            # Sometimes j is a struct with X, Y, Z fields
                            if hasattr(j, "X") and hasattr(j, "Y") and hasattr(j, "Z"):
                                frame_joints.append([j.X, j.Y, j.Z])
                            else:
                                frame_joints.append([0.0, 0.0, 0.0])

                        # Ensure we have 20 joints
                        if len(frame_joints) < Config.SKELETON_JOINTS:
                            # Pad with zeros if missing
                            frame_joints.extend(
                                [[0.0, 0.0, 0.0]]
                                * (Config.SKELETON_JOINTS - len(frame_joints))
                            )
                        elif len(frame_joints) > Config.SKELETON_JOINTS:
                            frame_joints = frame_joints[: Config.SKELETON_JOINTS]

                        skeleton_frames.append(frame_joints)
                    else:
                        skeleton_frames.append(np.zeros((Config.SKELETON_JOINTS, 3)))
                else:
                    skeleton_frames.append(np.zeros((Config.SKELETON_JOINTS, 3)))
    except Exception as e:
        # Return empty tensors on failure
        pass

    if not skeleton_frames:
        # Create dummy data if loading failed
        skeleton_tensor = torch.zeros(
            (10, Config.SKELETON_JOINTS, 3), dtype=torch.float32
        )
    else:
        skeleton_tensor = torch.tensor(skeleton_frames, dtype=torch.float32)

    # Geometric Normalization: Root-Relative (Joint - HipCenter)
    # Assuming HipCenter is index 0 based on Kinect standard and prompt list order
    if skeleton_tensor.shape[1] > 0:
        root = skeleton_tensor[:, 0:1, :]
        skeleton_tensor = skeleton_tensor - root

    # 2.b Load Audio
    full_audio_path = os.path.join(Config.INPUT_DIR, audio_path)
    try:
        waveform, sample_rate = torchaudio.load(full_audio_path)

        # Mix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.MFCC_N_MFCC,
            melkwargs={
                "n_fft": Config.MFCC_N_FFT,
                "hop_length": Config.MFCC_HOP_LENGTH,
                "center": False,
            },
        )
        audio_features = mfcc_transform(waveform)  # (Channels, n_mfcc, Time)
        audio_features = audio_features.squeeze(0).transpose(0, 1)  # (Time, n_mfcc)

    except Exception:
        # Dummy audio
        audio_features = torch.zeros(
            (skeleton_tensor.shape[0], Config.MFCC_N_MFCC), dtype=torch.float32
        )

    # 2.c Align Modalities (Truncate to min length)
    min_len = min(skeleton_tensor.shape[0], audio_features.shape[0])
    if min_len > 0:
        skeleton_tensor = skeleton_tensor[:min_len]
        audio_features = audio_features[:min_len]
    else:
        # Handle edge case where one is empty
        max_len = max(skeleton_tensor.shape[0], audio_features.shape[0])
        if skeleton_tensor.shape[0] == 0:
            skeleton_tensor = torch.zeros((max_len, Config.SKELETON_JOINTS, 3))
        if audio_features.shape[0] == 0:
            audio_features = torch.zeros((max_len, Config.MFCC_N_MFCC))

    # 3. Save to Cache
    try:
        np.savez_compressed(
            cache_path, skeleton=skeleton_tensor.numpy(), audio=audio_features.numpy()
        )
    except Exception:
        pass

    return skeleton_tensor, audio_features


def compute_global_stats(dataset_metadata_df):
    """
    Computes global mean and std for Skeleton and Audio from the training set.
    """
    if os.path.exists(Config.STATS_PATH):
        data = np.load(Config.STATS_PATH)
        return (
            torch.from_numpy(data["skel_mean"]),
            torch.from_numpy(data["skel_std"]),
            torch.from_numpy(data["audio_mean"]),
            torch.from_numpy(data["audio_std"]),
        )

    print("Computing global statistics...")
    skel_sum = torch.zeros(3, dtype=torch.double)
    skel_sq_sum = torch.zeros(3, dtype=torch.double)
    skel_count = 0

    audio_sum = torch.zeros(Config.MFCC_N_MFCC, dtype=torch.double)
    audio_sq_sum = torch.zeros(Config.MFCC_N_MFCC, dtype=torch.double)
    audio_count = 0

    # Iterate over training data
    for idx, row in dataset_metadata_df.iterrows():
        s, a = load_and_process_sample(
            row["sample_id"], row["data_path"], row["audio_path"], load_cached_data=True
        )

        # Skeleton (flatten joints, keep channels)
        if s.numel() > 0:
            s_flat = s.view(-1, 3).double()
            skel_sum += s_flat.sum(dim=0)
            skel_sq_sum += (s_flat**2).sum(dim=0)
            skel_count += s_flat.shape[0]

        # Audio
        if a.numel() > 0:
            a_flat = a.double()
            audio_sum += a_flat.sum(dim=0)
            audio_sq_sum += (a_flat**2).sum(dim=0)
            audio_count += a_flat.shape[0]

    # Compute Mean and Std
    skel_mean = (skel_sum / skel_count).float()
    skel_std = torch.sqrt(
        (skel_sq_sum / skel_count) - (skel_mean.double() ** 2)
    ).float()
    # Avoid division by zero
    skel_std[skel_std < 1e-6] = 1.0

    audio_mean = (audio_sum / audio_count).float()
    audio_std = torch.sqrt(
        (audio_sq_sum / audio_count) - (audio_mean.double() ** 2)
    ).float()
    audio_std[audio_std < 1e-6] = 1.0

    # Save
    np.savez(
        Config.STATS_PATH,
        skel_mean=skel_mean.numpy(),
        skel_std=skel_std.numpy(),
        audio_mean=audio_mean.numpy(),
        audio_std=audio_std.numpy(),
    )

    return skel_mean, skel_std, audio_mean, audio_std


class GestureDataset(Dataset):
    def __init__(self, metadata_path, stats=None, is_train=False, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.is_train = is_train
        self.transform = transform

        # Parse labels
        self.df["labels_list"] = self.df["labels"].apply(
            lambda x: (
                [int(i) for i in str(x).split(",")] if pd.notna(x) and x != "" else []
            )
        )

        self.stats = stats  # (skel_mean, skel_std, audio_mean, audio_std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        # Load Data
        skeleton, audio = load_and_process_sample(
            sample_id, row["data_path"], row["audio_path"], load_cached_data=True
        )

        # Augmentation: Temporal Resampling
        if self.is_train:
            alpha = np.random.uniform(
                Config.TEMPORAL_RESAMPLE_MIN, Config.TEMPORAL_RESAMPLE_MAX
            )
            new_len = int(skeleton.shape[0] * alpha)
            if new_len > 0:
                # Interpolate Skeleton (T, J, C) -> (1, C*J, T) -> interpolate -> reshape
                T, J, C = skeleton.shape
                s_in = (
                    skeleton.view(T, J * C).permute(1, 0).unsqueeze(0)
                )  # (1, Feat, T)
                s_out = F.interpolate(
                    s_in, size=new_len, mode="linear", align_corners=False
                )
                skeleton = s_out.squeeze(0).permute(1, 0).view(new_len, J, C)

                # Interpolate Audio (T, F) -> (1, F, T)
                a_in = audio.permute(1, 0).unsqueeze(0)
                a_out = F.interpolate(
                    a_in, size=new_len, mode="linear", align_corners=False
                )
                audio = a_out.squeeze(0).permute(1, 0)

        # Normalization (Global Z-Score)
        if self.stats is not None:
            skel_mean, skel_std, audio_mean, audio_std = self.stats
            skeleton = (skeleton - skel_mean) / skel_std
            audio = (audio - audio_mean) / audio_std

        # Augmentation: Channel Masking
        if self.is_train and np.random.rand() < Config.CHANNEL_MASK_PROB:
            # Mask Skeleton Channels (Joints)
            mask_s = torch.bernoulli(torch.ones(skeleton.shape[1]) * 0.9)  # 10% dropped
            skeleton = skeleton * mask_s.view(1, -1, 1)

            # Mask Audio Channels (MFCCs)
            mask_a = torch.bernoulli(torch.ones(audio.shape[1]) * 0.9)
            audio = audio * mask_a.view(1, -1)

        # Prepare Labels (Frame-wise expansion for training)
        # For training, we need frame-level targets.
        # The prompt implies sequence-to-sequence or CTC.
        # Given "Identify the gestures... ordered list", and "Levenshtein",
        # usually CTC or Seq2Seq. However, the prompt mentions "For each gesture, the initial frame... provided".
        # We can construct a dense frame-wise label tensor for CrossEntropy.

        label_seq = row["labels_list"]
        # Create frame-wise labels initialized to Background
        frame_labels = torch.full(
            (skeleton.shape[0],), Config.BACKGROUND_LABEL, dtype=torch.long
        )

        # If we have detailed timing info, we could map it.
        # But load_mat_data in analysis showed we have Begin/End.
        # We need to reload that info if we want precise alignment.
        # However, for this implementation, we will rely on the fact that `load_and_process_sample`
        # doesn't return the timing info.
        # To strictly follow the prompt "Use Frame-wise Cross Entropy Loss", we need frame targets.
        # I will parse the MAT file again here briefly to get timing for labels if training.

        if self.is_train:
            try:
                mat = scipy.io.loadmat(
                    os.path.join(Config.INPUT_DIR, row["data_path"]),
                    squeeze_me=True,
                    struct_as_record=False,
                )
                if hasattr(mat["Video"], "Labels"):
                    raw_lbls = mat["Video"].Labels
                    if not isinstance(raw_lbls, np.ndarray):
                        raw_lbls = [raw_lbls]
                    elif raw_lbls.size == 1:
                        raw_lbls = [raw_lbls.item()]

                    # Scaling factor if we resampled
                    scale = skeleton.shape[0] / getattr(
                        mat["Video"], "NumFrames", skeleton.shape[0]
                    )

                    for l in raw_lbls:
                        if (
                            hasattr(l, "Name")
                            and hasattr(l, "Begin")
                            and hasattr(l, "End")
                        ):
                            if l.Name in Config.LABEL_MAP:
                                lid = Config.LABEL_MAP[l.Name]
                                start = int(l.Begin * scale)
                                end = int(l.End * scale)
                                # Clamp
                                start = max(0, min(start, skeleton.shape[0] - 1))
                                end = max(0, min(end, skeleton.shape[0]))
                                frame_labels[start : end + 1] = lid
            except:
                pass

        return {
            "skeleton": skeleton,
            "audio": audio,
            "labels": torch.tensor(
                label_seq, dtype=torch.long
            ),  # Sequence labels for metric calc
            "frame_labels": frame_labels,  # Dense labels for Loss
            "sample_id": sample_id,
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences and create masks.
    """
    # Sort by length for packed_sequence (optional but good practice)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels_seq = [x["labels"] for x in batch]
    frame_labels = [x["frame_labels"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad Sequences (Batch, Time, ...)
    # padding_value=0 is fine for data (since it's normalized, 0 is mean)
    # For frame_labels, pad with BACKGROUND_LABEL
    padded_skeletons = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
    padded_frame_labels = pad_sequence(
        frame_labels, batch_first=True, padding_value=Config.BACKGROUND_LABEL
    )

    # Create Mask (Batch, Time) - 1 for valid, 0 for pad
    max_len = padded_skeletons.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "skeleton": padded_skeletons,  # (B, T, J, 3)
        "audio": padded_audios,  # (B, T, F)
        "mask": mask,  # (B, T)
        "lengths": lengths,  # (B,)
        "labels_seq": labels_seq,  # List of tensors (variable length)
        "frame_labels": padded_frame_labels,  # (B, T)
        "sample_ids": ids,
    }


def get_dataloaders():
    """
    Factory function to create Train, Val, and Test dataloaders.
    """
    set_seed(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Compute Stats on Train Data
    stats = compute_global_stats(train_df)

    # Datasets
    train_dataset = GestureDataset(
        Config.TRAIN_METADATA_PATH, stats=stats, is_train=True
    )
    val_dataset = GestureDataset(Config.VAL_METADATA_PATH, stats=stats, is_train=False)
    test_dataset = GestureDataset(
        Config.TEST_METADATA_PATH, stats=stats, is_train=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
