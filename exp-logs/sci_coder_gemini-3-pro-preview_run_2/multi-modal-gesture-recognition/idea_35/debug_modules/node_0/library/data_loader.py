import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import scipy.signal
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def load_audio_features(audio_path, target_num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    """
    try:
        if not os.path.exists(audio_path):
            return np.zeros(
                (target_num_frames, Config.AUDIO_FEATURE_DIM), dtype=np.float32
            )

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary (assuming 16kHz is standard)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        # Compute MFCC
        # Video FPS is approx 10-20. 16000 / 20 = 800 samples per frame.
        n_fft = 2048
        hop_length = 512
        n_mfcc = Config.AUDIO_FEATURE_DIM

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": 64},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Interpolate to match target_num_frames
        if mfcc.shape[0] != target_num_frames:
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        return mfcc.numpy()

    except Exception:
        return np.zeros((target_num_frames, Config.AUDIO_FEATURE_DIM), dtype=np.float32)


def process_sample(sample_row, is_train=True):
    """
    Reads multimodal data for a single sample and extracts features and labels.
    """
    data_path = os.path.join(Config.INPUT_DIR, sample_row["data_path"])
    audio_path = os.path.join(Config.INPUT_DIR, sample_row["audio_path"])

    try:
        # Load Mat File
        mat = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        frames = video.Frames
        num_frames = getattr(
            video,
            "NumFrames",
            len(frames) if isinstance(frames, (list, np.ndarray)) else 0,
        )

        if num_frames == 0:
            return None, None, None

        # --- Skeleton Processing ---
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        num_selected = len(Config.SELECTED_JOINTS)
        skeleton_data = np.zeros((num_frames, num_selected, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            if i >= num_frames:
                break
            try:
                skel = frame.Skeleton
                # Handle array of skeletons (multiple users) -> take first
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]

                # Check if skel contains joints array
                if hasattr(skel, "__len__") and len(skel) >= 20:
                    for j_idx, joint_idx in enumerate(Config.SELECTED_JOINTS):
                        joint = skel[joint_idx]
                        pos = joint.WorldPosition
                        skeleton_data[i, j_idx, 0] = pos.X
                        skeleton_data[i, j_idx, 1] = pos.Y
                        skeleton_data[i, j_idx, 2] = pos.Z
            except Exception:
                pass

        # Normalization: Center around HipCenter (Index 0) and Scale
        hip_center = skeleton_data[:, 0:1, :]
        skeleton_data = skeleton_data - hip_center
        skeleton_data = skeleton_data * Config.SCALE_FACTOR
        skeleton_flat = skeleton_data.reshape(num_frames, -1)

        # --- Audio Processing ---
        audio_feat = load_audio_features(audio_path, num_frames)

        # Concatenate Features
        features = np.concatenate([skeleton_flat, audio_feat], axis=1)  # (T, 49)

        # --- Label Processing ---
        cls_target = np.zeros(num_frames, dtype=np.int64)  # 0 is background
        bnd_target = np.zeros(num_frames, dtype=np.float32)

        if is_train:
            labels_raw = getattr(video, "Labels", [])

            GESTURE_MAP = {
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

            def process_label_obj(obj):
                try:
                    name = obj.Name
                    start = obj.Begin - 1  # Convert to 0-based
                    end = obj.End - 1

                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        start = max(0, start)
                        end = min(num_frames - 1, end)

                        cls_target[start : end + 1] = gid
                        # Boundary supervision: 1 at start and end frames
                        bnd_target[start] = 1.0
                        bnd_target[end] = 1.0
                except AttributeError:
                    pass

            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_label_obj(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_label_obj(l)
            else:
                process_label_obj(labels_raw)

        return features, cls_target, bnd_target

    except Exception:
        return None, None, None


def physically_consistent_augmentation(features):
    """
    Applies smooth noise to skeleton positions to maintain physical consistency.
    """
    T, D = features.shape
    num_skel = Config.NUM_JOINTS * Config.COORDS_PER_JOINT

    if num_skel > D:
        return features

    skel_data = features[:, :num_skel].reshape(T, Config.NUM_JOINTS, 3)
    audio_data = features[:, num_skel:]

    # Generate Gaussian Noise (Scale 0.005m = 5mm)
    noise = np.random.normal(0, 0.005, size=skel_data.shape)

    # Apply Temporal Low-Pass Filter (Butterworth)
    # This ensures the added noise implies a smooth velocity, not infinite jerk
    b, a = scipy.signal.butter(4, 0.2, btype="low", analog=False)
    smooth_noise = scipy.signal.lfilter(b, a, noise, axis=0)

    augmented_skel = skel_data + smooth_noise
    augmented_skel = augmented_skel.reshape(T, -1)

    return np.concatenate([augmented_skel, audio_data], axis=1).astype(np.float32)


class GestureDataset(Dataset):
    def __init__(
        self, metadata_file, is_train=True, load_cached_data=True, sample_size=None
    ):
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_file)

        if sample_size:
            self.metadata = self.metadata.head(sample_size)

        # Determine cache filename
        base_name = (
            "train"
            if "train.csv" in metadata_file
            else ("val" if "val.csv" in metadata_file else "test")
        )
        if sample_size:
            base_name += f"_sample{sample_size}"
        self.cache_path = os.path.join(Config.WORKING_DIR, f"{base_name}_data.npz")

        self.features_list = []
        self.labels_list = []
        self.boundaries_list = []

        # Caching Logic
        loaded = False
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached data from {self.cache_path}...")
                data = np.load(self.cache_path)
                features_flat = data["features"]
                labels_flat = data["labels"]
                bnd_flat = data["boundaries"]
                limits = data["limits"]

                for start, end in limits:
                    self.features_list.append(features_flat[start:end])
                    if is_train:
                        self.labels_list.append(labels_flat[start:end])
                        self.boundaries_list.append(bnd_flat[start:end])
                    else:
                        self.labels_list.append(np.zeros(end - start, dtype=np.int64))
                        self.boundaries_list.append(
                            np.zeros(end - start, dtype=np.float32)
                        )
                loaded = True
            except Exception as e:
                print(f"Cache load failed: {e}. Reprocessing...")

        if not loaded:
            self._process_and_cache()

    def _process_and_cache(self):
        print(f"Processing {len(self.metadata)} samples...")
        features_accum = []
        labels_accum = []
        bnd_accum = []
        limits = []
        current_idx = 0

        for idx, row in self.metadata.iterrows():
            feat, cls, bnd = process_sample(row, self.is_train)

            if feat is None:
                continue

            length = feat.shape[0]
            if length == 0:
                continue

            features_accum.append(feat)
            if self.is_train:
                labels_accum.append(cls)
                bnd_accum.append(bnd)
            else:
                labels_accum.append(np.zeros(length, dtype=np.int64))
                bnd_accum.append(np.zeros(length, dtype=np.float32))

            limits.append([current_idx, current_idx + length])
            current_idx += length

            self.features_list.append(feat)
            self.labels_list.append(
                cls if self.is_train else np.zeros(length, dtype=np.int64)
            )
            self.boundaries_list.append(
                bnd if self.is_train else np.zeros(length, dtype=np.float32)
            )

        # Save to cache
        if features_accum:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            features_flat = np.concatenate(features_accum, axis=0)
            labels_flat = np.concatenate(labels_accum, axis=0)
            bnd_flat = np.concatenate(bnd_accum, axis=0)
            limits_arr = np.array(limits)

            np.savez_compressed(
                self.cache_path,
                features=features_flat,
                labels=labels_flat,
                boundaries=bnd_flat,
                limits=limits_arr,
            )
            print(f"Saved cache to {self.cache_path}")

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        feat = self.features_list[idx]
        lbl = self.labels_list[idx]
        bnd = self.boundaries_list[idx]

        # Augmentation only for training
        if self.is_train:
            feat = physically_consistent_augmentation(feat)

        return torch.from_numpy(feat), torch.from_numpy(lbl), torch.from_numpy(bnd)


def collate_fn(batch):
    features, labels, boundaries = zip(*batch)

    lengths = torch.tensor([len(f) for f in features])

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    boundaries_padded = pad_sequence(boundaries, batch_first=True, padding_value=0)

    # Create Mask
    B, T, _ = features_padded.shape
    mask = torch.arange(T).expand(B, T) < lengths.unsqueeze(1)

    return features_padded, labels_padded, boundaries_padded, mask, lengths


def get_dataloaders(debug_size=None):
    train_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        is_train=True,
        sample_size=debug_size,
    )

    val_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        is_train=True,
        sample_size=debug_size,
    )

    test_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        is_train=False,
        sample_size=debug_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader
