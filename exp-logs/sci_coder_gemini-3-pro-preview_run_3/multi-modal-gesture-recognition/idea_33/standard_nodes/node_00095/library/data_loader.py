import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import scipy.io
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    CACHE_DIR,
    SEED,
    WINDOW_SIZE,
    STRIDE,
    NUM_JOINTS,
    AUDIO_N_MFCC,
    AUDIO_SAMPLE_RATE,
    GESTURE_MAP,
    INFERENCE_STRIDE,
    BATCH_SIZE,
)
from library.utils import load_skeleton_data, compute_kinematics

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class PolymorphicMatParser:
    """
    Robustly parses .mat files to extract skeleton data, handling structure variations.
    Wraps the provided utility function to satisfy architectural requirements.
    """

    @staticmethod
    def parse_skeleton(mat_path):
        """
        Parses the skeleton data from the given mat file path.
        Returns:
            np.ndarray: Skeleton data of shape (T, J, 3) or None if failed.
        """
        return load_skeleton_data(mat_path)


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        root_dir,
        cache_dir,
        is_train=True,
        load_cached=True,
        transform=None,
    ):
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.transform = transform

        # Determine split name for caching
        self.split_name = os.path.basename(metadata_path).replace(".csv", "")
        self.cache_file = os.path.join(cache_dir, f"dataset_{self.split_name}.npz")

        # Data containers (loaded from cache or computed)
        self.all_skeletons = None  # (TotalFrames, J, 3)
        self.all_audios = None  # (TotalFrames, n_mfcc)
        self.all_labels = None  # (TotalFrames,)
        self.sample_boundaries = (
            None  # (NumSamples, 2) -> [start, end] indices in global arrays
        )
        self.window_indices = (
            None  # (NumWindows, 3) -> [sample_idx, start_offset, end_offset]
        )
        self.sample_ids = None  # List of sample IDs

        # Audio transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=AUDIO_N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        # Load data
        if load_cached and os.path.exists(self.cache_file):
            self._load_from_cache()
        else:
            self._process_and_cache()

    def _load_from_cache(self):
        print(f"Loading cached data from {self.cache_file}...")
        data = np.load(self.cache_file)
        self.all_skeletons = data["skeletons"]
        self.all_audios = data["audios"]
        self.all_labels = data["labels"]
        self.sample_boundaries = data["sample_boundaries"]
        self.window_indices = data["window_indices"]
        self.sample_ids = data["sample_ids"]
        print(f"Loaded {len(self.window_indices)} windows.")

    def _process_and_cache(self):
        print(f"Processing data from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path)

        skeletons_list = []
        audios_list = []
        labels_list = []
        boundaries = []
        sample_ids = []

        current_global_idx = 0

        for idx, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(self.root_dir, row["data_path"])
            audio_path = os.path.join(self.root_dir, row["audio_path"])

            # 1. Load Skeleton
            skel_data = PolymorphicMatParser.parse_skeleton(data_path)

            # Fallback logic for NumFrames
            num_frames = 0
            if skel_data is not None:
                num_frames = skel_data.shape[0]
            else:
                try:
                    mat = scipy.io.loadmat(
                        data_path, squeeze_me=True, struct_as_record=False
                    )
                    num_frames = mat.Video.NumFrames
                except:
                    num_frames = 100  # Fallback default
                skel_data = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)

            # 2. Load Audio
            audio_features = np.zeros((num_frames, AUDIO_N_MFCC), dtype=np.float32)
            if os.path.exists(audio_path):
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    # Resample if necessary
                    if sample_rate != AUDIO_SAMPLE_RATE:
                        resampler = torchaudio.transforms.Resample(
                            sample_rate, AUDIO_SAMPLE_RATE
                        )
                        waveform = resampler(waveform)

                    # Compute MFCC
                    mfcc = self.mfcc_transform(waveform)
                    if mfcc.dim() == 3:
                        mfcc = mfcc.mean(dim=0)  # Average channels

                    # Interpolate to match video frames
                    mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, L_a)
                    mfcc = torch.nn.functional.interpolate(
                        mfcc, size=num_frames, mode="linear", align_corners=False
                    )
                    mfcc = (
                        mfcc.squeeze(0).transpose(0, 1).numpy()
                    )  # (num_frames, n_mfcc)
                    audio_features = mfcc
                except Exception:
                    pass

            # 3. Build Labels
            label_seq = np.zeros((num_frames,), dtype=np.int64)
            if "labels" in row and isinstance(row["labels"], str):
                try:
                    labels_meta = json.loads(row["labels"])
                    for l in labels_meta:
                        start = max(0, l["begin"] - 1)
                        end = min(num_frames, l["end"])
                        gid = l["id"]
                        if start < end:
                            label_seq[start:end] = gid
                except:
                    pass

            # 4. Append
            skeletons_list.append(skel_data)
            audios_list.append(audio_features)
            labels_list.append(label_seq)
            sample_ids.append(sample_id)

            boundaries.append([current_global_idx, current_global_idx + num_frames])
            current_global_idx += num_frames

        # Concatenate
        if not skeletons_list:
            self.all_skeletons = np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
            self.all_audios = np.zeros((0, AUDIO_N_MFCC), dtype=np.float32)
            self.all_labels = np.zeros((0,), dtype=np.int64)
            self.sample_boundaries = np.zeros((0, 2), dtype=np.int64)
            self.window_indices = np.zeros((0, 3), dtype=np.int64)
            self.sample_ids = []
        else:
            self.all_skeletons = np.concatenate(skeletons_list, axis=0).astype(
                np.float32
            )
            self.all_audios = np.concatenate(audios_list, axis=0).astype(np.float32)
            self.all_labels = np.concatenate(labels_list, axis=0).astype(np.int64)
            self.sample_boundaries = np.array(boundaries, dtype=np.int64)
            self.sample_ids = sample_ids

            # Generate Windows
            windows = []
            stride = STRIDE if self.is_train else INFERENCE_STRIDE

            for i, (start, end) in enumerate(self.sample_boundaries):
                length = end - start
                if length <= WINDOW_SIZE:
                    windows.append([i, 0, length])
                else:
                    for w_start in range(0, length - WINDOW_SIZE + 1, stride):
                        windows.append([i, w_start, w_start + WINDOW_SIZE])
                    if (length - WINDOW_SIZE) % stride != 0:
                        windows.append([i, length - WINDOW_SIZE, length])

            self.window_indices = np.array(windows, dtype=np.int64)

        # Cache
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(
            self.cache_file,
            skeletons=self.all_skeletons,
            audios=self.all_audios,
            labels=self.all_labels,
            sample_boundaries=self.sample_boundaries,
            window_indices=self.window_indices,
            sample_ids=self.sample_ids,
        )
        print(f"Cached processed data to {self.cache_file}")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        sample_idx, rel_start, rel_end = self.window_indices[idx]
        global_start_offset = self.sample_boundaries[sample_idx][0]

        abs_start = global_start_offset + rel_start
        abs_end = global_start_offset + rel_end

        skel = self.all_skeletons[abs_start:abs_end].copy()  # (T_actual, J, 3)
        audio = self.all_audios[abs_start:abs_end].copy()  # (T_actual, F)
        labels = self.all_labels[abs_start:abs_end].copy()  # (T_actual,)

        # Pad if shorter than WINDOW_SIZE
        T_actual = skel.shape[0]
        if T_actual < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - T_actual
            skel = np.pad(skel, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            audio = np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")
            labels = np.pad(labels, (0, pad_len), mode="constant", constant_values=0)

        # Augmentation (Train only)
        if self.is_train:
            # Random Rotation around Y-axis
            theta = np.random.uniform(-0.3, 0.3)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            T, J, _ = skel.shape
            skel_flat = skel.reshape(-1, 3)
            skel_flat = skel_flat @ R.T
            skel = skel_flat.reshape(T, J, 3)

            # Random Scaling
            scale = np.random.uniform(0.9, 1.1)
            skel = skel * scale

        # Compute Kinematics (on augmented skeleton)
        kinematics = compute_kinematics(skel)  # (T, J, 9)

        T, J, F = kinematics.shape
        kinematics_flat = kinematics.reshape(T, J * F)

        # Fuse with Audio
        fused_features = np.concatenate([kinematics_flat, audio], axis=1)

        x = torch.from_numpy(fused_features).float()
        y = torch.from_numpy(labels).long()

        return x, y, sample_idx, rel_start


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=2):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    train_dataset = GestureDataset(
        metadata_path=TRAIN_METADATA_PATH,
        root_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        is_train=True,
        load_cached=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = GestureDataset(
        metadata_path=VAL_METADATA_PATH,
        root_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        is_train=False,
        load_cached=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataset = GestureDataset(
        metadata_path=TEST_METADATA_PATH,
        root_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        is_train=False,
        load_cached=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
