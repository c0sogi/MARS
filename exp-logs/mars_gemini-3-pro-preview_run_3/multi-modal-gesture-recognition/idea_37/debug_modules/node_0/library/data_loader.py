import os
import json
import numpy as np
import pandas as pd
import scipy.io
import scipy.interpolate
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# ==========================================
# Constants & Mappings
# ==========================================
JOINT_MAP = {
    "HipCenter": 0,
    "Spine": 1,
    "ShoulderCenter": 2,
    "Head": 3,
    "ShoulderLeft": 4,
    "ElbowLeft": 5,
    "WristLeft": 6,
    "HandLeft": 7,
    "ShoulderRight": 8,
    "ElbowRight": 9,
    "WristRight": 10,
    "HandRight": 11,
    "HipLeft": 12,
    "KneeLeft": 13,
    "AnkleLeft": 14,
    "FootLeft": 15,
    "HipRight": 16,
    "KneeRight": 17,
    "AnkleRight": 18,
    "FootRight": 19,
}

JOINT_NAMES = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]

# ==========================================
# Helper Classes
# ==========================================


class PolymorphicSkeletonParser:
    """
    Parses .mat files to extract skeleton data, handling inconsistent structures
    (struct arrays vs cell arrays vs single objects).
    """

    def parse(self, mat_path):
        """
        Returns:
            np.ndarray: Shape (NumFrames, 20, 3) containing X, Y, Z in millimeters.
                        Returns None if parsing fails significantly.
        """
        try:
            # Load mat file
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat._fieldnames:
                return None

            video = mat.Video
            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames
            num_frames = len(frames) if isinstance(frames, (list, np.ndarray)) else 1

            # Initialize container: (Time, Joints, 3)
            skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

            # Ensure frames is iterable
            if not isinstance(frames, (list, np.ndarray)):
                frames = [frames]

            for f_idx, frame in enumerate(frames):
                if not hasattr(frame, "Skeleton") or frame.Skeleton is None:
                    # If skeleton missing, copy previous frame or leave as 0
                    if f_idx > 0:
                        skeleton_data[f_idx] = skeleton_data[f_idx - 1]
                    continue

                skel_obj = frame.Skeleton

                # Check if skel_obj is iterable (multiple joints) or a single struct
                # Usually it's an array of structs, one per joint
                joints_list = []
                if isinstance(skel_obj, (list, np.ndarray)):
                    joints_list = skel_obj
                elif hasattr(skel_obj, "JointsType"):
                    # Single object, unlikely for full skeleton but possible
                    joints_list = [skel_obj]
                else:
                    # Unknown structure, skip
                    if f_idx > 0:
                        skeleton_data[f_idx] = skeleton_data[f_idx - 1]
                    continue

                # Extract joints
                current_frame_joints = np.zeros((20, 3), dtype=np.float32)

                for j_obj in joints_list:
                    if not hasattr(j_obj, "JointsType") or not hasattr(
                        j_obj, "WorldPosition"
                    ):
                        continue

                    j_type = str(j_obj.JointsType)
                    if j_type in JOINT_MAP:
                        j_idx = JOINT_MAP[j_type]
                        pos = j_obj.WorldPosition
                        # WorldPosition might be struct with X,Y,Z or array
                        if (
                            hasattr(pos, "X")
                            and hasattr(pos, "Y")
                            and hasattr(pos, "Z")
                        ):
                            current_frame_joints[j_idx] = [pos.X, pos.Y, pos.Z]
                        elif isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                            current_frame_joints[j_idx] = pos[:3]

                # If we parsed joints, assign. If mostly empty, maybe copy prev?
                # For now, assign what we found.
                skeleton_data[f_idx] = current_frame_joints

            return skeleton_data

        except Exception as e:
            print(f"Error parsing {mat_path}: {e}")
            return None


class GestureDataset(Dataset):
    def __init__(self, metadata_csv, mode="train", load_cached_data=True):
        """
        Args:
            metadata_csv (str): Path to metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching.
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_csv)

        # Debug subset
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE)

        self.parser = PolymorphicSkeletonParser()
        self.samples = []  # List of dicts with 'skeleton', 'audio', 'labels'
        self.windows = []  # List of (sample_idx, start_frame, end_frame)

        self.cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{mode}.npz")
        if Config.DEBUG:
            self.cache_path = os.path.join(
                Config.CACHE_DIR, f"dataset_{mode}_debug.npz"
            )

        # Load Data
        self._load_data(load_cached_data)

        # Create Windows
        self._create_windows()

    def _load_data(self, load_cached_data):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading {self.mode} data from cache: {self.cache_path}")
                loaded = np.load(self.cache_path, allow_pickle=True)
                # Reconstruct samples list
                # Stored as arrays: 'sample_0_skel', 'sample_0_audio', ...
                num_samples = loaded["num_samples"]
                for i in range(num_samples):
                    self.samples.append(
                        {
                            "skeleton": loaded[f"sample_{i}_skel"],
                            "audio": loaded[f"sample_{i}_audio"],
                            "labels": loaded[f"sample_{i}_labels"],
                        }
                    )
                return
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing {len(self.df)} samples for {self.mode}...")

        # MFCC Transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SR,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        for idx, row in self.df.iterrows():
            # Paths
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # --- Skeleton ---
            skeleton = self.parser.parse(data_path)
            if skeleton is None:
                # Fallback: create dummy
                skeleton = np.zeros((100, 20, 3), dtype=np.float32)

            num_frames = skeleton.shape[0]

            # --- Audio ---
            if os.path.exists(audio_path):
                try:
                    wav, sr = sf.read(audio_path)
                    # Convert to mono if needed
                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)

                    # Resample if needed (assuming 16k target)
                    if sr != Config.AUDIO_SR:
                        # Simple resampling using scipy or just skip if close
                        # For robustness, we assume 16k or handle in MFCC
                        # But torchaudio MFCC expects specific SR.
                        # Let's use torch resample if needed
                        t_wav = torch.tensor(wav, dtype=torch.float32)
                        resampler = torchaudio.transforms.Resample(sr, Config.AUDIO_SR)
                        t_wav = resampler(t_wav)
                        wav = t_wav.numpy()

                    # Compute MFCC
                    t_wav = torch.tensor(wav, dtype=torch.float32)
                    mfcc = mfcc_transform(t_wav)  # (n_mfcc, time)
                    mfcc = mfcc.transpose(0, 1).numpy()  # (time, n_mfcc)

                    # Align Audio to Video Frames
                    # We need exactly num_frames
                    if mfcc.shape[0] != num_frames:
                        x_old = np.linspace(0, 1, mfcc.shape[0])
                        x_new = np.linspace(0, 1, num_frames)
                        f = scipy.interpolate.interp1d(
                            x_old, mfcc, axis=0, kind="linear", fill_value="extrapolate"
                        )
                        mfcc_aligned = f(x_new)
                    else:
                        mfcc_aligned = mfcc

                except Exception as e:
                    print(f"Audio error {audio_path}: {e}")
                    mfcc_aligned = np.zeros(
                        (num_frames, Config.N_MFCC), dtype=np.float32
                    )
            else:
                mfcc_aligned = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)

            # --- Labels ---
            labels = np.zeros(num_frames, dtype=np.int64)  # 0 = background
            if self.mode != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # 1-based indexing in mat file usually, need to check
                    # Python is 0-based. 'begin' and 'end' are frame indices.
                    # Assuming metadata is 1-based (Matlab standard), convert to 0-based
                    start = max(0, int(l["begin"]) - 1)
                    end = min(num_frames, int(l["end"]))
                    lid = int(l["id"])
                    labels[start:end] = lid

            self.samples.append(
                {
                    "skeleton": skeleton.astype(np.float32),
                    "audio": mfcc_aligned.astype(np.float32),
                    "labels": labels.astype(np.int64),
                }
            )

        # Save to cache
        save_dict = {"num_samples": len(self.samples)}
        for i, s in enumerate(self.samples):
            save_dict[f"sample_{i}_skel"] = s["skeleton"]
            save_dict[f"sample_{i}_audio"] = s["audio"]
            save_dict[f"sample_{i}_labels"] = s["labels"]

        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Saved cache to {self.cache_path}")

    def _create_windows(self):
        stride = Config.STRIDE_TRAIN if self.mode == "train" else Config.STRIDE_TEST
        w_size = Config.WINDOW_SIZE

        for s_idx, sample in enumerate(self.samples):
            num_frames = sample["skeleton"].shape[0]

            # If sample is shorter than window, pad it?
            # Or just take one window with padding.
            if num_frames < w_size:
                self.windows.append(
                    (s_idx, 0, num_frames)
                )  # Will handle padding in getitem
                continue

            # Sliding window
            # For test, we need to cover the end.
            for start in range(0, num_frames - w_size + 1, stride):
                self.windows.append((s_idx, start, start + w_size))

            # Handle remainder for test/val to ensure full coverage
            if self.mode != "train":
                last_start = self.windows[-1][1]
                if last_start + w_size < num_frames:
                    # Add one last window aligned to the end
                    self.windows.append((s_idx, num_frames - w_size, num_frames))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_idx, start, end = self.windows[idx]
        sample = self.samples[s_idx]

        # Extract Raw Data
        # Handle case where end > actual length (if we allowed that logic, though create_windows avoids it mostly)
        # Or if sample was shorter than window
        actual_len = sample["skeleton"].shape[0]

        # Initialize buffers
        skel_win = np.zeros((Config.WINDOW_SIZE, 20, 3), dtype=np.float32)
        audio_win = np.zeros((Config.WINDOW_SIZE, Config.N_MFCC), dtype=np.float32)
        label_win = np.zeros((Config.WINDOW_SIZE,), dtype=np.int64)

        # Copy data
        copy_len = min(Config.WINDOW_SIZE, actual_len - start)
        if copy_len > 0:
            skel_win[:copy_len] = sample["skeleton"][start : start + copy_len]
            audio_win[:copy_len] = sample["audio"][start : start + copy_len]
            label_win[:copy_len] = sample["labels"][start : start + copy_len]

        # --- Augmentation (Train Only) ---
        if self.mode == "train":
            # 1. Random Scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            skel_win = skel_win * scale

            # 2. Random Y-Rotation (-20 to +20 degrees)
            theta = np.deg2rad(np.random.uniform(-20, 20))
            c, s = np.cos(theta), np.sin(theta)
            # x' = x*c + z*s
            # z' = -x*s + z*c
            x = skel_win[:, :, 0].copy()
            z = skel_win[:, :, 2].copy()
            skel_win[:, :, 0] = x * c + z * s
            skel_win[:, :, 2] = -x * s + z * c

        # --- Kinematic Derivatives ---
        # Velocity: P_t - P_{t-1}
        # Acceleration: V_t - V_{t-1}
        # Prepend 0 for first frame diff

        # Shape: (T, 20, 3)
        pos = skel_win

        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate: (T, 20, 9)
        kinematics = np.concatenate([pos, vel, acc], axis=2)

        # Flatten joints: (T, 180)
        skel_feat = kinematics.reshape(Config.WINDOW_SIZE, -1)

        # Concatenate Audio: (T, 180 + 13) = (T, 193)
        final_input = np.concatenate([skel_feat, audio_win], axis=1)

        return {
            "feature": torch.tensor(final_input, dtype=torch.float32),
            "label": torch.tensor(label_win, dtype=torch.long),
            "sample_idx": s_idx,
            "window_idx": idx,
        }


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    set_seed(Config.SEED)

    # Datasets
    train_ds = GestureDataset(
        Config.TRAIN_CSV, mode="train", load_cached_data=load_cached_data
    )
    val_ds = GestureDataset(
        Config.VAL_CSV, mode="val", load_cached_data=load_cached_data
    )
    test_ds = GestureDataset(
        Config.TEST_CSV, mode="test", load_cached_data=load_cached_data
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
