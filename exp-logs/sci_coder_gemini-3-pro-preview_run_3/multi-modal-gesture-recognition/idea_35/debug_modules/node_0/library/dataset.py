import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import logging
from torch.utils.data import Dataset, DataLoader
from library.config import config

# Initialize logger
logger = logging.getLogger("GestureRecognition")


class PolymorphicParser:
    """
    Handles the robust extraction of skeleton data from MATLAB .mat files,
    accounting for variable structures (struct arrays, cell arrays, objects).
    """

    @staticmethod
    def parse_skeleton(mat_path, num_joints=20):
        """
        Parses a .mat file and extracts the skeleton WorldPositions.
        Returns a numpy array of shape (NumFrames, NumJoints, 3).
        """
        try:
            # Load with object support
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

            if not hasattr(mat, "Video"):
                return None

            video = mat["Video"]

            # Extract Frame Count
            if hasattr(video, "NumFrames"):
                num_frames = int(video.NumFrames)
            else:
                # Fallback if NumFrames is missing
                if hasattr(video, "Frames"):
                    num_frames = len(video.Frames)
                else:
                    return None

            # Initialize skeleton array (Frames, Joints, 3) with zeros (mm)
            skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

            if not hasattr(video, "Frames"):
                return skeleton_data  # Return zeros if no frames data

            frames = video.Frames

            # Handle case where Frames is a single object (1 frame)
            if not isinstance(frames, (list, np.ndarray)):
                frames = [frames]

            # Iterate through frames
            for t, frame in enumerate(frames):
                if t >= num_frames:
                    break

                # Check if Skeleton exists
                if not hasattr(frame, "Skeleton"):
                    continue

                skel = frame.Skeleton

                # Check if Skeleton has WorldPosition
                # Sometimes Skeleton is present but empty or None if no user tracked
                if skel is None or (isinstance(skel, np.ndarray) and skel.size == 0):
                    continue

                if not hasattr(skel, "WorldPosition"):
                    continue

                wp = skel.WorldPosition

                # WorldPosition should be an array of joints or a struct
                # We expect 20 joints.
                # Case 1: wp is a struct array/list of objects
                if isinstance(wp, (list, np.ndarray)) and len(wp) == num_joints:
                    for j in range(num_joints):
                        joint = wp[j]
                        if (
                            hasattr(joint, "X")
                            and hasattr(joint, "Y")
                            and hasattr(joint, "Z")
                        ):
                            skeleton_data[t, j, 0] = float(joint.X)
                            skeleton_data[t, j, 1] = float(joint.Y)
                            skeleton_data[t, j, 2] = float(joint.Z)
                # Case 2: wp is a single object (unlikely for 20 joints, but possible if structure differs)
                # Case 3: Direct array (unlikely based on description)

            return skeleton_data

        except Exception as e:
            logger.warning(f"Failed to parse skeleton for {mat_path}: {e}")
            # Return None to indicate failure
            return None


class KinematicAugmentor:
    """
    Applies consistent 3D augmentation to raw skeleton positions and
    derives Velocity and Acceleration *after* augmentation.
    """

    def __init__(self, training=True):
        self.training = training

    def __call__(self, positions):
        """
        Args:
            positions: Tensor (T, J, 3) in millimeters.
        Returns:
            Tensor (T, J, 9) containing [Pos, Vel, Acc].
        """
        # Ensure input is float32
        pos = positions.clone().float()

        if self.training:
            # 1. Random Scaling (0.9 to 1.1)
            scale = 0.9 + (torch.rand(1).item() * 0.2)
            pos = pos * scale

            # 2. Random Rotation around Y-axis (-15 to +15 degrees)
            theta_deg = (torch.rand(1).item() * 30) - 15
            theta_rad = theta_deg * (np.pi / 180.0)

            # Rotation Matrix for Y-axis
            # [ cos  0  sin]
            # [  0   1   0 ]
            # [-sin  0  cos]
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)

            # Apply rotation manually to avoid batch matmul overhead for simple axis
            x = pos[:, :, 0]
            y = pos[:, :, 1]
            z = pos[:, :, 2]

            x_new = x * cos_t + z * sin_t
            y_new = y
            z_new = -x * sin_t + z * cos_t

            pos = torch.stack([x_new, y_new, z_new], dim=-1)

        # 3. Derive Velocity (First Order Difference)
        # Pad first frame to maintain length
        vel = torch.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]  # Replicate first delta

        # 4. Derive Acceleration (Second Order Difference)
        acc = torch.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
        acc[0] = acc[1]

        # Concatenate: (T, J, 9)
        features = torch.cat([pos, vel, acc], dim=-1)
        return features


class GestureDataset(Dataset):
    def __init__(self, metadata_file, mode="train", load_cached_data=True):
        """
        Args:
            metadata_file: Path to the CSV metadata.
            mode: 'train', 'val', or 'test'.
            load_cached_data: Boolean to use cached .npz files.
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_file)
        self.cache_name = f"dataset_{mode}.npz"
        self.cache_path = os.path.join(config.CACHE_DIR, self.cache_name)

        # Data containers
        self.samples = (
            []
        )  # List of dicts: {'features': (T, D), 'labels': (T,), 'id': str}
        self.windows = []  # List of tuples: (sample_idx, start_frame, end_frame)

        # Load or Process
        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_data()
            self._save_cache()

        # Generate Windows
        self._create_windows()

        # Augmentor
        self.augmentor = KinematicAugmentor(training=(mode == "train"))

        # Audio MFCC Transform (used during processing, but defined here for clarity)
        # Note: We process audio during _process_data, not __getitem__ for efficiency

    def _process_data(self):
        logger.info(f"Processing {self.mode} data from scratch...")

        # MFCC Configuration
        # We assume 16kHz audio.
        # To align with video (approx 20fps), hop_length should be fs / fps.
        # Average FPS is 20. 16000 / 20 = 800.
        target_sr = 16000
        target_fps = 20
        hop_length = int(target_sr / target_fps)
        n_mfcc = 13

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": hop_length, "n_mels": 40},
        )

        processed_samples = []

        for idx, row in self.df.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

            # 1. Parse Skeleton (T, 20, 3)
            skeleton = PolymorphicParser.parse_skeleton(mat_path)
            if skeleton is None:
                continue  # Skip corrupt samples

            num_frames = skeleton.shape[0]

            # 2. Parse Audio
            if os.path.exists(audio_path):
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)

                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # Compute MFCC: (1, n_mfcc, time)
                    mfcc = mfcc_transform(waveform)
                    mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

                    # Align with video frames
                    # Simple resize/interpolation or cropping
                    audio_len = mfcc.shape[0]
                    if audio_len != num_frames:
                        # Interpolate to match num_frames
                        mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
                        mfcc = torch.nn.functional.interpolate(
                            mfcc, size=num_frames, mode="linear", align_corners=False
                        )
                        mfcc = mfcc.transpose(1, 2).squeeze(0)  # (num_frames, n_mfcc)

                except Exception as e:
                    logger.warning(f"Audio error {sample_id}: {e}")
                    mfcc = torch.zeros((num_frames, n_mfcc))
            else:
                mfcc = torch.zeros((num_frames, n_mfcc))

            # 3. Create Labels (Frame-wise)
            labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

            if self.mode != "test":
                import json

                try:
                    anns = json.loads(row["labels"])
                    for ann in anns:
                        gid = ann["id"]
                        start = max(0, ann["begin"] - 1)  # 1-based to 0-based
                        end = min(num_frames, ann["end"])
                        labels[start:end] = gid
                except:
                    pass

            # Store raw data (Skeleton + MFCC)
            # Skeleton: (T, 20, 3)
            # MFCC: (T, 13)
            # We store them separately in the dict to augment skeleton later
            processed_samples.append(
                {
                    "skeleton": skeleton.astype(np.float32),
                    "audio": mfcc.numpy().astype(np.float32),
                    "labels": labels,
                    "id": sample_id,
                }
            )

        self.samples = processed_samples

    def _save_cache(self):
        # Save as a single NPZ with object array
        logger.info(f"Saving cache to {self.cache_path}")
        np.savez_compressed(self.cache_path, data=np.array(self.samples, dtype=object))

    def _load_cache(self):
        logger.info(f"Loading cache from {self.cache_path}")
        try:
            loaded = np.load(self.cache_path, allow_pickle=True)
            self.samples = loaded["data"].tolist()
        except Exception as e:
            logger.error(f"Cache load failed: {e}. Reprocessing.")
            self._process_data()
            self._save_cache()

    def _create_windows(self):
        self.windows = []
        window_size = config.WINDOW_SIZE
        stride = config.STRIDE

        for s_idx, sample in enumerate(self.samples):
            num_frames = sample["skeleton"].shape[0]

            # If sample is shorter than window, pad or take whole?
            # We'll pad in __getitem__ if needed, here we just define start points
            if num_frames < window_size:
                self.windows.append((s_idx, 0, num_frames))
                continue

            # Sliding window
            # If test mode, we might want overlapping windows to average predictions
            # If train mode, standard stride

            current_stride = stride if self.mode == "train" else window_size // 2

            for start in range(0, num_frames - window_size + 1, current_stride):
                end = start + window_size
                self.windows.append((s_idx, start, end))

            # Handle remainder for test set to ensure full coverage
            if self.mode == "test" and (num_frames - window_size) % current_stride != 0:
                self.windows.append((s_idx, num_frames - window_size, num_frames))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_idx, start, end = self.windows[idx]
        sample = self.samples[s_idx]

        # 1. Extract Raw Data
        skel_raw = torch.from_numpy(sample["skeleton"][start:end])  # (T, 20, 3)
        audio_raw = torch.from_numpy(sample["audio"][start:end])  # (T, 13)
        labels = torch.from_numpy(sample["labels"][start:end])  # (T,)

        # 2. Pad if shorter than window (only for very short samples)
        curr_len = skel_raw.shape[0]
        if curr_len < config.WINDOW_SIZE:
            pad_len = config.WINDOW_SIZE - curr_len
            # Pad skeleton with zeros (or replicate last frame)
            skel_raw = torch.cat([skel_raw, torch.zeros(pad_len, 20, 3)], dim=0)
            audio_raw = torch.cat([audio_raw, torch.zeros(pad_len, 13)], dim=0)
            # Pad labels with 0 (background)
            labels = torch.cat([labels, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        # 3. Kinematic Augmentation & Feature Derivation
        # Input: (T, 20, 3) -> Output: (T, 20, 9)
        skel_features = self.augmentor(skel_raw)

        # 4. Flatten Skeleton Features
        # (T, 20, 9) -> (T, 180)
        T, J, F = skel_features.shape
        skel_flat = skel_features.view(T, J * F)

        # 5. Early Fusion
        # (T, 180) + (T, 13) -> (T, 193)
        features = torch.cat([skel_flat, audio_raw], dim=-1)

        return features.float(), labels.long()


def get_dataloaders(batch_size=None):
    """
    Factory function to create dataloaders for Train, Val, and Test.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Train Set
    train_ds = GestureDataset(
        config.TRAIN_METADATA, mode="train", load_cached_data=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set
    val_ds = GestureDataset(config.VAL_METADATA, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Test Set
    test_ds = GestureDataset(config.TEST_METADATA, mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader
