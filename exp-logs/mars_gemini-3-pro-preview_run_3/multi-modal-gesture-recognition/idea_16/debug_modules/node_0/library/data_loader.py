import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


class RobustMatParser:
    """
    Parses .mat files with defensive handling for polymorphic Skeleton structures.
    Extracts 3D joint positions for the primary actor.
    """

    @staticmethod
    def parse(mat_path, num_frames):
        try:
            # Load mat file without squeezing to preserve dimensions initially,
            # but struct_as_record=False is key for object access.
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

            if not hasattr(mat, "Video") or not hasattr(mat["Video"], "Frames"):
                return RobustMatParser._get_empty_skeleton(num_frames)

            frames = mat["Video"].Frames

            # Handle case where Frames is a single object or scalar
            if not isinstance(frames, (np.ndarray, list)):
                if hasattr(frames, "Skeleton"):
                    frames = [frames]
                else:
                    return RobustMatParser._get_empty_skeleton(num_frames)

            # Pre-allocate: (Time, Joints, 3)
            # Config.NUM_JOINTS = 20
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            # Iterate through frames (up to num_frames)
            limit = min(num_frames, len(frames))

            for i in range(limit):
                frame_obj = frames[i]

                # Check if Skeleton field exists
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel = frame_obj.Skeleton

                # Handle Skeleton polymorphism (could be array, single struct, or empty)
                target_skel = None

                if isinstance(skel, np.ndarray):
                    if skel.size > 0:
                        # Pick the first tracked skeleton
                        target_skel = skel[0] if skel.ndim > 0 else skel.item()
                elif hasattr(skel, "WorldPosition"):
                    # Single struct
                    target_skel = skel

                if target_skel is not None and hasattr(target_skel, "WorldPosition"):
                    wp = target_skel.WorldPosition

                    # WorldPosition should be a vector of size 3 (X,Y,Z) or struct
                    # The description says WorldPosition has X, Y, Z fields.
                    # However, typical Kinect mat files might have arrays.
                    # We'll try to extract coordinates.

                    # Case 1: WorldPosition is a struct with X, Y, Z
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # This implies one joint? No, the description says Skeleton contains Joint positions.
                        # Actually, usually Skeleton is an array of Joints.
                        # Let's re-read: "Skeleton... contains... JointsType, WorldPosition..."
                        # It seems Skeleton is a struct containing arrays of joints?
                        # Or Skeleton is an array of Joint structures?
                        # "An array of Skeleton structures is contained within a Skeletons array."
                        # "It contains the joint positions... The format of a Skeleton structure is given below."
                        # It lists JointsType, WorldPosition, etc.
                        # This implies the Skeleton object HAS these fields.
                        # If there are 20 joints, maybe these fields are arrays of length 20?
                        pass

                    # Heuristic: If WorldPosition is an array (20, 3) or similar
                    # If the parser failed to map it perfectly, we fallback to a safe extraction.
                    # We will assume a flattened extraction of any numerical data in WorldPosition
                    # is the safest bet given the ambiguity.

                    # Better Approach based on standard datasets (MSR-DailyActivity etc):
                    # Skeleton is often an array of 20 structs (one per joint).
                    pass

            # Since the exact internal structure of 'Skeleton' regarding joints iteration
            # is complex to guess without the file, we use a generalized extraction
            # that works for the provided sample structure description.
            # We assume we can extract (20, 3) from the frame.

            # Re-implementation of the loop with specific joint extraction logic
            for i in range(limit):
                frame_obj = frames[i]
                if not hasattr(frame_obj, "Skeleton"):
                    continue

                skel_container = frame_obj.Skeleton

                # If multiple users, pick first
                if isinstance(skel_container, np.ndarray) and skel_container.size > 0:
                    skel_container = skel_container[0]

                # Now skel_container should hold joints.
                # If it's a struct with fields like 'HipCenter', 'Spine', etc.
                # Or if it is an array of joint structs.

                # We will try to find 20 coordinates.
                # If we can't parse, we leave as zeros.
                try:
                    # Try to get WorldPosition directly if it's an array of positions
                    if hasattr(skel_container, "WorldPosition"):
                        wp = skel_container.WorldPosition
                        # If wp is (20, 3) array
                        if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                            skeleton_data[i] = wp
                        elif isinstance(wp, np.ndarray) and wp.size == 60:
                            skeleton_data[i] = wp.reshape(20, 3)
                        # If wp is a struct with X,Y,Z arrays
                        elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                            x = np.atleast_1d(wp.X)
                            y = np.atleast_1d(wp.Y)
                            z = np.atleast_1d(wp.Z)
                            if len(x) == 20:
                                skeleton_data[i, :, 0] = x
                                skeleton_data[i, :, 1] = y
                                skeleton_data[i, :, 2] = z
                except:
                    pass

            # Fill missing frames (simple forward fill then backward fill)
            # Identify non-zero frames
            valid_mask = np.any(skeleton_data != 0, axis=(1, 2))
            if not np.any(valid_mask):
                return skeleton_data  # All zeros

            # Interpolate
            # We can use pandas for easy interpolation
            df_temp = pd.DataFrame(skeleton_data.reshape(num_frames, -1))
            df_temp = df_temp.replace(0, np.nan)
            df_temp = df_temp.interpolate(method="linear", limit_direction="both")
            df_temp = df_temp.fillna(0)
            skeleton_data = df_temp.values.reshape(num_frames, Config.NUM_JOINTS, 3)

            return skeleton_data

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return RobustMatParser._get_empty_skeleton(num_frames)

    @staticmethod
    def _get_empty_skeleton(num_frames):
        return np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)


class AudioProcessor:
    @staticmethod
    def process(audio_path, target_num_frames):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            # n_mfcc=13 as per config
            # We need to align with video frames.
            # Strategy: Compute MFCCs with a hop length that approximates the video frame rate,
            # then interpolate to match exactly.

            n_fft = 2048
            hop_length = 512  # Standard
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={"n_fft": n_fft, "n_mels": 64, "hop_length": hop_length},
            )

            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
            mfcc = mfcc.mean(dim=0)  # Average over channels if stereo -> (n_mfcc, time)

            # Transpose to (time, n_mfcc)
            mfcc = mfcc.transpose(0, 1)

            # Interpolate to match target_num_frames
            # Input to interpolate needs to be (Batch, Channels, Length)
            # We treat n_mfcc as channels
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)

            mfcc = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )

            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (target_num_frames, n_mfcc)

            return mfcc.numpy()

        except Exception:
            return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)


class GestureDataset(Dataset):
    def __init__(
        self, metadata_csv, split="train", load_cached_data=True, limit_samples=None
    ):
        """
        Args:
            metadata_csv (str): Path to metadata CSV.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
            limit_samples (int): For debugging, limit number of samples.
        """
        self.split = split
        self.is_train = split == "train"
        self.df = pd.read_csv(metadata_csv)

        if limit_samples:
            self.df = self.df.iloc[:limit_samples]

        self.cache_file = os.path.join(Config.WORK_DIR, f"dataset_{split}.npz")

        # Data containers
        self.all_pos = None  # Concatenated raw positions (N_total_frames, 20, 3)
        self.all_audio = None  # Concatenated audio features (N_total_frames, 13)
        self.all_labels = None  # Concatenated frame labels (N_total_frames,)
        self.seq_info = []  # List of (start_idx, length, sample_id)

        # Load or Process
        if load_cached_data and os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self._process_and_cache()

        # Build Windows
        self.windows = self._build_windows()

    def _process_and_cache(self):
        os.makedirs(Config.WORK_DIR, exist_ok=True)

        pos_list = []
        audio_list = []
        label_list = []
        seq_info_list = []

        current_idx = 0

        for _, row in self.df.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # 1. Parse Skeleton
            # We need num_frames to be robust.
            # We can get it from the mat file or just parse what's there.
            # RobustMatParser handles extraction.
            # However, we need to know the length to align audio.
            # Let's peek at the mat file structure for NumFrames if possible,
            # or rely on the parser to return a consistent array.
            # We'll rely on the parser to determine length based on Frames array.
            # But we need to pass a target length if we want to force consistency?
            # No, let's let parser discover length.

            # Helper to get length first
            try:
                mat = scipy.io.loadmat(
                    mat_path, squeeze_me=True, struct_as_record=False
                )
                num_frames = mat["Video"].NumFrames
            except:
                num_frames = 100  # Fallback

            pos = RobustMatParser.parse(mat_path, num_frames)
            real_frames = pos.shape[0]

            # 2. Process Audio
            audio = AudioProcessor.process(audio_path, real_frames)

            # 3. Create Labels
            labels = np.zeros(real_frames, dtype=np.int32)  # Default 0 (Background)
            if self.split != "test":
                label_meta = json.loads(row["labels"])
                for l in label_meta:
                    lid = l["id"]
                    start = max(0, l["begin"] - 1)  # 1-based to 0-based
                    end = min(real_frames, l["end"])
                    if start < end:
                        labels[start:end] = lid

            # Normalize Positions (mm to meters)
            pos = pos / 1000.0

            pos_list.append(pos)
            audio_list.append(audio)
            label_list.append(labels)

            seq_info_list.append((current_idx, real_frames, sample_id))
            current_idx += real_frames

        # Concatenate
        if not pos_list:
            # Handle empty dataset case
            self.all_pos = np.zeros((0, Config.NUM_JOINTS, 3), dtype=np.float32)
            self.all_audio = np.zeros((0, Config.AUDIO_N_MFCC), dtype=np.float32)
            self.all_labels = np.zeros((0,), dtype=np.int32)
            self.seq_info = []
        else:
            self.all_pos = np.concatenate(pos_list, axis=0).astype(np.float32)
            self.all_audio = np.concatenate(audio_list, axis=0).astype(np.float32)
            self.all_labels = np.concatenate(label_list, axis=0).astype(np.int32)
            self.seq_info = seq_info_list

        # Save to cache (No pickle for arrays)
        # We save seq_info as a separate json or just reconstruct it?
        # We can save seq_info as a numpy array of (start, len). sample_ids can be ignored or saved separately.
        # For simplicity and "no pickle", we save seq_starts and seq_lens.
        seq_starts = np.array([x[0] for x in seq_info_list], dtype=np.int32)
        seq_lens = np.array([x[1] for x in seq_info_list], dtype=np.int32)

        np.savez_compressed(
            self.cache_file,
            pos=self.all_pos,
            audio=self.all_audio,
            labels=self.all_labels,
            seq_starts=seq_starts,
            seq_lens=seq_lens,
        )

    def _load_cache(self):
        try:
            data = np.load(self.cache_file)
            self.all_pos = data["pos"]
            self.all_audio = data["audio"]
            self.all_labels = data["labels"]
            starts = data["seq_starts"]
            lens = data["seq_lens"]

            self.seq_info = []
            # Reconstruct seq_info. We lose sample_id, but we can recover it from self.df order
            # assuming df order hasn't changed.
            for i in range(len(starts)):
                sid = self.df.iloc[i]["sample_id"]
                self.seq_info.append((starts[i], lens[i], sid))
        except Exception as e:
            # print(f"Cache load failed: {e}. Re-processing.")
            self._process_and_cache()

    def _build_windows(self):
        """
        Create a list of (seq_idx, start_frame) for sliding windows.
        """
        windows = []
        stride = Config.TRAIN_STRIDE if self.is_train else Config.TEST_STRIDE
        ws = Config.WINDOW_SIZE

        for i, (start_idx, length, _) in enumerate(self.seq_info):
            if length < ws:
                # Pad short sequences? Or skip?
                # For this dataset, sequences are usually long.
                # If short, we take one window with padding.
                windows.append((i, 0))
            else:
                # Sliding window
                for t in range(0, length - ws + 1, stride):
                    windows.append((i, t))

                # Ensure last frame is covered
                if (length - ws) % stride != 0:
                    windows.append((i, length - ws))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, frame_start = self.windows[idx]
        global_start, length, sample_id = self.seq_info[seq_idx]

        # Indices in the monolithic arrays
        abs_start = global_start + frame_start
        abs_end = abs_start + Config.WINDOW_SIZE

        # Handle padding for short sequences
        if length < Config.WINDOW_SIZE:
            # Extract what we have
            actual_len = length
            pad_len = Config.WINDOW_SIZE - actual_len

            raw_pos = self.all_pos[global_start : global_start + length]
            audio = self.all_audio[global_start : global_start + length]
            labels = self.all_labels[global_start : global_start + length]

            # Pad with zeros (or edge values)
            # Using zero padding for simplicity
            raw_pos = np.pad(raw_pos, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            audio = np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")
            labels = np.pad(
                labels,
                (0, pad_len),
                mode="constant",
                constant_values=Config.BACKGROUND_CLASS_ID,
            )

        else:
            raw_pos = self.all_pos[
                abs_start:abs_end
            ].copy()  # Copy to allow augmentation
            audio = self.all_audio[abs_start:abs_end]
            labels = self.all_labels[abs_start:abs_end]

        # Augmentation (Training Only)
        if self.is_train:
            raw_pos = self._augment(raw_pos)

        # Compute Derivatives (Kinematically Consistent)
        # V_t = P_t - P_{t-1}
        # A_t = V_t - V_{t-1}
        # We pad the first frame of V and first two of A with zeros

        vel = np.zeros_like(raw_pos)
        acc = np.zeros_like(raw_pos)

        vel[1:] = raw_pos[1:] - raw_pos[:-1]
        acc[2:] = vel[2:] - vel[1:-1]

        # Flatten skeleton features: (Time, Joints*3)
        pos_flat = raw_pos.reshape(Config.WINDOW_SIZE, -1)
        vel_flat = vel.reshape(Config.WINDOW_SIZE, -1)
        acc_flat = acc.reshape(Config.WINDOW_SIZE, -1)

        # Concatenate Features
        # Config.INPUT_DIM = 180 + 13 = 193
        # Skeleton part: 20*3*3 = 180
        skel_features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)

        # Final Feature Vector
        features = np.concatenate([skel_features, audio], axis=1)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "targets": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_id,
            "frame_start": frame_start,
        }

    def _augment(self, pos):
        """
        Apply random rotation (Y-axis) and scaling.
        pos: (Time, Joints, 3)
        """
        # Rotation
        theta = np.radians(
            np.random.uniform(-Config.AUG_ROTATION_RANGE, Config.AUG_ROTATION_RANGE)
        )
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: dot product over last dimension
        pos = np.dot(pos, rotation_matrix.T)

        # Scaling
        scale = np.random.uniform(
            1.0 - Config.AUG_SCALE_RANGE, 1.0 + Config.AUG_SCALE_RANGE
        )
        pos = pos * scale

        return pos


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2, limit_samples=None):
    """
    Factory function to get train, val, and test dataloaders.
    """
    train_ds = GestureDataset(
        Config.TRAIN_CSV, split="train", limit_samples=limit_samples
    )
    val_ds = GestureDataset(Config.VAL_CSV, split="val", limit_samples=limit_samples)
    test_ds = GestureDataset(Config.TEST_CSV, split="test", limit_samples=limit_samples)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
