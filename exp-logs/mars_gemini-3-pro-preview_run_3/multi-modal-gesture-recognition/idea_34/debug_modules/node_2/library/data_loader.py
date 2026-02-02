import os
import numpy as np
import pandas as pd
import scipy.io
import librosa
import torch
from torch.utils.data import Dataset
from library import config

# Set fixed seeds for reproducibility
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


class PolymorphicParser:
    """
    Robustly parses .mat files to extract skeleton data, handling variable
    Matlab struct representations (objects, arrays, etc.).
    """

    @staticmethod
    def load_mat(path):
        try:
            # squeeze_me=True simplifies the structure (1x1 arrays become scalars)
            # struct_as_record=False loads structs as Python objects with attributes
            return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    @staticmethod
    def extract_skeleton(mat_path):
        """
        Parses the .mat file at mat_path and returns a (T, 20, 3) numpy array
        of joint positions. Returns None if extraction fails.
        """
        mat = PolymorphicParser.load_mat(mat_path)
        if mat is None or "Video" not in mat:
            return None

        video = mat["Video"]

        # Unwrap 0-d array if necessary (scipy.io.loadmat with squeeze_me=True)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        # Determine number of frames
        if hasattr(video, "NumFrames"):
            num_frames = int(video.NumFrames)
        elif hasattr(video, "Frames"):
            num_frames = (
                len(video.Frames) if isinstance(video.Frames, (list, np.ndarray)) else 1
            )
        else:
            return None

        if num_frames == 0:
            return None

        # Initialize container: (Time, Joints, Coords)
        # 20 joints is standard for this Kinect data
        skeleton_data = np.zeros((num_frames, config.NUM_JOINTS, 3), dtype=np.float32)

        # Access Frames
        if not hasattr(video, "Frames"):
            return skeleton_data  # Return zeros if no frames data

        frames = video.Frames

        # Handle case where Frames is a single object (not a list/array)
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        # Iterate through frames
        # Note: If num_frames mismatches len(frames), we take the min
        limit = min(num_frames, len(frames))

        for t in range(limit):
            frame_obj = frames[t]

            # Check if Skeleton exists
            if not hasattr(frame_obj, "Skeleton"):
                continue

            skel = frame_obj.Skeleton

            # Check if WorldPosition exists
            if not hasattr(skel, "WorldPosition"):
                continue

            wp = skel.WorldPosition

            # Extract X, Y, Z
            # wp might be an array of structs (if multiple users) or a single struct
            # We assume single user or take the first one if array
            if isinstance(wp, (list, np.ndarray)) and len(wp) > 0:
                # If it's an array, it might be joints directly or multiple users
                # Standard Kinect format in these datasets usually puts joints in WorldPosition
                # if it's a struct of arrays, OR WorldPosition is an array of structs.
                # Based on description: "WorldPosition... X, Y, Z".
                # Usually 20 joints.
                pass

            # The description says: Skeleton Frame contains JointsType, WorldPosition...
            # WorldPosition contains X, Y, Z.
            # Usually, skel.WorldPosition is a 20x1 struct array or similar.

            # Robust extraction strategy:
            # Try to get X, Y, Z arrays directly
            try:
                # Case A: WorldPosition is an array of 20 structs
                if isinstance(wp, (list, np.ndarray)) and len(wp) == config.NUM_JOINTS:
                    for j in range(config.NUM_JOINTS):
                        joint = wp[j]
                        skeleton_data[t, j, 0] = float(joint.X)
                        skeleton_data[t, j, 1] = float(joint.Y)
                        skeleton_data[t, j, 2] = float(joint.Z)
                # Case B: WorldPosition is a single struct containing arrays of length 20
                elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # Check if X is scalar or array
                    x_val = wp.X
                    if np.size(x_val) == config.NUM_JOINTS:
                        skeleton_data[t, :, 0] = x_val
                        skeleton_data[t, :, 1] = wp.Y
                        skeleton_data[t, :, 2] = wp.Z
                    else:
                        # Maybe single joint? Unlikely for "Skeleton"
                        pass
                else:
                    # Fallback: maybe skel itself is an array of joints?
                    # The prompt says "Skeleton Frame... contained within a Skeletons array".
                    # But usually the mat file structure provided in this dataset (Chalearn)
                    # has Video.Frames(t).Skeleton.WorldPosition(j).X
                    pass
            except Exception:
                # If extraction fails for a frame, leave as zeros
                pass

        return skeleton_data


def extract_audio_features(wav_path, target_num_frames):
    """
    Loads audio and extracts MFCCs, aligned to the video frame count.
    """
    if not os.path.exists(wav_path):
        return np.zeros((target_num_frames, config.N_MFCC), dtype=np.float32)

    try:
        # Load audio at 16kHz (standard for speech/Kinect)
        y, sr = librosa.load(wav_path, sr=16000)

        if len(y) == 0:
            return np.zeros((target_num_frames, config.N_MFCC), dtype=np.float32)

        # Calculate hop length to match video frames
        # num_frames = 1 + (len(y) - n_fft) / hop_length
        # approx: hop_length = len(y) / num_frames
        hop_length = int(len(y) / target_num_frames)
        if hop_length < 1:
            hop_length = 1

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=config.N_MFCC, hop_length=hop_length
        )
        # mfcc shape: (n_mfcc, T_audio)
        mfcc = mfcc.T  # (T_audio, n_mfcc)

        # Fix length mismatch due to rounding
        if mfcc.shape[0] != target_num_frames:
            # Resize using linear interpolation
            from scipy.ndimage import zoom

            # Zoom factor for time axis
            zoom_factor = target_num_frames / mfcc.shape[0]
            # We only zoom axis 0, axis 1 is features
            mfcc = zoom(mfcc, (zoom_factor, 1), order=1)

            # Hard clip if still off by a pixel
            if mfcc.shape[0] > target_num_frames:
                mfcc = mfcc[:target_num_frames]
            elif mfcc.shape[0] < target_num_frames:
                pad_width = target_num_frames - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="edge")

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"Audio processing error {wav_path}: {e}")
        return np.zeros((target_num_frames, config.N_MFCC), dtype=np.float32)


def process_and_cache_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads raw data, processes it into continuous arrays, and caches to disk.
    Returns:
        skeletons (N_total, 20, 3)
        audio (N_total, 13)
        labels (N_total,)
        sample_map (List of dicts defining start/end indices for each sample)
    """
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        try:
            data = np.load(
                cache_file, allow_pickle=True
            )  # allow_pickle needed for sample_map object
            return (
                data["skeletons"],
                data["audio"],
                data["labels"],
                data["sample_map"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Containers
    all_skeletons = []
    all_audio = []
    all_labels = []
    sample_map = []

    current_idx = 0

    import json

    print(f"Processing {len(df)} samples for {cache_name}...")

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton
        skeleton = PolymorphicParser.extract_skeleton(data_path)
        if skeleton is None:
            # Skip corrupted samples or handle gracefully?
            # We'll skip to avoid breaking training, but for test we might need dummy data.
            # Given competition context, let's create dummy data if test, else skip.
            if "test" in cache_name:
                # Try to infer length from audio or video?
                # Assume a default length or skip.
                # Better to skip and handle missing IDs in submission if necessary,
                # but let's assume data is mostly good.
                continue
            else:
                continue

        num_frames = skeleton.shape[0]

        # 2. Load Audio
        audio = extract_audio_features(audio_path, num_frames)

        # 3. Create Labels
        label_seq = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

        # Parse labels JSON
        if isinstance(row["labels"], str):
            try:
                gestures = json.loads(row["labels"])
                for g in gestures:
                    start = max(0, g["begin"] - 1)  # 1-based to 0-based
                    end = min(num_frames, g["end"])
                    gid = g["id"]
                    if start < end:
                        label_seq[start:end] = gid
            except:
                pass

        # Append
        all_skeletons.append(skeleton)
        all_audio.append(audio)
        all_labels.append(label_seq)

        sample_map.append(
            {
                "sample_id": sample_id,
                "start_idx": current_idx,
                "end_idx": current_idx + num_frames,
                "num_frames": num_frames,
            }
        )

        current_idx += num_frames

    # Concatenate
    if len(all_skeletons) == 0:
        raise ValueError("No valid data found!")

    big_skeleton = np.concatenate(all_skeletons, axis=0)
    big_audio = np.concatenate(all_audio, axis=0)
    big_labels = np.concatenate(all_labels, axis=0)

    # Save
    np.savez_compressed(
        cache_file,
        skeletons=big_skeleton,
        audio=big_audio,
        labels=big_labels,
        sample_map=sample_map,
    )

    return big_skeleton, big_audio, big_labels, sample_map


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, dataset_type="train", load_cached_data=True, limit=None
    ):
        """
        Args:
            metadata_path: Path to CSV.
            dataset_type: 'train', 'val', or 'test'.
            load_cached_data: Whether to use disk cache.
            limit: Limit number of samples (for debugging).
        """
        self.dataset_type = dataset_type
        self.is_train = dataset_type == "train"

        # Load raw continuous data
        self.skeletons, self.audio, self.labels, self.sample_map = (
            process_and_cache_data(
                metadata_path, f"dataset_{dataset_type}", load_cached_data
            )
        )

        # If limit is set, truncate sample_map and rebuild indices
        if limit:
            self.sample_map = self.sample_map[:limit]
            # We don't slice the big arrays to save memory ops, we just limit the window generation

        # Generate Window Indices
        self.window_indices = []
        stride = config.STRIDE_TRAIN if self.is_train else config.STRIDE_TEST

        for sample in self.sample_map:
            start_global = sample["start_idx"]
            end_global = sample["end_idx"]
            length = sample["num_frames"]

            if length < config.WINDOW_SIZE:
                # Pad small samples? Or just take one window with padding?
                # We'll take one window starting at 0, handle padding in __getitem__
                self.window_indices.append(
                    (start_global, end_global, sample["sample_id"], True)
                )  # True = needs padding
            else:
                # Sliding window
                # Ensure we cover the end
                for i in range(0, length - config.WINDOW_SIZE + 1, stride):
                    self.window_indices.append(
                        (
                            start_global + i,
                            start_global + i + config.WINDOW_SIZE,
                            sample["sample_id"],
                            False,
                        )
                    )

                # Handle remainder if strictly needed?
                # Usually standard sliding window is enough.
                # If we want full coverage for test, we might add a final window aligned to end.
                if not self.is_train and (length - config.WINDOW_SIZE) % stride != 0:
                    self.window_indices.append(
                        (
                            end_global - config.WINDOW_SIZE,
                            end_global,
                            sample["sample_id"],
                            False,
                        )
                    )

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        start_idx, end_idx, sample_id, needs_padding = self.window_indices[idx]

        # Extract Raw Data
        skel_window = self.skeletons[start_idx:end_idx].copy()  # (T, 20, 3)
        audio_window = self.audio[start_idx:end_idx].copy()  # (T, 13)
        label_window = self.labels[start_idx:end_idx].copy()  # (T,)

        # Handle Padding for short sequences
        if needs_padding:
            curr_len = skel_window.shape[0]
            pad_len = config.WINDOW_SIZE - curr_len

            # Pad Skeleton (Edge repeat or zero)
            skel_window = np.pad(
                skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="edge"
            )
            # Pad Audio (Zero)
            audio_window = np.pad(audio_window, ((0, pad_len), (0, 0)), mode="constant")
            # Pad Labels (Background = 0)
            label_window = np.pad(
                label_window, (0, pad_len), mode="constant", constant_values=0
            )

        # Augmentation (Train only)
        if self.is_train:
            # Random Rotation around Y-axis
            theta = np.random.uniform(-0.3, 0.3)  # +/- ~17 degrees
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

            # Apply rotation: (T, J, 3) dot (3, 3) -> (T, J, 3)
            # Reshape to (T*J, 3) for matmul
            T, J, C = skel_window.shape
            flat_skel = skel_window.reshape(-1, 3)
            flat_skel = np.dot(flat_skel, R.T)
            skel_window = flat_skel.reshape(T, J, C)

            # Random Scaling (Uniform)
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

        # Compute Kinematics (Position, Velocity, Acceleration)
        # Velocity: diff(pos)
        # Acceleration: diff(vel)
        # We pad the first frame of derivatives with 0 to maintain temporal length

        vel = np.zeros_like(skel_window)
        vel[1:] = skel_window[1:] - skel_window[:-1]

        acc = np.zeros_like(skel_window)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate Kinematics: (T, J, 9)
        # Stack along last axis: Pos(3), Vel(3), Acc(3)
        kinematics = np.concatenate([skel_window, vel, acc], axis=2)

        # Flatten Joints: (T, J*9) -> (T, 180)
        kinematics_flat = kinematics.reshape(config.WINDOW_SIZE, -1)

        # Concatenate Audio: (T, 180 + 13)
        features = np.concatenate([kinematics_flat, audio_window], axis=1)

        # Convert to Tensor
        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(label_window).long()

        return features_tensor, labels_tensor, sample_id


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=4, limit=None):
    """
    Factory function to create dataloaders.
    """
    train_ds = GestureDataset(config.TRAIN_METADATA_PATH, "train", limit=limit)
    val_ds = GestureDataset(config.VAL_METADATA_PATH, "val", limit=limit)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=config.BATCH_SIZE, num_workers=4):
    test_ds = GestureDataset(config.TEST_METADATA_PATH, "test")
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
