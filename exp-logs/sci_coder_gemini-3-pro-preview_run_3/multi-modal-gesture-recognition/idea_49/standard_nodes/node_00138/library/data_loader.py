import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    Config,
    robust_load_mat,
    get_skeleton_data,
    process_audio,
    augment_skeleton,
    compute_kinematics,
)


class PolymorphicParser:
    """
    Handles robust parsing of .mat files which may have inconsistent structures.
    """

    @staticmethod
    def parse_skeleton(mat_path, num_frames):
        """
        Loads a .mat file and extracts skeleton data.

        Args:
            mat_path (str): Path to the .mat file.
            num_frames (int): Number of frames to extract.

        Returns:
            np.ndarray: Skeleton data of shape (num_frames, 20, 3).
        """
        mat = robust_load_mat(mat_path)
        if mat is None:
            # Return zero-filled array if loading fails, to maintain pipeline integrity
            return np.zeros((num_frames, 20, 3), dtype=np.float32)

        return get_skeleton_data(mat, num_frames)


class AudioProcessor:
    """
    Handles audio feature extraction.
    """

    @staticmethod
    def process(audio_path, target_frames):
        """
        Extracts MFCC features and aligns them to the video frame count.

        Args:
            audio_path (str): Path to the audio file.
            target_frames (int): Number of video frames for alignment.

        Returns:
            np.ndarray: MFCC features of shape (target_frames, N_MFCC).
        """
        return process_audio(audio_path, target_frames)


class SkeletonProcessor:
    """
    Handles skeleton preprocessing, augmentation, and kinematic derivation.
    """

    @staticmethod
    def process(skeleton, augment=False):
        """
        Processes raw skeleton data into kinematic features.

        Args:
            skeleton (np.ndarray): Raw skeleton data (T, 20, 3).
            augment (bool): Whether to apply data augmentation (rotation, scale, noise).

        Returns:
            np.ndarray: Kinematic features (Position, Velocity, Acceleration).
        """
        # 1. Augmentation (Geometric + Noise Injection)
        if augment:
            # Noise sigma 0.01 is specified in the idea description
            skeleton_aug = augment_skeleton(
                skeleton, rotation=True, scale=True, noise_sigma=0.01
            )
        else:
            skeleton_aug = skeleton

        # 2. Kinematic Derivation (Centering + Derivatives)
        # compute_kinematics handles root-relative centering internally
        kinematics = compute_kinematics(skeleton_aug)

        return kinematics


class GestureDataset(Dataset):
    """
    Dataset class for multi-modal gesture recognition.
    Handles caching, windowing, and on-the-fly preprocessing.
    """

    def __init__(self, csv_file, mode="train", load_cached_data=True, max_samples=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            max_samples (int, optional): Limit the number of samples for debugging.
        """
        self.mode = mode
        self.df = pd.read_csv(csv_file)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]

        self.samples = []

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache file path
        cache_filename = f"dataset_{mode}.npz"
        if max_samples is not None:
            cache_filename = f"dataset_{mode}_{max_samples}.npz"

        self.cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Caching Logic
        data_loaded = False
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading {mode} data from cache: {self.cache_path}")
                data = np.load(self.cache_path, allow_pickle=True)
                self.samples = data["samples"]
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing data.")

        if not data_loaded:
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads raw files, extracts raw features, and saves to cache.
        """
        print(f"Processing {self.mode} data from scratch...")
        processed_samples = []

        for idx, row in self.df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Load .mat file to get frame count
            mat = robust_load_mat(data_path)
            if mat is None:
                continue

            if "Video" not in mat:
                continue

            video = mat["Video"]
            if isinstance(video, np.ndarray) and video.ndim == 0:
                video = video.item()

            num_frames = int(video.NumFrames) if hasattr(video, "NumFrames") else 0
            if num_frames == 0:
                continue

            # Parse Raw Data using helper classes
            # We store RAW skeleton to allow dynamic augmentation during training
            raw_skeleton = PolymorphicParser.parse_skeleton(data_path, num_frames)

            # Extract Audio Features (deterministic)
            mfcc = AudioProcessor.process(audio_path, num_frames)

            # Process Labels
            labels = np.zeros(num_frames, dtype=np.int64)
            if self.mode != "test":
                try:
                    label_list = json.loads(row["labels"])
                    for l in label_list:
                        # Ensure indices are within bounds
                        start = max(0, int(l["begin"]) - 1)
                        end = min(num_frames, int(l["end"]))
                        labels[start:end] = int(l["id"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            processed_samples.append(
                {
                    "id": sample_id,
                    "skeleton": raw_skeleton,
                    "audio": mfcc,
                    "labels": labels,
                    "length": num_frames,
                }
            )

        self.samples = processed_samples

        # Save to cache
        print(f"Saving {len(self.samples)} samples to cache: {self.cache_path}")
        np.savez_compressed(self.cache_path, samples=self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        skeleton = sample["skeleton"]
        audio = sample["audio"]
        labels = sample["labels"]
        length = sample["length"]
        sample_id = sample["id"]

        # Training: Sliding Window + Augmentation
        if self.mode == "train":
            # Random Windowing
            if length > Config.WINDOW_SIZE:
                start = np.random.randint(0, length - Config.WINDOW_SIZE)
                end = start + Config.WINDOW_SIZE
            else:
                start = 0
                end = length

            skel_window = skeleton[start:end]
            audio_window = audio[start:end]
            label_window = labels[start:end]

            # Padding if shorter than window
            if len(skel_window) < Config.WINDOW_SIZE:
                pad_len = Config.WINDOW_SIZE - len(skel_window)
                # Pad skeleton: (Time, Joints, Coords)
                skel_window = np.pad(
                    skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
                # Pad audio: (Time, Features)
                audio_window = np.pad(
                    audio_window, ((0, pad_len), (0, 0)), mode="constant"
                )
                # Pad labels
                label_window = np.pad(
                    label_window, (0, pad_len), mode="constant", constant_values=0
                )

            # Apply Augmentation and Kinematics via SkeletonProcessor
            kinematics = SkeletonProcessor.process(skel_window, augment=True)

            # Concatenate Audio
            features = np.concatenate([kinematics, audio_window], axis=1)

            return {
                "features": torch.tensor(features, dtype=torch.float32),
                "labels": torch.tensor(label_window, dtype=torch.long),
            }

        # Validation/Test: Full Sequence (or handled by batch_size=1) + No Augmentation
        else:
            # Apply Kinematics without Augmentation
            kinematics = SkeletonProcessor.process(skeleton, augment=False)

            # Concatenate Audio
            features = np.concatenate([kinematics, audio], axis=1)

            return {
                "features": torch.tensor(features, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "id": sample_id,
            }
