import os
import json
import numpy as np
import scipy.io
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SKELETON_JOINTS,
    AUDIO_MFCC_DIM,
    BACKGROUND_CLASS_ID,
    NUM_CLASSES,
)

# Joint Name Mapping based on description order
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
JOINT_MAP = {name.lower(): i for i, name in enumerate(JOINT_NAMES)}


class DataParser:
    """
    Handles robust ingestion of raw multi-modal data and caching.
    """

    @staticmethod
    def load_polymorphic_mat(mat_path):
        """
        Robustly loads skeleton data from .mat file, handling various internal structures.

        Args:
            mat_path (str): Path to the .mat file.

        Returns:
            np.ndarray: Skeleton data of shape (NumFrames, 20, 3) or None if failure.
        """
        try:
            # Load mat file, treating structs as objects
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            # Access top-level variable as dictionary key (Cite debug_lesson_2)
            if "Video" not in mat:
                return None

            video = mat["Video"]

            # Unwrap 0-d array wrapper if present (Cite debug_lesson_16)
            if isinstance(video, np.ndarray) and video.ndim == 0:
                video = video.item()

            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames

            # Handle case where Frames is a single object (scalar struct)
            if not isinstance(frames, (list, np.ndarray)):
                frames = [frames]

            num_frames = len(frames)
            # Initialize with zeros
            skeleton_data = np.zeros((num_frames, SKELETON_JOINTS, 3), dtype=np.float32)

            for f_idx, frame in enumerate(frames):
                # Check if frame has Skeleton attribute
                if not hasattr(frame, "Skeleton"):
                    continue

                skel_obj = frame.Skeleton

                # Skip if empty or None
                if skel_obj is None:
                    continue

                # Determine if skel_obj is an array (multiple joints) or a single struct
                joints_list = []

                # Case 1: Array of joint objects
                if isinstance(skel_obj, (list, np.ndarray)):
                    if len(skel_obj) > 0:
                        joints_list = skel_obj
                # Case 2: Single object (rare, but possible if only 1 joint tracked?)
                elif hasattr(skel_obj, "JointsType"):
                    joints_list = [skel_obj]

                # Iterate through found joints and map them
                for joint in joints_list:
                    # Extract Joint Type
                    j_type = None
                    if hasattr(joint, "JointsType"):
                        j_type = str(joint.JointsType)

                    # Extract World Position
                    pos = [0.0, 0.0, 0.0]
                    if hasattr(joint, "WorldPosition"):
                        wp = joint.WorldPosition
                        # Handle WorldPosition as struct (X,Y,Z) or array
                        if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                            pos = [float(wp.X), float(wp.Y), float(wp.Z)]
                        elif isinstance(wp, (list, np.ndarray)) and len(wp) >= 3:
                            pos = [float(wp[0]), float(wp[1]), float(wp[2])]

                    # Map to canonical index
                    if j_type:
                        idx = JOINT_MAP.get(j_type.lower())
                        if idx is not None:
                            skeleton_data[f_idx, idx] = pos

            return skeleton_data

        except Exception as e:
            # In production we might log this, but for now we return None to skip sample
            return None

    @staticmethod
    def extract_audio_features(audio_path, num_target_frames):
        """
        Extracts MFCC features and aligns them to video frames via interpolation.

        Args:
            audio_path (str): Path to the .wav file.
            num_target_frames (int): Number of video frames to align to.

        Returns:
            np.ndarray: Audio features of shape (num_target_frames, AUDIO_MFCC_DIM).
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Define MFCC transform
            mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=AUDIO_MFCC_DIM,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            # Compute MFCC: (Channels, n_mfcc, time)
            mfcc = mfcc_transform(waveform)

            # Average over audio channels if stereo to get mono features
            if mfcc.shape[0] > 1:
                mfcc = torch.mean(mfcc, dim=0, keepdim=True)

            # Current shape: (1, n_mfcc, time_steps)

            # Interpolate to match the exact number of video frames
            if num_target_frames > 0:
                mfcc = torch.nn.functional.interpolate(
                    mfcc, size=num_target_frames, mode="linear", align_corners=False
                )

            # Reshape to (num_target_frames, n_mfcc)
            mfcc = mfcc.squeeze(0).transpose(0, 1)

            return mfcc.numpy()

        except Exception:
            # Return zero features if audio fails
            return np.zeros((num_target_frames, AUDIO_MFCC_DIM), dtype=np.float32)

    @classmethod
    def process_dataset(
        cls, metadata_path, split_name, load_cache=True, debug_size=None
    ):
        """
        Processes the dataset defined in metadata CSV.
        Loads raw files, parses them, pads them to a common length, and caches as .npz.

        Args:
            metadata_path (str): Path to the metadata CSV.
            split_name (str): Name of the split (train/val/test) for cache naming.
            load_cache (bool): Whether to attempt loading from cache.
            debug_size (int, optional): Limit number of samples for debugging.

        Returns:
            dict: Dictionary containing 'skeletons', 'audio', 'labels', 'lengths', 'sample_ids'.
        """
        cache_file = os.path.join(CACHE_DIR, f"dataset_{split_name}.npz")

        # 1. Try Loading Cache
        if load_cache and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            try:
                # allow_pickle=False ensures we are loading clean numeric data
                data = np.load(cache_file, allow_pickle=False)
                # Check for required keys
                required_keys = [
                    "skeletons",
                    "audio",
                    "labels",
                    "lengths",
                    "sample_ids",
                ]
                if all(k in data for k in required_keys):
                    return dict(data)
            except Exception as e:
                print(f"Cache load failed ({e}). Recomputing from scratch...")

        # 2. Compute from Scratch
        print(f"Processing dataset {split_name} from scratch...")
        df = pd.read_csv(metadata_path)

        if debug_size is not None:
            df = df.head(debug_size)

        skeletons_list = []
        audio_list = []
        labels_list = []
        lengths_list = []
        ids_list = []

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(INPUT_DIR, row["data_path"])
            audio_path = os.path.join(INPUT_DIR, row["audio_path"])

            # A. Load Skeleton
            skel = cls.load_polymorphic_mat(mat_path)
            if skel is None:
                # If skeleton parsing fails, we cannot use this sample
                continue

            num_frames = skel.shape[0]
            if num_frames == 0:
                continue

            # B. Load Audio (Aligned to Skeleton frames)
            aud = cls.extract_audio_features(audio_path, num_frames)

            # C. Process Labels (Convert JSON intervals to Frame-wise Mask)
            label_seq = np.zeros(num_frames, dtype=np.int32)  # Default 0 (Background)

            if "labels" in row and isinstance(row["labels"], str):
                try:
                    anns = json.loads(row["labels"])
                    for ann in anns:
                        gid = ann["id"]
                        # Convert 1-based indexing (Matlab) to 0-based
                        start = max(0, ann["begin"] - 1)
                        end = min(num_frames, ann["end"])
                        if start < end:
                            label_seq[start:end] = gid
                except:
                    # If labels are malformed or empty (test set), keep as zeros
                    pass

            skeletons_list.append(skel)
            audio_list.append(aud)
            labels_list.append(label_seq)
            lengths_list.append(num_frames)
            ids_list.append(sample_id)

        # 3. Pad and Stack
        if not lengths_list:
            return {}

        max_len = max(lengths_list)
        N = len(lengths_list)

        # Initialize dense arrays
        dense_skel = np.zeros((N, max_len, SKELETON_JOINTS, 3), dtype=np.float32)
        dense_audio = np.zeros((N, max_len, AUDIO_MFCC_DIM), dtype=np.float32)
        dense_labels = np.zeros((N, max_len), dtype=np.int32)

        for i in range(N):
            L = lengths_list[i]
            dense_skel[i, :L] = skeletons_list[i]
            dense_audio[i, :L] = audio_list[i]
            dense_labels[i, :L] = labels_list[i]

        # 4. Save to Cache
        # Convert IDs to unicode numpy array to avoid pickle
        dense_ids = np.array(ids_list, dtype="U")
        dense_lengths = np.array(lengths_list, dtype=np.int32)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            cache_file,
            skeletons=dense_skel,
            audio=dense_audio,
            labels=dense_labels,
            lengths=dense_lengths,
            sample_ids=dense_ids,
        )

        return {
            "skeletons": dense_skel,
            "audio": dense_audio,
            "labels": dense_labels,
            "lengths": dense_lengths,
            "sample_ids": dense_ids,
        }
