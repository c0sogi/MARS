import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from library.config import Config
from library.utils import load_or_compute, parse_mat_file


class FeatureExtractor:
    """
    Handles the extraction and alignment of multimodal features (Skeleton + Audio).
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.num_joints = Config.NUM_JOINTS

        # Initialize MFCC transform
        # We assume input audio is resampled to Config.AUDIO_SAMPLE_RATE if needed
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        ).to(self.device)

    def extract_skeleton_features(self, mat_data) -> np.ndarray:
        """
        Extracts, normalizes, and computes velocity for skeleton data.
        Returns: numpy array of shape (T, 120) -> (Pos + Vel)
        """
        if mat_data is None:
            return None

        # Access 'Video' via dictionary key since loadmat returns a dict
        if "Video" not in mat_data:
            return None

        video = mat_data["Video"]
        if not hasattr(video, "Frames") or not hasattr(video, "NumFrames"):
            return None

        num_frames = int(video.NumFrames)
        frames = video.Frames

        # Pre-allocate (T, 20, 3)
        # If frames is a single object (unlikely for video), wrap it
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        # Ensure we don't go out of bounds if metadata NumFrames differs from actual array length
        actual_len = len(frames)
        num_frames = min(num_frames, actual_len)

        skeleton_seq = np.zeros((num_frames, self.num_joints, 3), dtype=np.float32)

        for t in range(num_frames):
            frame_obj = frames[t]
            # Check if Skeleton exists and track state is valid
            if hasattr(frame_obj, "Skeleton") and frame_obj.Skeleton is not None:
                skel = frame_obj.Skeleton
                # Check for WorldPosition
                if hasattr(skel, "WorldPosition"):
                    wp = skel.WorldPosition
                    # WorldPosition might be a struct array or single struct
                    # We expect 20 joints.
                    # If it's a struct array, we iterate. If it's a struct with arrays X,Y,Z, we access.
                    # Based on description: "WorldPosition... X, Y, Z"
                    # And "JointsType" implies multiple joints.

                    # Heuristic to parse WorldPosition based on scipy.io behavior
                    try:
                        # Case A: WorldPosition is an array of structs (one per joint)
                        if (
                            isinstance(wp, (list, np.ndarray))
                            and len(wp) == self.num_joints
                        ):
                            for j in range(self.num_joints):
                                j_pos = wp[j]
                                skeleton_seq[t, j, 0] = j_pos.X
                                skeleton_seq[t, j, 1] = j_pos.Y
                                skeleton_seq[t, j, 2] = j_pos.Z
                        # Case B: WorldPosition is a single struct containing arrays for X, Y, Z?
                        # Less likely given the description, but possible.
                        # Let's assume Case A or flat struct access if single joint (unlikely).
                        # Fallback: if we can't iterate, leave as zeros (untracked).
                        pass
                    except Exception:
                        pass

            # Simple imputation: if frame is all zeros (untracked) and t > 0, copy previous
            if t > 0 and np.all(skeleton_seq[t] == 0):
                skeleton_seq[t] = skeleton_seq[t - 1]

        # 1. Normalization: Subtract HipCenter (Index 0)
        # Shape: (T, 20, 3)
        hip_centers = skeleton_seq[:, 0:1, :]  # (T, 1, 3)
        skeleton_seq_norm = skeleton_seq - hip_centers

        # 2. Flatten to (T, 60)
        skeleton_flat = skeleton_seq_norm.reshape(num_frames, -1)

        # 3. Velocity: (P_t - P_{t-1})
        # Pad first frame with zeros
        velocity = np.zeros_like(skeleton_flat)
        velocity[1:] = skeleton_flat[1:] - skeleton_flat[:-1]

        # 4. Concatenate: (T, 120)
        features = np.concatenate([skeleton_flat, velocity], axis=1)

        return features

    def extract_audio_features(
        self, audio_path: str, target_num_frames: int
    ) -> np.ndarray:
        """
        Extracts MFCCs and aligns them to the video frame count.
        Returns: numpy array of shape (T, 13)
        """
        if not os.path.exists(audio_path):
            # Return zeros if audio missing
            return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                ).to(waveform.device)
                waveform = resampler(waveform)

            waveform = waveform.to(self.device)

            # Compute MFCC -> (Channel, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)

            # We need to align 'time' dimension to 'target_num_frames'
            # Input to interpolate must be (Batch, Channels, Length)
            # MFCC is (1, n_mfcc, time) or (n_mfcc, time). Ensure 3D.
            if mfcc.dim() == 2:
                mfcc = mfcc.unsqueeze(0)

            # Interpolate
            # Mode 'linear' requires 3D input (N, C, L)
            mfcc_aligned = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )

            # Reshape to (T, n_mfcc)
            # Squeeze batch dim -> (n_mfcc, T) -> transpose -> (T, n_mfcc)
            features = mfcc_aligned.squeeze(0).transpose(0, 1).cpu().numpy()
            return features

        except Exception as e:
            # print(f"Audio processing error {audio_path}: {e}")
            return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

    def process_row(self, row) -> dict:
        """
        Process a single row from the metadata DataFrame.
        """
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton
        mat_data = parse_mat_file(data_path)
        skel_features = self.extract_skeleton_features(mat_data)

        if skel_features is None:
            # Fallback for completely broken files
            # Assume a minimal length or skip?
            # We'll create a dummy length based on audio or default
            # But usually data_path exists. If not, return None to filter out.
            return None

        num_frames = skel_features.shape[0]

        # 2. Load Audio and align
        audio_features = self.extract_audio_features(audio_path, num_frames)

        # 3. Concatenate
        # (T, 120) + (T, 13) -> (T, 133)
        final_features = np.concatenate([skel_features, audio_features], axis=1)

        # 4. Generate Labels (if available)
        labels = np.zeros(num_frames, dtype=np.int64)  # Default class 0 (background)

        # 'labels' column in row is a list of dicts (already parsed by metadata script?
        # No, metadata script saved it as JSON string. We need to parse if not parsed.)
        # However, the caller of this function will likely pass the row from a DF.
        # We should handle the parsing outside or check type.

        label_data = row["labels"]
        if isinstance(label_data, str):
            label_data = json.loads(label_data)

        if isinstance(label_data, list):
            for l in label_data:
                # Matlab 1-based indexing: Begin, End
                # Python 0-based: [Begin-1 : End]
                start_frame = max(0, int(l["begin"]) - 1)
                end_frame = min(num_frames, int(l["end"]))
                label_id = int(l["id"])

                if start_frame < end_frame:
                    labels[start_frame:end_frame] = label_id

        return {
            "sample_id": sample_id,
            "features": final_features.astype(np.float32),
            "labels": labels.astype(np.int64),
        }


def _compute_dataset(metadata_path: str, subset_size: int = None):
    """
    Internal function to compute features for a dataset.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if subset_size is not None:
        df = df.head(subset_size)

    extractor = FeatureExtractor()

    all_features = []
    all_labels = []
    all_ids = []

    # Iterate without progress bar as requested
    for _, row in df.iterrows():
        result = extractor.process_row(row)
        if result is not None:
            all_features.append(result["features"])
            all_labels.append(result["labels"])
            all_ids.append(result["sample_id"])

    # Convert lists to object arrays for storage
    # We use object arrays because lengths T vary per sample
    features_arr = np.array(all_features, dtype=object)
    labels_arr = np.array(all_labels, dtype=object)
    ids_arr = np.array(all_ids, dtype=str)

    return {"features": features_arr, "labels": labels_arr, "ids": ids_arr}


def get_train_data(load_cached_data: bool = True, subset_size: int = None):
    """
    Returns training data: {'features': [...], 'labels': [...], 'ids': [...]}
    """

    def compute_fn():
        return _compute_dataset(Config.TRAIN_METADATA_PATH, subset_size)

    # If subset_size is set, we shouldn't use the full cache.
    # For simplicity in this template, we'll append subset suffix to cache path if debugging.
    cache_path = Config.TRAIN_CACHE_PATH
    if subset_size is not None:
        cache_path = cache_path.replace(".npz", f"_subset_{subset_size}.npz")

    return load_or_compute(cache_path, compute_fn, load_cached_data)


def get_val_data(load_cached_data: bool = True, subset_size: int = None):
    """
    Returns validation data.
    """

    def compute_fn():
        return _compute_dataset(Config.VAL_METADATA_PATH, subset_size)

    cache_path = Config.VAL_CACHE_PATH
    if subset_size is not None:
        cache_path = cache_path.replace(".npz", f"_subset_{subset_size}.npz")

    return load_or_compute(cache_path, compute_fn, load_cached_data)


def get_test_data(load_cached_data: bool = True, subset_size: int = None):
    """
    Returns test data.
    """

    def compute_fn():
        return _compute_dataset(Config.TEST_METADATA_PATH, subset_size)

    cache_path = Config.TEST_CACHE_PATH
    if subset_size is not None:
        cache_path = cache_path.replace(".npz", f"_subset_{subset_size}.npz")

    return load_or_compute(cache_path, compute_fn, load_cached_data)
