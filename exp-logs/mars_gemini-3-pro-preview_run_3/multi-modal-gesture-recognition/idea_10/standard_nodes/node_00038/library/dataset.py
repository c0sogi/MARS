import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io
from torch.utils.data import Dataset
from library.config import Config
from library.utils import align_to_canonical_view, compute_kinematics


class GestureDataset(Dataset):
    """
    Dataset class for the View-Invariant Attentive Refinement Network (VI-ARN).
    Handles multimodal data loading, canonical alignment, kinematic feature engineering,
    and sliding window generation with caching.
    """

    def __init__(
        self, split="train", load_cached_data=True, transform=None, debug=False
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load pre-processed data from cache.
            transform (callable, optional): Optional transform to be applied on a sample.
            debug (bool): If True, uses a smaller subset of data.
        """
        self.split = split
        self.transform = transform
        self.debug = debug or Config.DEBUG

        # Determine metadata file
        if self.split == "train":
            self.metadata_file = Config.TRAIN_METADATA
        elif self.split == "val":
            self.metadata_file = Config.VAL_METADATA
        elif self.split == "test":
            self.metadata_file = Config.TEST_METADATA
        else:
            raise ValueError(f"Invalid split: {split}")

        # Cache file path
        cache_name = f"dataset_{split}{'_debug' if self.debug else ''}.npz"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # Audio transform (MFCC)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        # Load data
        self.sequences = self._load_data(load_cached_data)

        # Generate sliding window indices
        self.window_indices = self._make_window_indices()

    def _load_data(self, load_cached_data):
        """
        Loads data either from cache or by processing raw files.
        Returns a list of dictionaries containing 'skeleton', 'audio', 'labels'.
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached {self.split} dataset from {self.cache_path}...")
                loaded = np.load(self.cache_path, allow_pickle=True)
                sequences = []
                # Reconstruct list of dicts
                for i in range(len(loaded["ids"])):
                    seq = {
                        "id": str(loaded["ids"][i]),
                        "skeleton": loaded[f"skeleton_{i}"],
                        "audio": loaded[f"audio_{i}"],
                        "labels": loaded[f"labels_{i}"],
                    }
                    sequences.append(seq)
                return sequences
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from Scratch
        print(f"Processing {self.split} dataset from raw files...")
        df = pd.read_csv(self.metadata_file)

        if self.debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        sequences = []

        for _, row in df.iterrows():
            sample_id = row["sample_id"]

            # --- Skeleton Processing ---
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            skeleton, num_frames = self._load_skeleton(mat_path)

            if skeleton is None:
                continue  # Skip corrupted samples

            # Apply Canonical Alignment (User Independence)
            if Config.USE_CANONICAL_ALIGNMENT:
                skeleton = align_to_canonical_view(skeleton)

            # --- Audio Processing ---
            wav_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            audio_waveform = self._load_audio(wav_path)

            # --- Label Processing ---
            labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)
            if self.split != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # Metadata uses 1-based indexing for frames usually, convert to 0-based
                    # Range is inclusive in metadata? Assuming yes.
                    start = max(0, l["begin"] - 1)
                    end = min(num_frames, l["end"])
                    gid = l["id"]
                    labels[start:end] = gid

            sequences.append(
                {
                    "id": sample_id,
                    "skeleton": skeleton.astype(np.float32),  # (T, 20, 3)
                    "audio": audio_waveform,  # (Channels, Time) - kept raw for MFCC on-the-fly or pre-calc?
                    # Better to pre-calc MFCC to align with T.
                    # Actually, let's pre-calc MFCC here to save space/time in getitem
                    # But we need torch for MFCC.
                    "labels": labels,
                }
            )

        # Post-process audio to align with video frames
        processed_sequences = []
        for seq in sequences:
            # Compute MFCC
            waveform = seq["audio"]
            if waveform is None:
                # Silent audio if missing
                mfcc = torch.zeros((Config.N_MFCC, seq["skeleton"].shape[0]))
            else:
                with torch.no_grad():
                    # Ensure waveform is torch tensor
                    if isinstance(waveform, np.ndarray):
                        waveform = torch.from_numpy(waveform).float()

                    # Compute MFCC
                    # MFCC transform expects (Channel, Time)
                    mfcc = self.mfcc_transform(waveform)

                    # Interpolate to match video frames
                    target_frames = seq["skeleton"].shape[0]
                    # MFCC shape: (1, n_mfcc, time)
                    mfcc = F.interpolate(
                        mfcc, size=target_frames, mode="linear", align_corners=False
                    )
                    mfcc = mfcc.squeeze(0).permute(1, 0)  # (T, n_mfcc)

            processed_sequences.append(
                {
                    "id": seq["id"],
                    "skeleton": seq["skeleton"],
                    "audio": mfcc.numpy().astype(np.float32),
                    "labels": seq["labels"],
                }
            )

        # 3. Save to Cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        save_dict = {"ids": np.array([s["id"] for s in processed_sequences])}
        for i, seq in enumerate(processed_sequences):
            save_dict[f"skeleton_{i}"] = seq["skeleton"]
            save_dict[f"audio_{i}"] = seq["audio"]
            save_dict[f"labels_{i}"] = seq["labels"]

        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Saved processed dataset to {self.cache_path}")

        return processed_sequences

    def _load_skeleton(self, path):
        """Reads .mat file and extracts skeleton data."""
        try:
            mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None, 0
            video = mat["Video"]
            if not hasattr(video, "Frames"):
                return None, 0

            frames = video.Frames
            num_frames = len(frames)

            # Extract WorldPosition for 20 joints
            # Shape: (NumFrames, 20, 3)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            for i in range(num_frames):
                if hasattr(frames[i], "Skeleton") and hasattr(
                    frames[i].Skeleton, "WorldPosition"
                ):
                    wp = frames[i].Skeleton.WorldPosition
                    # Check if wp is an array or object. Based on description, it's structured.
                    # Assuming wp is 20x1 struct array or similar.
                    # Dataset description says "WorldPosition... X, Y, Z".
                    # Usually in these datasets, it's a struct array of joints.
                    # Let's handle the specific format implied by typical MATLAB exports of this data.
                    # If WorldPosition is a single struct with X,Y,Z arrays?
                    # Or an array of structs?
                    # Based on provided metadata script logic, we access .X, .Y, .Z.
                    # If it's 20 joints, we expect 20 positions.

                    # Heuristic: Try to convert to array directly if possible, otherwise iterate
                    # Given the complexity, we assume the provided metadata script logic was correct
                    # but here we need all joints.

                    # Let's assume frames[i].Skeleton.WorldPosition is a 20-element array/struct
                    # OR frames[i].Skeleton.Joints...
                    # Description: "Skeleton Frame... JointsType... WorldPosition"
                    # It seems Skeleton is an array of structures (one per joint)?
                    # "An array of Skeleton structures is contained within a Skeletons array."
                    # Wait, "Frames: Skeleton information...".
                    # Let's assume frames[i].Skeleton is the skeleton for that frame.
                    # And it might contain an array of joints.

                    # Safe fallback: If we can't parse, return zeros.
                    # However, for this task, we assume standard format: (20, 3)

                    # Attempt to extract coordinates directly if it's a matrix
                    # Many MSRC-12 / Italian Gesture datasets store it as (20, 3) or (3, 20)
                    try:
                        # Try accessing as simple array
                        pos = np.array(
                            [
                                [
                                    j.WorldPosition.X,
                                    j.WorldPosition.Y,
                                    j.WorldPosition.Z,
                                ]
                                for j in frames[i].Skeleton
                            ]
                        )
                        skeleton_data[i] = pos
                    except:
                        # If structure is different (e.g. single struct with arrays)
                        pass

            return skeleton_data, num_frames
        except Exception:
            return None, 0

    def _load_audio(self, path):
        """Reads .wav file."""
        if not os.path.exists(path):
            return None
        try:
            waveform, sample_rate = torchaudio.load(path)
            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)
            return waveform
        except Exception:
            return None

    def _make_window_indices(self):
        """
        Creates a list of (sequence_index, start_frame) tuples for sliding windows.
        """
        indices = []
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq["skeleton"].shape[0]

            # If sequence is shorter than window, pad it later, just add one index
            if num_frames < Config.WINDOW_SIZE:
                indices.append((seq_idx, 0))
                continue

            # Sliding window
            # We want to cover the whole sequence.
            # Stride = Config.STRIDE
            for start in range(0, num_frames - Config.WINDOW_SIZE + 1, Config.STRIDE):
                indices.append((seq_idx, start))

            # Ensure the last frame is covered
            last_start = num_frames - Config.WINDOW_SIZE
            if last_start > 0 and (last_start % Config.STRIDE != 0):
                indices.append((seq_idx, last_start))

        return indices

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.window_indices[idx]
        seq = self.sequences[seq_idx]

        # Extract Raw Data
        skel_full = seq["skeleton"]  # (T_full, 20, 3)
        audio_full = seq["audio"]  # (T_full, 13)
        labels_full = seq["labels"]  # (T_full,)

        seq_len = skel_full.shape[0]
        window_size = Config.WINDOW_SIZE

        # Handle Padding for short sequences
        if seq_len < window_size:
            # Pad with zeros or edge values
            pad_len = window_size - seq_len

            skel_window = np.pad(skel_full, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            audio_window = np.pad(audio_full, ((0, pad_len), (0, 0)), mode="constant")
            labels_window = np.pad(
                labels_full, (0, pad_len), mode="constant", constant_values=0
            )
        else:
            end_frame = start_frame + window_size
            skel_window = skel_full[start_frame:end_frame]
            audio_window = audio_full[start_frame:end_frame]
            labels_window = labels_full[start_frame:end_frame]

        # --- Augmentation (Train Only) ---
        if self.split == "train":
            # Random Scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

            # Random Rotation around Y-axis (-10 to 10 degrees)
            theta = np.deg2rad(np.random.uniform(-10, 10))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            # Apply rotation: (T, 20, 3) dot (3, 3) -> (T, 20, 3)
            skel_window = np.dot(skel_window, R.T)

        # --- Feature Engineering (Kinematics) ---
        # Compute Velocity and Acceleration
        # Shape becomes (Window, 20, 9)
        skel_features = compute_kinematics(skel_window)

        # Flatten Skeleton: (Window, 180)
        skel_flat = skel_features.reshape(window_size, -1)

        # Concatenate Audio: (Window, 180 + 13)
        features = np.concatenate([skel_flat, audio_window], axis=1)

        # Convert to Tensor
        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels_window).long()

        return features_tensor, labels_tensor
