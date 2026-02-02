import os
import json
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# ==========================================
# Helper Functions
# ==========================================


def log_modulus_transform(x):
    """
    Applies the log-modulus transformation: f(x) = sign(x) * log(1 + |x|)
    """
    return np.sign(x) * np.log1p(np.abs(x))


def parse_skeleton_data(mat_path, num_frames):
    """
    Robustly parses the .mat file to extract skeleton joint positions.
    Handles polymorphic structures (struct, object, cell arrays).
    Returns: numpy array of shape (num_frames, 20, 3)
    """
    try:
        # Load mat file
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        # Initialize container
        # 20 joints, 3 coordinates (X, Y, Z)
        skeletons = np.zeros((num_frames, config.JOINTS_COUNT, 3), dtype=np.float32)

        if "Video" not in mat._fieldnames:
            return skeletons

        video = mat.Video
        if not hasattr(video, "Frames"):
            return skeletons

        frames = video.Frames

        # Handle case where Frames is a single object or list
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        # Iterate through frames
        # Use min(len(frames), num_frames) to avoid index errors
        limit = min(len(frames), num_frames)

        for i in range(limit):
            frame_data = frames[i]

            # Check if Skeleton exists and is not empty/zero
            if not hasattr(frame_data, "Skeleton"):
                continue

            skel = frame_data.Skeleton

            # Check if skel is valid (sometimes it's 0 or empty if no user)
            if isinstance(skel, (int, float)) or skel is None:
                continue

            # Check for WorldPosition
            if not hasattr(skel, "WorldPosition"):
                continue

            wp = skel.WorldPosition

            # Extract coordinates based on structure type
            # Case A: wp is an object with X, Y, Z attributes (arrays or scalars)
            if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                try:
                    # Ensure we are getting 20 joints
                    # If X is scalar, this might be a single joint or error, but usually it's 20x1
                    x = np.atleast_1d(wp.X)
                    y = np.atleast_1d(wp.Y)
                    z = np.atleast_1d(wp.Z)

                    if len(x) == config.JOINTS_COUNT:
                        skeletons[i, :, 0] = x
                        skeletons[i, :, 1] = y
                        skeletons[i, :, 2] = z
                except Exception:
                    pass

            # Case B: wp might be a matrix directly (rare but possible in some mat versions)
            elif isinstance(wp, np.ndarray) and wp.shape == (config.JOINTS_COUNT, 3):
                skeletons[i] = wp

        # Simple imputation for missing frames (forward fill then backward fill)
        # Identify empty frames (all zeros)
        # Note: (0,0,0) is technically possible but unlikely for all joints.
        # We assume if sum of absolute values is 0, it's missing.
        valid_mask = np.sum(np.abs(skeletons), axis=(1, 2)) > 1e-6

        if np.any(valid_mask):
            # Forward fill
            last_valid = skeletons[np.argmax(valid_mask)]  # First valid
            for i in range(num_frames):
                if valid_mask[i]:
                    last_valid = skeletons[i]
                else:
                    skeletons[i] = last_valid

            # Backward fill (for start gaps)
            first_valid_idx = np.argmax(valid_mask)
            if first_valid_idx > 0:
                skeletons[:first_valid_idx] = skeletons[first_valid_idx]

        return skeletons

    except Exception as e:
        # print(f"Error parsing {mat_path}: {e}")
        return np.zeros((num_frames, config.JOINTS_COUNT, 3), dtype=np.float32)


def process_audio(audio_path, num_video_frames):
    """
    Loads audio, extracts MFCCs, aligns to video frames, and applies instance normalization.
    Returns: numpy array of shape (num_video_frames, n_mfcc)
    """
    target_len = num_video_frames

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Extract MFCC
        # We use a hop length that roughly approximates video frame rate,
        # but we will resample strictly later.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            melkwargs={
                "n_fft": config.N_FFT,
                "n_mels": 64,
                "hop_length": config.HOP_LENGTH,
                "center": False,
            },
        )

        mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).numpy()  # Shape: (n_mfcc, time)

        # Align with video frames using linear interpolation
        # Input shape for resample: (n_mfcc, time)
        # Output shape: (n_mfcc, num_video_frames)
        if mfcc.shape[1] > 0:
            mfcc_resampled = scipy.signal.resample(mfcc, target_len, axis=1)
            mfcc_out = mfcc_resampled.T  # (num_video_frames, n_mfcc)
        else:
            mfcc_out = np.zeros((target_len, config.N_MFCC), dtype=np.float32)

        # Instance Normalization (per sequence)
        # Mean=0, Std=1
        mean = np.mean(mfcc_out, axis=0, keepdims=True)
        std = np.std(mfcc_out, axis=0, keepdims=True)
        mfcc_out = (mfcc_out - mean) / (std + 1e-6)

        return mfcc_out.astype(np.float32)

    except Exception as e:
        # print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((target_len, config.N_MFCC), dtype=np.float32)


def generate_dense_labels(labels_list, num_frames):
    """
    Converts sparse label list to dense frame-wise label array.
    """
    dense_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)

    for label in labels_list:
        gid = label["id"]
        start = max(0, label["begin"] - 1)  # 1-based to 0-based
        end = min(num_frames, label["end"])

        if start < end:
            dense_labels[start:end] = gid

    return dense_labels


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, is_train=False, load_cached_data=True, debug=config.DEBUG
    ):
        """
        Args:
            metadata_path: Path to csv metadata.
            is_train: Boolean, enables augmentation.
            load_cached_data: Boolean, enables loading from .npz cache.
            debug: Boolean, loads subset of data.
        """
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_path)

        if debug:
            self.metadata = self.metadata.iloc[: config.DEBUG_SUBSET_SIZE]

        # Parse JSON labels
        self.metadata["parsed_labels"] = self.metadata["labels"].apply(json.loads)

        # Cache setup
        self.cache_name = os.path.basename(metadata_path).replace(".csv", "")
        if debug:
            self.cache_name += "_debug"
        self.cache_path = os.path.join(
            config.CACHE_DIR, f"dataset_{self.cache_name}.npz"
        )

        # Data containers
        self.sequences = (
            []
        )  # List of dicts: {'skel': (T,20,3), 'audio': (T,13), 'label': (T,)}
        self.windows = []  # List of tuples: (seq_idx, start_frame)

        # Load Data
        self._load_data(load_cached_data)

        # Create Windows
        self._create_windows()

    def _load_data(self, load_cached):
        # Try loading from cache
        if load_cached and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached dataset from {self.cache_path}...")
                data = np.load(self.cache_path, allow_pickle=True)
                # Reconstruct list of dicts
                # We stored them as object array or separate arrays?
                # Storing variable length sequences in npz is tricky.
                # We assume we stored a flat object array of dicts.
                loaded_seqs = data["sequences"]
                self.sequences = list(loaded_seqs)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Processing {len(self.metadata)} sequences...")
        processed_sequences = []

        for idx, row in self.metadata.iterrows():
            # Paths
            rgb_path = os.path.join(config.INPUT_DIR, row["rgb_path"])
            depth_path = os.path.join(config.INPUT_DIR, row["depth_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])
            data_path = os.path.join(config.INPUT_DIR, row["data_path"])

            # 1. Determine Num Frames (Ground Truth)
            # We rely on the .mat file or video file.
            # Ideally .mat has NumFrames.
            num_frames = 0
            try:
                mat_info = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                if hasattr(mat_info, "Video") and hasattr(mat_info.Video, "NumFrames"):
                    num_frames = int(mat_info.Video.NumFrames)
            except:
                pass

            # Fallback to RGB video if mat fails
            if num_frames == 0 and os.path.exists(rgb_path):
                # We can't use cv2 here easily without dependency, but we have pims or similar?
                # Actually, let's assume mat file works or use audio length * fps estimate.
                # Let's trust the mat file parsing in parse_skeleton_data to handle length implicitly
                # by reading the array length if NumFrames is bad.
                # But we need a target length for alignment.
                # Let's use a safe default or try to read from parse_skeleton.
                pass

            # 2. Parse Skeleton
            # If num_frames is unknown, parse_skeleton_data will try to infer from array size
            # but we pass a large number or handle inside.
            # Let's read the mat file properly in parse_skeleton first to get frames.
            skeletons = parse_skeleton_data(
                data_path, num_frames if num_frames > 0 else 10000
            )

            # Update num_frames based on actual skeleton data if it was 0
            real_frames = skeletons.shape[0]
            if num_frames == 0:
                num_frames = real_frames
            else:
                # Truncate or pad skeleton to match num_frames
                if real_frames > num_frames:
                    skeletons = skeletons[:num_frames]
                elif real_frames < num_frames:
                    pad = np.zeros(
                        (num_frames - real_frames, config.JOINTS_COUNT, 3),
                        dtype=np.float32,
                    )
                    skeletons = np.concatenate([skeletons, pad], axis=0)

            # 3. Process Audio
            audio_feat = process_audio(audio_path, num_frames)

            # 4. Process Labels
            labels = generate_dense_labels(row["parsed_labels"], num_frames)

            processed_sequences.append(
                {
                    "skel": skeletons.astype(np.float32),
                    "audio": audio_feat.astype(np.float32),
                    "label": labels.astype(np.int64),
                }
            )

        self.sequences = processed_sequences

        # Save to cache
        try:
            np.savez_compressed(
                self.cache_path, sequences=np.array(self.sequences, dtype=object)
            )
            print(f"Saved dataset to {self.cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _create_windows(self):
        self.windows = []
        for seq_idx, seq_data in enumerate(self.sequences):
            num_frames = seq_data["label"].shape[0]

            # Sliding window
            # We want to cover the whole sequence.
            # If sequence is shorter than window, pad it?
            # Or just skip? Usually sequences are long enough.
            # If shorter, we pad.

            if num_frames < config.WINDOW_SIZE:
                # Special case: single window with padding handled in __getitem__
                self.windows.append((seq_idx, 0))
            else:
                # Standard sliding
                for start in range(
                    0, num_frames - config.WINDOW_SIZE + 1, config.STRIDE
                ):
                    self.windows.append((seq_idx, start))

                # Handle remainder if significant?
                # Standard approach: strict windows.

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]
        seq_data = self.sequences[seq_idx]

        full_skel = seq_data["skel"]  # (T_full, 20, 3)
        full_audio = seq_data["audio"]  # (T_full, 13)
        full_label = seq_data["label"]  # (T_full,)

        seq_len = full_label.shape[0]
        w_size = config.WINDOW_SIZE

        # Determine extraction range
        # We need start-1 for velocity calculation
        extract_start = start_frame - 1
        extract_end = start_frame + w_size

        # Handle boundary conditions for extraction
        pad_left = 0
        if extract_start < 0:
            pad_left = abs(extract_start)
            extract_start = 0

        # Slice
        skel_window = full_skel[extract_start:extract_end].copy()

        # Pad left if needed (replicate first frame)
        if pad_left > 0:
            pad = np.repeat(skel_window[[0]], pad_left, axis=0)
            skel_window = np.concatenate([pad, skel_window], axis=0)

        # Handle short sequences (pad right)
        current_len = skel_window.shape[0]
        needed_len = w_size + 1  # +1 for previous frame
        if current_len < needed_len:
            pad_right = needed_len - current_len
            pad = np.zeros((pad_right, config.JOINTS_COUNT, 3), dtype=np.float32)
            skel_window = np.concatenate([skel_window, pad], axis=0)

        # --- Augmentation (Skeleton) ---
        if self.is_train:
            # Random Rotation around Y axis
            theta = np.random.uniform(-0.3, 0.3)  # Radians
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            # Apply rotation: (T, J, 3) @ (3, 3) -> (T, J, 3)
            # Reshape to (T*J, 3) for matmul
            shape_orig = skel_window.shape
            flat_skel = skel_window.reshape(-1, 3)
            flat_skel = flat_skel @ R.T
            skel_window = flat_skel.reshape(shape_orig)

            # Random Scaling
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

        # --- Kinematics Derivation ---
        # Pos: skel_window[1:] (Size W)
        # Prev: skel_window[:-1] (Size W)
        pos = skel_window[1:]
        prev_pos = skel_window[:-1]

        vel = pos - prev_pos

        # Acc: Vel - PrevVel?
        # We need one more frame back for true Acc, or just diff Vel.
        # Diff Vel: vel[t] - vel[t-1].
        # With current window, we can compute Vel for W frames.
        # For Acc, we duplicate first velocity or accept 0 error at t=0.
        # Let's use: acc[t] = vel[t] - vel[t-1]. For t=0, acc=0.
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # --- Log-Modulus Transform ---
        pos = log_modulus_transform(pos)
        vel = log_modulus_transform(vel)
        acc = log_modulus_transform(acc)

        # Concatenate: (W, 20, 9) -> Flatten to (W, 180)
        feat_skel = np.concatenate([pos, vel, acc], axis=2)
        feat_skel = feat_skel.reshape(w_size, -1)

        # --- Audio Window ---
        # Audio is already normalized per sequence. Just slice.
        # We need exactly W frames corresponding to pos.
        audio_window = full_audio[start_frame : start_frame + w_size]

        # Pad audio if short
        if audio_window.shape[0] < w_size:
            pad_len = w_size - audio_window.shape[0]
            pad = np.zeros((pad_len, config.N_MFCC), dtype=np.float32)
            audio_window = np.concatenate([audio_window, pad], axis=0)

        # --- Labels ---
        label_window = full_label[start_frame : start_frame + w_size]
        if label_window.shape[0] < w_size:
            pad_len = w_size - label_window.shape[0]
            # Pad with background (0)
            pad = np.zeros(pad_len, dtype=np.int64)
            label_window = np.concatenate([label_window, pad], axis=0)

        # Convert to tensors
        # Skeleton: (W, 180)
        # Audio: (W, 13)
        # Label: (W,)
        return {
            "skeleton": torch.from_numpy(feat_skel).float(),
            "audio": torch.from_numpy(audio_window).float(),
            "label": torch.from_numpy(label_window).long(),
        }


def get_loaders(batch_size=config.BATCH_SIZE, debug=config.DEBUG):
    """
    Creates DataLoaders for train, val, and test.
    """
    utils.set_seed()

    # Train
    train_ds = GestureDataset(config.TRAIN_METADATA_PATH, is_train=True, debug=debug)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Val
    val_ds = GestureDataset(config.VAL_METADATA_PATH, is_train=False, debug=debug)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Test
    test_ds = GestureDataset(config.TEST_METADATA_PATH, is_train=False, debug=debug)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader
