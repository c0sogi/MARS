import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import library.config as config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, metadata_path, is_train=True, load_cached=True, cache_path=None):
        """
        Dataset class for Multi-modal Gesture Recognition.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            is_train (bool): Flag to enable augmentation and label loading.
            load_cached (bool): Whether to attempt loading from cache.
            cache_path (str): Path to the .npz cache file.
        """
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_path)
        self.cache_path = cache_path

        # Load data (either from cache or process from scratch)
        self.data_x, self.data_y, self.offsets, self.sample_ids = self._load_data(
            load_cached
        )

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes it from raw files.
        """
        # 1. Try Loading Cache
        if load_cached and self.cache_path and os.path.exists(self.cache_path):
            try:
                # Allow pickle is set to False to ensure we strictly use npy format
                # We use byte strings for sample_ids to avoid pickle
                data = np.load(self.cache_path, allow_pickle=False)

                data_x = data["data_x"]
                data_y = data["data_y"]
                offsets = data["offsets"]
                # Decode byte strings back to normal strings
                sample_ids = [s.decode("utf-8") for s in data["sample_ids"]]

                print(f"Successfully loaded cached data from {self.cache_path}")
                print(f"Total samples: {len(sample_ids)}, Total frames: {len(data_x)}")
                return data_x, data_y, offsets, sample_ids
            except Exception as e:
                print(f"Cache loading failed: {e}. Reprocessing from scratch...")

        # 2. Process from Scratch
        print("Processing raw data...")
        all_features = []
        all_targets = []
        offsets = []
        sample_ids = []
        current_offset = 0

        # Ensure cache directory exists
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        for idx, row in self.metadata.iterrows():
            sample_id = str(row["sample_id"])
            # Construct full paths
            data_path = os.path.join(config.INPUT_DIR, str(row["data_path"]))
            audio_path = os.path.join(config.INPUT_DIR, str(row["audio_path"]))

            try:
                # --- Load Skeleton Data ---
                # Load .mat file
                mat = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                if "Video" not in mat:
                    continue
                video = mat["Video"]
                num_frames = int(video.NumFrames)

                # Extract Frames
                frames = video.Frames
                if not isinstance(frames, np.ndarray):
                    frames = np.array([frames])

                # Pre-allocate skeleton array: (T, NumJoints, 3)
                # Cite solution_lesson_node_00020: Feature Relevance Outweighs Architectural Complexity
                skeleton_raw = np.zeros(
                    (num_frames, config.NUM_JOINTS, 3), dtype=np.float32
                )

                # Iterate frames to extract skeleton
                for f_idx in range(min(num_frames, len(frames))):
                    try:
                        frame_obj = frames[f_idx]
                        skel = frame_obj.Skeleton
                        if isinstance(skel, np.ndarray) and len(skel) > 0:
                            skel = skel[0]

                        if hasattr(skel, "WorldPosition"):
                            w_pos = skel.WorldPosition
                            # Check if w_pos has enough joints (original data has 20)
                            if isinstance(w_pos, np.ndarray) and len(w_pos) >= 20:
                                for i, joint_idx in enumerate(config.SELECTED_JOINTS):
                                    joint = w_pos[joint_idx]
                                    if hasattr(joint, "X"):
                                        skeleton_raw[f_idx, i, 0] = float(joint.X)
                                        skeleton_raw[f_idx, i, 1] = float(joint.Y)
                                        skeleton_raw[f_idx, i, 2] = float(joint.Z)
                    except Exception:
                        pass

                # --- Normalization ---
                # Subtract HipCenter from all joints
                # HipCenter is index 0 in original, which is index 0 in SELECTED_JOINTS
                hip_center = skeleton_raw[:, 0:1, :]  # (T, 1, 3)
                skeleton_norm = skeleton_raw - hip_center

                # Flatten: (T, 60)
                skeleton_flat = skeleton_norm.reshape(num_frames, -1)

                # --- Velocity ---
                # First derivative: (T, 60)
                velocity = np.zeros_like(skeleton_flat)
                velocity[1:] = skeleton_flat[1:] - skeleton_flat[:-1]

                # --- Audio Processing ---
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Resample if necessary
                if sample_rate != config.AUDIO_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=config.AUDIO_SAMPLE_RATE
                    )
                    waveform = resampler(waveform)

                # Extract MFCC
                # n_fft matches ~25ms window, hop_length matches ~10fps alignment
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=config.AUDIO_SAMPLE_RATE,
                    n_mfcc=config.AUDIO_DIM,
                    melkwargs={
                        "n_fft": 400,
                        "hop_length": config.AUDIO_HOP_LENGTH,
                        "n_mels": 23,
                        "center": False,
                    },
                )
                mfcc = mfcc_transform(waveform)  # Shape: (Channels, n_mfcc, Time)

                # Remove channel dim (mono) -> (n_mfcc, Time)
                if mfcc.dim() == 3:
                    mfcc = mfcc.mean(dim=0)

                # Align Audio to Video Frames via Interpolation
                # Input to interpolate: (Batch, Channels, Time) -> (1, n_mfcc, Time)
                mfcc_in = mfcc.unsqueeze(0)

                # Interpolate to match num_frames
                mfcc_aligned = F.interpolate(
                    mfcc_in, size=num_frames, mode="linear", align_corners=False
                )

                # Reshape to (T, n_mfcc)
                audio_features = mfcc_aligned.squeeze(0).transpose(0, 1).numpy()

                # --- Feature Concatenation ---
                # (T, 60 + 60 + 13) = (T, 133)
                features = np.concatenate(
                    [skeleton_flat, velocity, audio_features], axis=1
                )

                # --- Label Generation ---
                targets = np.zeros(num_frames, dtype=np.int32)

                # Only process labels if training/val and Labels exist
                if hasattr(video, "Labels"):
                    labels_raw = video.Labels

                    # Helper to apply label
                    def apply_label(l_obj):
                        try:
                            name = l_obj.Name
                            if name in config.GESTURE_MAP:
                                gid = config.GESTURE_MAP[name]
                                # Matlab 1-based indexing
                                start_f = int(l_obj.Begin) - 1
                                end_f = int(l_obj.End)
                                # Clip
                                start_f = max(0, start_f)
                                end_f = min(num_frames, end_f)
                                if end_f > start_f:
                                    targets[start_f:end_f] = gid
                        except AttributeError:
                            pass

                    # Handle array vs single object
                    if isinstance(labels_raw, np.ndarray):
                        if labels_raw.ndim == 0:
                            apply_label(labels_raw.item())
                        else:
                            for l in labels_raw:
                                apply_label(l)
                    else:
                        apply_label(labels_raw)

                # Store processed data
                all_features.append(features.astype(np.float32))
                all_targets.append(targets.astype(np.int32))
                sample_ids.append(sample_id)

                # Update offsets
                length = num_frames
                offsets.append([current_offset, current_offset + length])
                current_offset += length

            except Exception as e:
                # Skip sample on error
                # print(f"Error processing {sample_id}: {e}")
                continue

        if not all_features:
            raise RuntimeError("No data could be processed from the input files.")

        # Concatenate all data into contiguous arrays
        data_x = np.concatenate(all_features, axis=0)
        data_y = np.concatenate(all_targets, axis=0)
        offsets = np.array(offsets, dtype=np.int32)

        # Save to cache if path provided
        if self.cache_path:
            # Convert sample_ids to fixed-length byte strings for pickle-free saving
            sample_ids_bytes = np.array(sample_ids, dtype="S")
            np.savez(
                self.cache_path,
                data_x=data_x,
                data_y=data_y,
                offsets=offsets,
                sample_ids=sample_ids_bytes,
            )
            print(f"Saved processed data to cache: {self.cache_path}")

        return data_x, data_y, offsets, sample_ids

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        # Retrieve start and end indices from offsets
        start, end = self.offsets[idx]

        # Slice the contiguous arrays
        x = self.data_x[start:end]
        y = self.data_y[start:end]

        # Convert to Tensor
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y).long()

        # Apply Augmentation (Training Only)
        if self.is_train:
            # Add Gaussian noise to features
            noise = torch.randn_like(x_tensor) * config.NOISE_STD
            x_tensor = x_tensor + noise

        return x_tensor, y_tensor, self.sample_ids[idx]


def collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    Pads sequences to the length of the longest sequence in the batch.
    """
    # Unpack batch
    xs, ys, ids = zip(*batch)

    # Get original lengths
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> Output shape: (Batch, MaxTime, Feat)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = pad_sequence(
        ys, batch_first=True, padding_value=0
    )  # 0 is background class

    return xs_padded, ys_padded, lengths, ids


def get_dataloaders():
    """
    Factory function to create Train, Val, and Test dataloaders.
    """
    # Initialize Datasets
    train_ds = GestureDataset(
        config.TRAIN_METADATA_PATH,
        is_train=True,
        load_cached=True,
        cache_path=config.TRAIN_CACHE_PATH,
    )

    val_ds = GestureDataset(
        config.VAL_METADATA_PATH,
        is_train=False,
        load_cached=True,
        cache_path=config.VAL_CACHE_PATH,
    )

    test_ds = GestureDataset(
        config.TEST_METADATA_PATH,
        is_train=False,
        load_cached=True,
        cache_path=config.TEST_CACHE_PATH,
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
