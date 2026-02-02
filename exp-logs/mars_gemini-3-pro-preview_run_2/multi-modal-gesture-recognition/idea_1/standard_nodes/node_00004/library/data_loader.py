import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class GestureDataset(Dataset):
    def __init__(self, metadata_file, is_train=True, load_cached_data=True, limit=0):
        self.is_train = is_train
        self.metadata = pd.read_csv(metadata_file)

        # Limit dataset for debugging
        if limit > 0:
            self.metadata = self.metadata.iloc[:limit]

        # Determine cache file path
        if is_train:
            if "val.csv" in metadata_file:
                self.cache_file = Config.CACHE_FILE_VAL
            else:
                self.cache_file = Config.CACHE_FILE_TRAIN
        else:
            self.cache_file = Config.CACHE_FILE_TEST

        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        # Check if cache exists and load
        if load_cached_data and os.path.exists(self.cache_file):
            try:
                print(f"Loading cached data from {self.cache_file}...")
                data = np.load(self.cache_file, allow_pickle=True)
                # Filter None values from cache (Cite debug_lesson_2)
                return [x for x in data.tolist() if x is not None]
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process data
        print(f"Processing {len(self.metadata)} samples from {self.metadata.shape}...")
        processed_data = []

        for idx, row in self.metadata.iterrows():
            # We must return a sample for every row, especially for test set
            sample = self._process_single_item(row)
            # Filter None values (Cite debug_lesson_2)
            if sample is not None:
                processed_data.append(sample)

        # Save cache
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            np.save(self.cache_file, np.array(processed_data, dtype=object))
            print(f"Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return processed_data

    def _process_single_item(self, row):
        # Validate paths are strings to prevent TypeError in os.path.join (Cite debug_lesson_1)
        if not isinstance(row["data_path"], str) or not isinstance(
            row["audio_path"], str
        ):
            return None

        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Load MAT file (Skeleton & Labels)
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except:
            return None

        if "Video" not in mat:
            return None

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        if num_frames == 0:
            return None

        # 2. Extract Skeleton Features
        # Shape: (NumFrames, 20, 3)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Handle Frames structure (can be scalar, array, or list)
        if not isinstance(frames, np.ndarray) and num_frames == 1:
            frames = [frames]
        elif isinstance(frames, np.ndarray) and frames.ndim == 0:
            frames = [frames.item()]

        for i in range(min(num_frames, len(frames))):
            frame_obj = frames[i]
            if not hasattr(frame_obj, "Skeleton"):
                continue

            skel = frame_obj.Skeleton
            # Handle multiple users (pick first)
            if isinstance(skel, np.ndarray):
                if skel.size > 0:
                    curr_skel = skel[0]
                else:
                    continue
            else:
                curr_skel = skel

            if curr_skel is None:
                continue

            # Extract joints
            # Heuristic: Check if curr_skel is array of joints or has 'Joint' field
            joints_source = None
            if isinstance(curr_skel, np.ndarray) and len(curr_skel) == 20:
                joints_source = curr_skel
            elif hasattr(curr_skel, "Joint"):
                joints_source = curr_skel.Joint

            if joints_source is not None:
                for j_idx in range(min(20, len(joints_source))):
                    joint = joints_source[j_idx]
                    if hasattr(joint, "WorldPosition"):
                        wp = joint.WorldPosition
                        if hasattr(wp, "X"):
                            skeleton_data[i, j_idx, 0] = wp.X
                            skeleton_data[i, j_idx, 1] = wp.Y
                            skeleton_data[i, j_idx, 2] = wp.Z
                        elif isinstance(wp, np.ndarray) and wp.size >= 3:
                            skeleton_data[i, j_idx, :] = wp[:3]

        # Normalize relative to HipCenter (Index 0)
        hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_center

        # Select specific upper-body joints
        selected_skel = skeleton_data[:, Config.SELECTED_JOINT_INDICES, :]  # (T, 12, 3)

        # Flatten joints: (T, 36)
        skel_features = selected_skel.reshape(num_frames, -1)

        # Compute Velocity: (T, 36)
        velocity = np.zeros_like(skel_features)
        velocity[1:] = skel_features[1:] - skel_features[:-1]

        # 3. Audio Features
        audio_features = np.zeros((num_frames, Config.N_MFCC), dtype=np.float32)
        if os.path.exists(audio_path):
            try:
                waveform, sample_rate = torchaudio.load(audio_path)

                # Resample
                if sample_rate != Config.AUDIO_SR:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=Config.AUDIO_SR
                    )
                    waveform = resampler(waveform)

                # Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # MFCC extraction
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=Config.AUDIO_SR,
                    n_mfcc=Config.N_MFCC,
                    melkwargs={
                        "n_fft": 2048,
                        "hop_length": Config.HOP_LENGTH,
                        "n_mels": 64,
                    },
                )
                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

                # Interpolate to match video frames
                # Input to interpolate must be (Batch, Channels, Time) -> (1, 13, Time)
                if mfcc.dim() == 2:
                    mfcc = mfcc.unsqueeze(0)
                elif mfcc.dim() == 3 and mfcc.shape[0] > 1:
                    # If multiple channels, take mean or slice, but we ensured mono
                    pass

                mfcc_aligned = F.interpolate(
                    mfcc, size=num_frames, mode="linear", align_corners=False
                )

                # (1, 13, NumFrames) -> (NumFrames, 13)
                audio_features = mfcc_aligned.squeeze(0).permute(1, 0).numpy()

            except Exception:
                pass

        # 4. Concatenate Features
        final_features = np.concatenate(
            [skel_features, velocity, audio_features], axis=1
        )

        # 5. Labels
        labels = np.zeros(num_frames, dtype=np.int64)

        if self.is_train:
            raw_labels = getattr(video, "Labels", [])

            def process_label_entry(l):
                try:
                    name = l.Name
                    start = int(l.Begin) - 1  # 1-based to 0-based
                    end = int(l.End)  # inclusive
                    if name in Config.GESTURE_MAP:
                        gid = Config.GESTURE_MAP[name]
                        start = max(0, start)
                        end = min(num_frames, end)
                        labels[start:end] = gid
                except:
                    pass

            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0:
                    process_label_entry(raw_labels.item())
                else:
                    for l in raw_labels:
                        process_label_entry(l)
            else:
                process_label_entry(raw_labels)

        return {
            "features": final_features.astype(np.float32),
            "labels": labels,
            "sample_id": row["sample_id"],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "sample_id": sample["sample_id"],
        }


def pad_collate(batch):
    # Sort by length for potential packing
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)

    # Pad sequences
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        "features": padded_features,
        "labels": padded_labels,
        "lengths": lengths,
        "sample_ids": ids,
    }


def get_dataloaders(debug=False):
    limit = Config.DEBUG_SUBSET_SIZE if debug else 0

    train_ds = GestureDataset(Config.TRAIN_METADATA_PATH, is_train=True, limit=limit)
    val_ds = GestureDataset(Config.VAL_METADATA_PATH, is_train=True, limit=limit)
    test_ds = GestureDataset(Config.TEST_METADATA_PATH, is_train=False, limit=limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
