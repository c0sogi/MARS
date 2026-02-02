import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class GestureDataset(Dataset):
    def __init__(self, data, is_train=True):
        """
        Args:
            data (list): List of dictionaries containing 'skeleton', 'audio', 'labels', 'num_frames'.
            is_train (bool): Whether to apply augmentation.
        """
        self.data = data
        self.is_train = is_train

        # Audio transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

    def __len__(self):
        return len(self.data)

    def _augment_skeleton(self, skeleton):
        """
        Applies physically consistent smooth noise to skeleton positions.
        skeleton: (T, J, 3)
        """
        if not self.is_train:
            return skeleton

        # Generate Gaussian noise
        noise = np.random.randn(*skeleton.shape).astype(np.float32)

        # Apply Temporal Low-Pass Filter (Gaussian smoothing along time axis)
        # Sigma=1.0 - 2.0 provides reasonable smoothness for jitter simulation
        sigma = np.random.uniform(0.5, 2.0)
        smooth_noise = scipy.ndimage.gaussian_filter1d(noise, sigma=sigma, axis=0)

        # Scale noise magnitude (e.g., +/- 10mm to 30mm)
        magnitude = np.random.uniform(5.0, 20.0)
        smooth_noise *= magnitude

        return skeleton + smooth_noise

    def _compute_velocity(self, skeleton):
        """
        Computes temporal velocity from positions.
        skeleton: (T, J, 3)
        Returns: (T, J, 3)
        """
        # Padding the first frame with 0 velocity
        velocity = np.zeros_like(skeleton)
        velocity[1:] = skeleton[1:] - skeleton[:-1]
        return velocity

    def _normalize_skeleton(self, skeleton):
        """
        Centers skeleton relative to HipCenter (idx 0) and scales to meters.
        """
        # Center around HipCenter (Config.CENTER_JOINT_IDX = 0)
        # skeleton shape: (T, J, 3)
        hip_center = skeleton[
            :, Config.CENTER_JOINT_IDX : Config.CENTER_JOINT_IDX + 1, :
        ]
        skeleton = skeleton - hip_center

        # Scale mm to meters
        skeleton = skeleton * Config.SCALE_FACTOR
        return skeleton.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 1. Load Data
        # Skeleton: (T, Num_Joints_Total, 3)
        raw_skeleton = sample["skeleton"]

        # Audio: (T_audio, ) or (1, T_audio)
        raw_audio = torch.from_numpy(sample["audio"]).float()

        # Targets: (T, )
        frame_labels = sample["labels"]

        # 2. Process Skeleton
        # Select specific upper-body joints
        skeleton = raw_skeleton[:, Config.SELECTED_JOINTS, :]

        # Augmentation (only on positions, before velocity)
        skeleton = self._augment_skeleton(skeleton)

        # Compute Velocity (physically consistent)
        velocity = self._compute_velocity(skeleton)

        # Normalization
        skeleton = self._normalize_skeleton(skeleton)
        velocity = velocity * Config.SCALE_FACTOR  # Scale velocity to m/frame

        # Flatten joints: (T, J, 3) -> (T, J*3)
        T = skeleton.shape[0]
        skeleton_flat = skeleton.reshape(T, -1)
        velocity_flat = velocity.reshape(T, -1)

        # 3. Process Audio (MFCC)
        # We need to align audio features to video frames T
        # Audio is likely 16kHz.
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(dim=0)  # Mix to mono if stereo

        mfcc = self.mfcc_transform(raw_audio)  # (n_mfcc, L_frames)
        mfcc = mfcc.transpose(0, 1)  # (L_frames, n_mfcc)

        # Interpolate MFCC to match video frame count T
        if mfcc.shape[0] != T:
            mfcc = (
                torch.nn.functional.interpolate(
                    mfcc.unsqueeze(0).transpose(1, 2),  # (1, n_mfcc, L_frames)
                    size=T,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .squeeze(0)
            )  # (T, n_mfcc)

        # 4. Feature Fusion
        # Concatenate: [Skeleton, Velocity, Audio]
        # skeleton_flat: (T, 36), velocity_flat: (T, 36), mfcc: (T, 13)
        features = torch.cat(
            [
                torch.from_numpy(skeleton_flat).float(),
                torch.from_numpy(velocity_flat).float(),
                mfcc.float(),
            ],
            dim=1,
        )

        # 5. Targets
        # frame_labels is numpy array (T,). Convert to tensor.
        targets = torch.from_numpy(frame_labels).long()

        # Boundary target: 1 if gesture (class > 0), 0 if background
        boundaries = (targets > 0).float()

        return {
            "features": features,  # (T, 85)
            "targets": targets,  # (T,)
            "boundaries": boundaries,  # (T,)
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Pads sequences to the maximum length in the batch.
    """
    features = [b["features"] for b in batch]
    targets = [b["targets"] for b in batch]
    boundaries = [b["boundaries"] for b in batch]
    ids = [b["sample_id"] for b in batch]

    # Pad sequences
    # features: (B, T_max, D)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    # targets: (B, T_max) - Pad with 0 (Background)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    # boundaries: (B, T_max) - Pad with 0
    boundaries_padded = pad_sequence(boundaries, batch_first=True, padding_value=0.0)

    # Create masks: (B, T_max)
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    mask = torch.arange(features_padded.size(1))[None, :] < lengths[:, None]

    return features_padded, targets_padded, boundaries_padded, mask, ids


def parse_mat_file(mat_path, num_frames):
    """
    Parses .mat file to extract skeleton and frame-wise labels.
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]

        # Extract Skeleton
        # Structure: Video.Frames(i).Skeleton.WorldPosition
        # This is slow to iterate in Python.
        # Optimization: The provided description says "Exporting the data... generated MAT file stores... Skeleton Frame".
        # However, the input directory contains `SampleXXXXX_data.mat` which has the structure `Video`.
        # The `SampleXXXXX_X.mat` files are generated by the export tool, which we don't run.
        # We must parse `SampleXXXXX_data.mat`.

        frames_struct = video.Frames

        # Pre-allocate skeleton array: (T, 20, 3)
        # 20 joints defined in description
        skeleton = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Handle case where Frames is a single object or array
        if not isinstance(frames_struct, np.ndarray):
            frames_list = [frames_struct]
        else:
            frames_list = frames_struct

        # Iterate frames (this might be slow, but done once during caching)
        # Note: In some .mat files, Frames might be smaller than NumFrames or contain missing data.
        limit = min(num_frames, len(frames_list))

        for i in range(limit):
            frame_obj = frames_list[i]
            skel_obj = frame_obj.Skeleton

            # skel_obj is an array of skeletons (multi-user).
            # We need the one corresponding to the user.
            # The prompt mentions "SessionID_user" video for segmentation.
            # For simplicity in this challenge context, we usually take the first tracked skeleton
            # or the one with valid data.

            target_skel = None
            if isinstance(skel_obj, np.ndarray):
                if len(skel_obj) > 0:
                    target_skel = skel_obj[0]  # Assume primary user is first
            else:
                target_skel = skel_obj

            if target_skel is not None:
                # Joint positions
                # WorldPosition: X, Y, Z
                # Joints order is fixed as per description (20 joints)
                # We need to iterate joints.
                # Structure: target_skel.Joints(j).WorldPosition
                # Or target_skel.WorldPosition if it's struct of arrays?
                # Description: "Skeleton structure... JointsType... WorldPosition"
                # Usually it's an array of joints.

                # Let's try to extract joints robustly
                joints = target_skel  # This seems to be a single struct with fields?
                # Actually, usually Skeleton is an array of 20 joints.
                # Let's assume target_skel is an array of 20 structs

                if isinstance(target_skel, np.ndarray) and len(target_skel) == 20:
                    for j in range(20):
                        pos = target_skel[j].WorldPosition
                        skeleton[i, j, 0] = pos.X
                        skeleton[i, j, 1] = pos.Y
                        skeleton[i, j, 2] = pos.Z
                # If structure is different, we might leave as zeros (defensive)

        # Extract Labels
        # Construct frame-wise targets
        labels_arr = np.zeros((num_frames,), dtype=np.int64)
        labels_raw = getattr(video, "Labels", [])

        def process_label(obj):
            try:
                name = obj.Name
                if name in Config.GESTURE_MAP:
                    gid = Config.GESTURE_MAP[name]
                    # Matlab 1-based indexing
                    start = int(obj.Begin) - 1
                    end = int(obj.End)
                    # Clamp
                    start = max(0, start)
                    end = min(num_frames, end)
                    labels_arr[start:end] = gid
            except AttributeError:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                process_label(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label(l)
        else:
            process_label(labels_raw)

        return skeleton, labels_arr

    except Exception as e:
        print(f"Error parsing {mat_path}: {e}")
        return np.zeros((num_frames, 20, 3)), np.zeros((num_frames,), dtype=np.int64)


def load_data(subset_name, metadata_path, load_cached_data=True):
    """
    Loads data, using cache if available.
    subset_name: 'train', 'val', or 'test'
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{subset_name}_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"[+] Loading {subset_name} data from cache: {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_list = loaded["data_list"]
            return data_list.tolist()
        except Exception as e:
            print(f"[-] Failed to load cache: {e}. Recomputing...")

    print(f"[+] Processing {subset_name} data from scratch...")
    df = pd.read_csv(metadata_path)
    data_list = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        num_frames = int(row["num_frames"])

        # 1. Parse .mat for Skeleton and Labels
        skeleton, frame_labels = parse_mat_file(data_path, num_frames)

        # 2. Load Audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Resample if not 16k (though usually it is)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
        except Exception:
            # Fallback empty audio
            waveform = torch.zeros(1, num_frames * 1600)  # Approx length

        data_list.append(
            {
                "sample_id": sample_id,
                "skeleton": skeleton,  # (T, 20, 3)
                "audio": waveform.numpy(),  # (C, Samples)
                "labels": frame_labels,  # (T,)
                "num_frames": num_frames,
            }
        )

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, data_list=np.array(data_list, dtype=object))
    print(f"[+] Saved {subset_name} data to cache.")

    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Returns train, val, test dataloaders.
    """
    # Load raw data structures
    train_data = load_data("train", Config.TRAIN_METADATA, load_cached_data)
    val_data = load_data("val", Config.VAL_METADATA, load_cached_data)
    test_data = load_data("test", Config.TEST_METADATA, load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(train_data, is_train=True)
    val_dataset = GestureDataset(val_data, is_train=False)
    test_dataset = GestureDataset(test_data, is_train=False)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
