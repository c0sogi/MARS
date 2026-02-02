import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    GESTURE_MAP,
    UPPER_BODY_JOINTS,
    HYPERPARAMS,
    SEED,
    INPUT_DIM,
)
from library.utils import set_seed


class RobustMATParser:
    """
    Parses MAT files to extract Skeleton data and Label annotations safely.
    """

    @staticmethod
    def parse(mat_path):
        try:
            # Load mat file
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None, None

            video = mat["Video"]

            # Extract Skeleton Data
            # Expected structure: Video.Frames is an array of frame objects
            # Each frame has Skeleton info.
            # We aim to extract WorldPosition for all joints across all frames.

            frames = getattr(video, "Frames", [])
            num_frames = getattr(video, "NumFrames", 0)

            # Handle cases where Frames might be a single object or empty
            if isinstance(frames, np.ndarray) and frames.size > 0:
                pass
            elif not isinstance(frames, np.ndarray) and num_frames > 0:
                # Single frame or weird structure, wrap in list
                frames = [frames]
            else:
                return None, None

            skeleton_data = []

            # Iterate through frames to extract skeleton
            # We assume the first tracked skeleton is the user of interest
            for f in frames:
                try:
                    # Check if Skeleton exists
                    skel = getattr(f, "Skeleton", None)
                    if skel is None:
                        skeleton_data.append(np.zeros((20, 3)))
                        continue

                    # skel might be an array of skeletons (multi-user) or a single struct
                    # We take the first valid one or the first one
                    if isinstance(skel, np.ndarray):
                        if skel.size == 0:
                            curr_skel = None
                        else:
                            curr_skel = skel[0]  # Assume first user
                    else:
                        curr_skel = skel

                    if curr_skel is None:
                        skeleton_data.append(np.zeros((20, 3)))
                        continue

                    # Extract WorldPosition
                    # WorldPosition might be a struct with X, Y, Z
                    wp = getattr(curr_skel, "WorldPosition", None)

                    if wp is None:
                        # Try JointsType iteration if WorldPosition is not direct
                        # But usually WorldPosition is per joint if Skeleton is an array of joints?
                        # Based on description: "Skeleton ... contains JointsType, WorldPosition..."
                        # It implies Skeleton is a struct containing arrays or Skeleton is an array of Joint structs.
                        # Let's assume Skeleton is an array of 20 Joint structures.

                        # Re-evaluating based on description: "Skeleton Frame: An array of Skeleton structures... contained within a Skeletons array"
                        # This implies Video.Frames[i].Skeleton is the array of joints?
                        # Let's try to iterate over the skeleton object as an iterable of joints

                        joints_pos = []
                        # If skel is iterable (array of joints)
                        if isinstance(skel, np.ndarray) and skel.size == 20:
                            for j in skel:
                                p = getattr(j, "WorldPosition", None)
                                if p:
                                    joints_pos.append([p.X, p.Y, p.Z])
                                else:
                                    joints_pos.append([0.0, 0.0, 0.0])
                        else:
                            # Fallback
                            joints_pos = [[0.0, 0.0, 0.0]] * 20

                        skeleton_data.append(np.array(joints_pos))
                    else:
                        # If WorldPosition is directly on Skeleton (unlikely for 20 joints)
                        skeleton_data.append(np.zeros((20, 3)))

                except Exception:
                    skeleton_data.append(np.zeros((20, 3)))

            # Shape: (T, 20, 3)
            skeleton_array = np.array(skeleton_data)

            # Extract Labels
            # Labels structure: Name, Begin, End
            labels_raw = getattr(video, "Labels", [])
            parsed_labels = []

            def process_label(l):
                try:
                    name = l.Name
                    start = int(l.Begin)
                    end = int(l.End)
                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        parsed_labels.append({"id": gid, "start": start, "end": end})
                except:
                    pass

            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_label(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_label(l)
            else:
                process_label(labels_raw)

            return skeleton_array, parsed_labels

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return None, None


class PhysicallyConsistentAugmentor:
    """
    Applies temporally correlated noise to skeleton joints and derives velocity.
    """

    def __init__(self, sigma=0.005, filter_order=2, cutoff_freq=0.1):
        self.sigma = sigma
        self.b, self.a = butter(filter_order, cutoff_freq, btype="low")

    def augment(self, skeleton_pos):
        """
        Args:
            skeleton_pos: (T, J, 3) tensor
        Returns:
            augmented_features: (T, InputDim) tensor including pos, vel, audio
        """
        # skeleton_pos is (T, J, 3)
        T, J, C = skeleton_pos.shape

        # 1. Generate Gaussian Noise
        noise = np.random.normal(0, self.sigma, size=(T, J, C))

        # 2. Apply Temporal Low-Pass Filter
        # Apply along time axis (0)
        smooth_noise = filtfilt(self.b, self.a, noise, axis=0)

        # 3. Add to positions
        aug_pos = skeleton_pos + smooth_noise

        # 4. Derive Velocity
        # v[t] = p[t] - p[t-1], v[0] = 0
        velocity = np.zeros_like(aug_pos)
        velocity[1:] = aug_pos[1:] - aug_pos[:-1]

        return aug_pos, velocity


class GestureDataset(Dataset):
    def __init__(
        self, metadata_path, is_train=True, load_cached_data=True, sample_limit=None
    ):
        self.is_train = is_train
        self.augmentor = PhysicallyConsistentAugmentor() if is_train else None

        # Load metadata
        self.df = pd.read_csv(metadata_path)
        if sample_limit:
            self.df = self.df.iloc[:sample_limit]

        self.sample_ids = self.df["sample_id"].tolist()
        self.data_paths = self.df["data_path"].tolist()
        self.audio_paths = self.df["audio_path"].tolist()

        # Cache setup
        self.cache_file = os.path.join(
            CACHE_DIR,
            f"{'train' if is_train else 'val' if 'val' in metadata_path else 'test'}_data.npz",
        )

        self.data_cache = {}

        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}...")
            try:
                loaded = np.load(self.cache_file, allow_pickle=True)
                # Convert back to dict
                # Keys are stored as array of keys, values as 'arr_0', etc?
                # Or we store as dict in savez.
                # np.savez stores args as keys.
                # We will store a single object array to handle the dict structure or flat keys.
                # Let's use flat keys: "sample_id_skel", "sample_id_audio", "sample_id_labels"

                # Check if it's the dict format we expect
                if "index" in loaded:
                    # Reconstruct dict
                    index = loaded["index"]
                    for idx in index:
                        sid = str(idx)
                        self.data_cache[sid] = {
                            "skeleton": loaded[f"{sid}_skel"],
                            "audio": loaded[f"{sid}_audio"],
                            "labels": (
                                loaded[f"{sid}_labels"]
                                if f"{sid}_labels" in loaded
                                else None
                            ),
                        }
                else:
                    # Fallback or empty
                    pass
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
                self._process_and_cache()
        else:
            self._process_and_cache()

    def _process_and_cache(self):
        print(f"Processing dataset (Is Train: {self.is_train})...")
        processed_data = {}

        for idx, row in self.df.iterrows():
            sid = row["sample_id"]
            mat_rel_path = row["data_path"]
            audio_rel_path = row["audio_path"]

            full_mat_path = os.path.join(INPUT_DIR, mat_rel_path)
            full_audio_path = os.path.join(INPUT_DIR, audio_rel_path)

            # 1. Parse Skeleton & Labels
            skel_raw, labels_meta = RobustMATParser.parse(full_mat_path)

            if skel_raw is None:
                # Fallback for missing data
                skel_raw = np.zeros((100, 20, 3))  # Dummy
                labels_meta = []

            # Select Upper Body Joints
            # skel_raw is (T, 20, 3)
            # Map UPPER_BODY_JOINTS indices
            if skel_raw.shape[1] >= 20:
                skel_body = skel_raw[:, UPPER_BODY_JOINTS, :]
            else:
                skel_body = np.zeros((skel_raw.shape[0], len(UPPER_BODY_JOINTS), 3))

            # Normalize: Center to HipCenter (Index 0 in UPPER_BODY_JOINTS is 0 in raw 20 list)
            # HipCenter is index 0 in raw list.
            # In skel_body, HipCenter is at index 0 (since 0 is in UPPER_BODY_JOINTS).
            hip_pos = skel_body[:, 0:1, :]  # (T, 1, 3)
            skel_norm = (skel_body - hip_pos) * 0.001  # mm to meters

            # 2. Process Audio
            audio_feat = self._extract_audio(
                full_audio_path, target_len=skel_norm.shape[0]
            )

            # 3. Process Labels (Dense)
            T = skel_norm.shape[0]
            if self.is_train and labels_meta:
                # Create frame-wise labels
                cls_target = np.zeros(T, dtype=np.int64)  # 0 is background
                bnd_target = np.zeros(T, dtype=np.float32)

                for l in labels_meta:
                    gid = l["id"]
                    start = max(0, l["start"] - 1)  # 1-based to 0-based
                    end = min(T, l["end"])

                    if start < end:
                        cls_target[start:end] = gid
                        # Boundary: 1 at start and end frames
                        bnd_target[start] = 1.0
                        if end < T:
                            bnd_target[end] = 1.0

                labels_data = {"cls": cls_target, "bnd": bnd_target}
            else:
                labels_data = None

            processed_data[sid] = {
                "skeleton": skel_norm.astype(np.float32),
                "audio": audio_feat.astype(np.float32),
                "labels": labels_data,
            }

        self.data_cache = processed_data

        # Save to cache
        save_dict = {"index": list(processed_data.keys())}
        for k, v in processed_data.items():
            save_dict[f"{k}_skel"] = v["skeleton"]
            save_dict[f"{k}_audio"] = v["audio"]
            if v["labels"] is not None:
                save_dict[f"{k}_labels"] = np.stack(
                    [v["labels"]["cls"], v["labels"]["bnd"]], axis=0
                )

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        np.savez_compressed(self.cache_file, **save_dict)
        print("Data processed and cached.")

    def _extract_audio(self, path, target_len):
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(path)

            # Resample to 16k if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Compute MFCC
            # n_mfcc=13
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=13,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
            mfcc = mfcc.mean(dim=0).transpose(0, 1).numpy()  # (time, n_mfcc)

            # Interpolate to match video frames
            curr_len = mfcc.shape[0]
            if curr_len != target_len:
                # Resize using linear interpolation
                x_old = np.linspace(0, 1, curr_len)
                x_new = np.linspace(0, 1, target_len)
                mfcc_new = np.zeros((target_len, 13))
                for i in range(13):
                    mfcc_new[:, i] = np.interp(x_new, x_old, mfcc[:, i])
                return mfcc_new
            return mfcc

        except Exception:
            return np.zeros((target_len, 13))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        data = self.data_cache.get(sid)

        if data is None:
            # Should not happen if cache logic is correct
            return self.__getitem__((idx + 1) % len(self))

        skel = data["skeleton"]  # (T, 12, 3)
        audio = data["audio"]  # (T, 13)

        # Handle Labels
        if self.is_train and data["labels"] is not None:
            # If loaded from disk, labels might be combined array
            if isinstance(data["labels"], np.ndarray):
                cls_target = data["labels"][0].astype(np.int64)
                bnd_target = data["labels"][1].astype(np.float32)
            else:
                cls_target = data["labels"]["cls"]
                bnd_target = data["labels"]["bnd"]
        else:
            # Dummy targets for test/val if missing
            cls_target = np.zeros(skel.shape[0], dtype=np.int64)
            bnd_target = np.zeros(skel.shape[0], dtype=np.float32)

        # Augmentation
        if self.is_train and self.augmentor:
            skel_aug, vel_aug = self.augmentor.augment(skel)
        else:
            skel_aug = skel
            # Compute velocity without noise
            vel_aug = np.zeros_like(skel)
            vel_aug[1:] = skel[1:] - skel[:-1]

        # Flatten Skeleton Features
        # (T, 12, 3) -> (T, 36)
        T = skel.shape[0]
        skel_flat = skel_aug.reshape(T, -1)
        vel_flat = vel_aug.reshape(T, -1)

        # Concatenate All Features
        # (T, 36+36+13) = (T, 85)
        features = np.concatenate([skel_flat, vel_flat, audio], axis=1)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "cls_target": torch.tensor(cls_target, dtype=torch.long),
            "bnd_target": torch.tensor(bnd_target, dtype=torch.float32),
            "sample_id": sid,
        }


def collate_fn(batch):
    """
    Pads sequences to the maximum length in the batch.
    """
    # Sort by length for packing (optional but good practice)
    batch.sort(key=lambda x: x["features"].shape[0], reverse=True)

    features = [x["features"] for x in batch]
    cls_targets = [x["cls_target"] for x in batch]
    bnd_targets = [x["bnd_target"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    max_len = lengths.max().item()
    input_dim = features[0].shape[1]

    # Pad Features
    padded_features = torch.zeros(len(batch), max_len, input_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)  # 1 for valid, 0 for pad

    # Pad Targets
    padded_cls = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_bnd = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, (feat, cls_t, bnd_t, length) in enumerate(
        zip(features, cls_targets, bnd_targets, lengths)
    ):
        padded_features[i, :length, :] = feat
        mask[i, :length] = 1
        padded_cls[i, :length] = cls_t
        padded_bnd[i, :length] = bnd_t

    return {
        "features": padded_features,
        "mask": mask,
        "cls_target": padded_cls,
        "bnd_target": padded_bnd,
        "lengths": lengths,
        "sample_ids": ids,
    }


def get_dataloaders(batch_size=8, num_workers=2, sample_limit=None):
    train_ds = GestureDataset(
        TRAIN_METADATA_PATH,
        is_train=True,
        load_cached_data=True,
        sample_limit=sample_limit,
    )
    val_ds = GestureDataset(
        VAL_METADATA_PATH,
        is_train=False,
        load_cached_data=True,
        sample_limit=sample_limit,
    )
    test_ds = GestureDataset(
        TEST_METADATA_PATH,
        is_train=False,
        load_cached_data=True,
        sample_limit=sample_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
