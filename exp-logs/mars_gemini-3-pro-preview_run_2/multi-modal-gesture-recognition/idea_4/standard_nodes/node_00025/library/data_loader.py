import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io
from torch.utils.data import Dataset
from library.utils import seed_everything

# Constants
CACHE_DIR = "./working/idea_4/"
INPUT_DIR = "./input"

# Joint Indices for Upper Body (based on standard Kinect v2 mapping and prompt order)
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

GESTURE_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_file,
        root_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        load_cached_data=True,
        is_test=False,
        max_samples=None,
    ):
        """
        Args:
            metadata_file (str): Path to metadata CSV.
            root_dir (str): Root directory of input data.
            cache_dir (str): Directory to store processed cache.
            load_cached_data (bool): Whether to load from cache if available.
            is_test (bool): Whether this is the test set (no labels).
            max_samples (int): Limit number of samples for debugging.
        """
        self.root_dir = root_dir
        self.is_test = is_test
        self.metadata = pd.read_csv(metadata_file)

        # Fix for debug_lesson_1: Sanitize dirty data (NaNs) to prevent os.path.join failure.
        # We fill with empty strings so os.path.join works; downstream file loaders will handle the invalid paths.
        if "data_path" in self.metadata.columns:
            self.metadata["data_path"] = self.metadata["data_path"].fillna("")
        if "audio_path" in self.metadata.columns:
            self.metadata["audio_path"] = self.metadata["audio_path"].fillna("")

        if max_samples is not None:
            self.metadata = self.metadata.iloc[:max_samples]

        self.sample_ids = self.metadata["sample_id"].tolist()

        # Cache setup
        os.makedirs(cache_dir, exist_ok=True)
        dataset_name = os.path.basename(metadata_file).replace(".csv", "")
        self.cache_file = os.path.join(cache_dir, f"{dataset_name}_data.npz")

        self.data = {}

        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}...")
            try:
                # Allow pickle is sometimes needed if numpy detects object arrays,
                # but we try to save pure numeric types.
                loaded = np.load(self.cache_file, allow_pickle=True)
                for key in loaded.files:
                    self.data[key] = loaded[key]
                print("Cache loaded successfully.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
                self._process_and_cache()
        else:
            print(f"Processing data for {dataset_name}...")
            self._process_and_cache()

    def _process_and_cache(self):
        cache_dict = {}

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(self.root_dir, row["data_path"])
            audio_path = os.path.join(self.root_dir, row["audio_path"])

            # Process Skeleton
            skeleton_feats, num_frames = self._process_skeleton(data_path)

            # Process Audio
            audio_feats = self._process_audio(audio_path, num_frames)

            # Fuse Modalities
            if skeleton_feats is not None and audio_feats is not None:
                min_len = min(len(skeleton_feats), len(audio_feats))
                features = np.concatenate(
                    [skeleton_feats[:min_len], audio_feats[:min_len]], axis=1
                )
                final_len = min_len
            elif skeleton_feats is not None:
                features = skeleton_feats
                final_len = len(skeleton_feats)
            else:
                # Fallback for completely failed samples
                features = np.zeros((100, 72 + 13), dtype=np.float32)
                final_len = 100

            # Process Labels
            if self.is_test:
                labels = np.zeros(final_len, dtype=np.int64)
            else:
                labels = self._process_labels(data_path, final_len)

            # Store
            feat_key = f"{sample_id}_features"
            label_key = f"{sample_id}_labels"

            features = features.astype(np.float32)
            labels = labels.astype(np.int64)

            cache_dict[feat_key] = features
            cache_dict[label_key] = labels

            self.data[feat_key] = features
            self.data[label_key] = labels

        # Save to disk
        np.savez(self.cache_file, **cache_dict)
        print(f"Data cached to {self.cache_file}")

    def _process_skeleton(self, mat_path):
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return None, 0

            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)
            frames = getattr(video, "Frames", [])

            if num_frames == 0 or len(frames) == 0:
                return None, 0

            skeleton_data = []

            for i in range(num_frames):
                try:
                    frame_obj = frames[i]
                    skel = getattr(frame_obj, "Skeleton", None)

                    joints = None
                    if skel is not None:
                        # Handle array of skeletons (multi-user) -> take first
                        if isinstance(skel, np.ndarray) and len(skel) > 0:
                            skel = skel[0]
                        elif isinstance(skel, np.ndarray) and len(skel) == 0:
                            skel = None

                        if skel is not None and hasattr(skel, "WorldPosition"):
                            wp = skel.WorldPosition
                            curr_joints = []

                            # Extract 20 joints
                            # Case 1: WorldPosition is array of structs/objects
                            if isinstance(wp, np.ndarray) and len(wp) >= 20:
                                for j_idx in range(20):
                                    pos = wp[j_idx]
                                    curr_joints.append([pos.X, pos.Y, pos.Z])
                            # Case 2: WorldPosition has X, Y, Z arrays
                            elif hasattr(wp, "X") and isinstance(wp.X, np.ndarray):
                                for j_idx in range(20):
                                    curr_joints.append(
                                        [wp.X[j_idx], wp.Y[j_idx], wp.Z[j_idx]]
                                    )
                            else:
                                curr_joints = [[0.0, 0.0, 0.0]] * 20

                            joints = np.array(curr_joints)  # (20, 3)

                    if joints is None:
                        joints = np.zeros((20, 3))

                    skeleton_data.append(joints)
                except Exception:
                    skeleton_data.append(np.zeros((20, 3)))

            skeleton_data = np.array(skeleton_data)  # (T, 20, 3)

            # Feature Engineering
            # 1. Select Upper Body (12 joints)
            upper_body = skeleton_data[:, UPPER_BODY_INDICES, :]  # (T, 12, 3)

            # 2. Normalize relative to HipCenter (Index 0 in upper_body list)
            hip_center = upper_body[:, 0:1, :]
            normalized = upper_body - hip_center  # (T, 12, 3)

            # Flatten spatial dims
            normalized_flat = normalized.reshape(num_frames, -1)  # (T, 36)

            # 3. Compute Velocity (Temporal Derivative)
            velocity = np.zeros_like(normalized_flat)
            velocity[1:] = normalized_flat[1:] - normalized_flat[:-1]

            # Concatenate
            features = np.concatenate([normalized_flat, velocity], axis=1)  # (T, 72)

            return features, num_frames

        except Exception as e:
            # print(f"Error processing skeleton {mat_path}: {e}")
            return None, 0

    def _process_audio(self, audio_path, target_frames):
        try:
            if not os.path.exists(audio_path) or target_frames == 0:
                return np.zeros((target_frames, 13))

            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (channel, n_mfcc, time)

            # Average channels
            mfcc = mfcc.mean(dim=0)  # (n_mfcc, time)

            # Interpolate to match video frames
            mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).permute(1, 0).numpy()  # (target_frames, n_mfcc)

            return mfcc
        except Exception:
            return np.zeros((target_frames, 13))

    def _process_labels(self, mat_path, num_frames):
        labels = np.zeros(num_frames, dtype=np.int64)
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]

            if not hasattr(video, "Labels"):
                return labels

            raw_labels = video.Labels

            def extract_label(obj):
                try:
                    name = obj.Name
                    # Matlab 1-based to Python 0-based
                    start = int(obj.Begin) - 1
                    end = int(obj.End)

                    start = max(0, start)
                    end = min(num_frames, end)

                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        labels[start:end] = gid
                except:
                    pass

            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0:
                    extract_label(raw_labels.item())
                else:
                    for l in raw_labels:
                        extract_label(l)
            else:
                extract_label(raw_labels)

            return labels
        except Exception:
            return labels

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        features = self.data.get(f"{sample_id}_features")
        labels = self.data.get(f"{sample_id}_labels")

        # Convert to torch tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        return features, labels, len(features)


def collate_fn(batch):
    """
    Collate function for padding variable length sequences.
    """
    # batch is list of (features, labels, length)
    features, labels, lengths = zip(*batch)

    # Pad features (batch, max_len, dim)
    padded_features = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )

    # Pad labels (batch, max_len)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    # Create mask (batch, max_len)
    batch_size = len(lengths)
    max_len = padded_features.size(1)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    return padded_features, padded_labels, torch.tensor(lengths), mask
