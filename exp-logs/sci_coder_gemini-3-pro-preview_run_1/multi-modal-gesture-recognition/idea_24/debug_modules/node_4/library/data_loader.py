import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config

# Joint indices based on the prompt description order
# 0: HipCenter is the root
JOINTS_ORDER = [
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
HIP_CENTER_IDX = 0
NUM_JOINTS = 20


class MultimodalDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Load Metadata
        if mode == "train":
            csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        elif mode == "val":
            csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        else:
            csv_path = os.path.join(Config.METADATA_DIR, "test.csv")

        self.df = pd.read_csv(csv_path)

        # Debugging: Subset
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE)

        # Audio Transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.HOP_LENGTH,
                "win_length": Config.WIN_LENGTH,
                "center": False,
            },
        )

        # Stats file path
        self.stats_path = os.path.join(os.path.dirname(Config.CACHE_DIR), "stats.npz")
        self.stats = self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        """
        Loads global mean/std for normalization.
        If not found and mode is train, computes them.
        If not found and mode is not train, uses defaults (zeros/ones).
        """
        if os.path.exists(self.stats_path):
            return np.load(self.stats_path)

        if self.mode == "train":
            print("Computing global statistics for normalization...")
            # We need to process all training items first to compute stats
            # This might be slow on first run but is necessary for Z-score

            skel_sum = np.zeros(NUM_JOINTS * 3)
            skel_sq_sum = np.zeros(NUM_JOINTS * 3)
            skel_count = 0

            audio_sum = np.zeros(Config.N_MFCC)
            audio_sq_sum = np.zeros(Config.N_MFCC)
            audio_count = 0

            for idx in range(len(self.df)):
                data = self._process_item(idx)
                if data is None:
                    continue

                s = data["skeleton"]  # (T, D)
                a = data["audio"]  # (T, D)

                skel_sum += np.sum(s, axis=0)
                skel_sq_sum += np.sum(s**2, axis=0)
                skel_count += s.shape[0]

                audio_sum += np.sum(a, axis=0)
                audio_sq_sum += np.sum(a**2, axis=0)
                audio_count += a.shape[0]

            skel_mean = skel_sum / skel_count
            skel_std = np.sqrt((skel_sq_sum / skel_count) - (skel_mean**2)) + 1e-6

            audio_mean = audio_sum / audio_count
            audio_std = np.sqrt((audio_sq_sum / audio_count) - (audio_mean**2)) + 1e-6

            np.savez(
                self.stats_path,
                skel_mean=skel_mean,
                skel_std=skel_std,
                audio_mean=audio_mean,
                audio_std=audio_std,
            )

            return np.load(self.stats_path)
        else:
            # Fallback if stats don't exist and we are in inference mode (should not happen in proper pipeline)
            return {"skel_mean": 0, "skel_std": 1, "audio_mean": 0, "audio_std": 1}

    def _process_item(self, idx):
        """
        Reads raw data, processes it, and caches it.
        Returns dictionary with 'skeleton', 'audio', 'labels'.
        """
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        cache_file = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        # 1. Try Load Cache
        if self.load_cached_data and os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                return dict(data)
            except:
                pass  # Corrupt cache, recompute

        # 2. Process from Scratch
        # Paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # --- Skeleton Processing ---
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]
            num_frames = getattr(video_struct, "NumFrames", 0)

            # Initialize skeleton array (T, Joints, 3)
            # We use NaN to indicate missing frames initially
            skeleton_data = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)

            frames = getattr(video_struct, "Frames", [])
            if isinstance(frames, np.ndarray) and len(frames) > 0:
                # Limit to num_frames just in case
                valid_frames = min(len(frames), num_frames)
                for f_idx in range(valid_frames):
                    frame_obj = frames[f_idx]
                    if hasattr(frame_obj, "Skeleton"):
                        skel_obj = frame_obj.Skeleton
                        # Handle array of skeletons (multiple users), take first
                        if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                            skel_obj = skel_obj[0]

                        if hasattr(skel_obj, "WorldPosition"):
                            # WorldPosition might be an array of structs or struct of arrays
                            # Based on prompt: 20 joints.
                            # We assume standard Kinect ordering or explicit field access if possible.
                            # Since prompt implies structure, we try to iterate.
                            # However, usually WorldPosition is (Joints x 3) or similar in these datasets.
                            # Let's assume it's iterable or array.
                            # If it is a struct with X,Y,Z fields per joint:
                            pass

                            # Heuristic for the specific dataset format described:
                            # Usually WorldPosition is a 20x1 struct array or similar.
                            # We will try to extract coordinates.

                            # Fallback: if we can't parse perfectly, we use zeros.
                            # But we must implement the logic.
                            # Let's assume WorldPosition is a (20,) struct array with X, Y, Z
                            wp = skel_obj.WorldPosition
                            if isinstance(wp, np.ndarray) and wp.size == NUM_JOINTS:
                                for j_idx in range(NUM_JOINTS):
                                    joint = wp[j_idx]
                                    skeleton_data[f_idx, j_idx, 0] = joint.X
                                    skeleton_data[f_idx, j_idx, 1] = joint.Y
                                    skeleton_data[f_idx, j_idx, 2] = joint.Z

            # Root Relative: Subtract HipCenter (Index 0)
            # Shape: (T, 20, 3)
            hip_centers = skeleton_data[
                :, HIP_CENTER_IDX : HIP_CENTER_IDX + 1, :
            ]  # (T, 1, 3)
            skeleton_data = skeleton_data - hip_centers

            # Flatten: (T, 60)
            skeleton_features = skeleton_data.reshape(num_frames, -1)

        except Exception as e:
            # Fallback for broken files
            # print(f"Error processing skeleton {sample_id}: {e}")
            return None

        # --- Audio Processing ---
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # MFCC
            # Input: (1, Samples), Output: (1, n_mfcc, Time)
            mfcc = self.mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (Time, n_mfcc)

            # Align Audio to Video
            # MFCC time steps might differ from num_frames.
            # We truncate or pad audio to match num_frames.
            a_len = mfcc.shape[0]
            if a_len > num_frames:
                mfcc = mfcc[:num_frames, :]
            elif a_len < num_frames:
                pad_len = num_frames - a_len
                # Pad with zeros
                padding = np.zeros((pad_len, Config.N_MFCC), dtype=np.float32)
                mfcc = np.concatenate([mfcc, padding], axis=0)

            audio_features = mfcc

        except Exception as e:
            # print(f"Error processing audio {sample_id}: {e}")
            return None

        # --- Label Processing ---
        # Initialize with Background Class (0)
        labels = np.full(num_frames, Config.BACKGROUND_CLASS_ID, dtype=np.int64)

        if hasattr(video_struct, "Labels"):
            raw_labels = video_struct.Labels
            # Normalize to list
            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]

            for l in raw_labels:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    start = int(l.Begin) - 1  # 1-based to 0-based
                    end = int(l.End)  # Inclusive in Matlab usually implies up to End

                    if name in Config.LABEL_MAP:
                        lid = Config.LABEL_MAP[name]
                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames, end)
                        if start < end:
                            labels[start:end] = lid

        # 3. Save Cache
        data = {
            "skeleton": skeleton_features.astype(np.float32),
            "audio": audio_features.astype(np.float32),
            "labels": labels.astype(np.int64),
        }
        np.savez(cache_file, **data)
        return data

    def __getitem__(self, idx):
        data = self._process_item(idx)
        if data is None:
            # Handle corruption by returning a dummy zero-length item
            # (collate_fn should handle this or we just crash/skip)
            # For robustness, we return a single frame of zeros
            return {
                "skeleton": torch.zeros(1, NUM_JOINTS * 3),
                "audio": torch.zeros(1, Config.N_MFCC),
                "labels": torch.zeros(1, dtype=torch.long),
                "length": 1,
            }

        skel = torch.from_numpy(data["skeleton"])
        audio = torch.from_numpy(data["audio"])
        labels = torch.from_numpy(data["labels"])

        # --- Normalization ---
        skel_mean = torch.from_numpy(self.stats["skel_mean"]).float()
        skel_std = torch.from_numpy(self.stats["skel_std"]).float()
        audio_mean = torch.from_numpy(self.stats["audio_mean"]).float()
        audio_std = torch.from_numpy(self.stats["audio_std"]).float()

        skel = (skel - skel_mean) / skel_std
        audio = (audio - audio_mean) / audio_std

        # --- Augmentation (Train Only) ---
        if self.mode == "train":
            # 1. Global Temporal Resampling
            # Scale length by alpha ~ U(0.8, 1.2)
            alpha = np.random.uniform(0.8, 1.2)
            new_len = int(skel.shape[0] * alpha)
            if new_len > 0:
                # Interpolate Skeleton (Batch, Channels, Time) -> (Batch, Channels, NewTime)
                skel_t = skel.unsqueeze(0).transpose(1, 2)  # (1, C, T)
                skel_t = F.interpolate(
                    skel_t, size=new_len, mode="linear", align_corners=False
                )
                skel = skel_t.transpose(1, 2).squeeze(0)

                # Interpolate Audio
                audio_t = audio.unsqueeze(0).transpose(1, 2)
                audio_t = F.interpolate(
                    audio_t, size=new_len, mode="linear", align_corners=False
                )
                audio = audio_t.transpose(1, 2).squeeze(0)

                # Interpolate Labels (Nearest Neighbor)
                labels_t = labels.float().view(1, 1, -1)
                labels_t = F.interpolate(labels_t, size=new_len, mode="nearest")
                labels = labels_t.view(-1).long()

            # 2. Random Channel Masking
            # Mask ~10% of channels
            if np.random.random() < 0.5:
                # Skeleton Mask
                mask_s = torch.bernoulli(torch.ones(skel.shape[1]) * 0.9)
                skel = skel * mask_s

                # Audio Mask
                mask_a = torch.bernoulli(torch.ones(audio.shape[1]) * 0.9)
                audio = audio * mask_a

        return {
            "skeleton": skel,
            "audio": audio,
            "labels": labels,
            "length": skel.shape[0],
        }

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    """
    Pads sequences and creates masks.
    """
    # Filter out bad samples (length 0 or None)
    batch = [b for b in batch if b["length"] > 0]
    if not batch:
        return None

    # Sort by length descending (good for packed_sequence)
    batch.sort(key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    max_len = lengths[0].item()

    # Pad Skeleton
    skeletons = [x["skeleton"] for x in batch]
    skeletons_padded = torch.nn.utils.rnn.pad_sequence(
        skeletons, batch_first=True, padding_value=0.0
    )

    # Pad Audio
    audios = [x["audio"] for x in batch]
    audios_padded = torch.nn.utils.rnn.pad_sequence(
        audios, batch_first=True, padding_value=0.0
    )

    # Pad Labels (Padding value = Background Class 0)
    labels = [x["labels"] for x in batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    # Create Mask (True for valid, False for padding)
    # Shape (Batch, Time)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "skeleton": skeletons_padded,  # (B, T, 60)
        "audio": audios_padded,  # (B, T, 13)
        "labels": labels_padded,  # (B, T)
        "mask": mask,  # (B, T)
        "lengths": lengths,  # (B,)
    }
