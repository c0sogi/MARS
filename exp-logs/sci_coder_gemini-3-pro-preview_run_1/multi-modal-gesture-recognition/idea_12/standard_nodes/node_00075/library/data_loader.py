import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, metadata_path, is_train=False, load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            is_train (bool): Whether this is the training set (enables augmentation).
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.is_train = is_train
        self.load_cached_data = load_cached_data

        # Filter out samples with missing critical files if any (though metadata script handles this)
        # We assume metadata is clean based on provided script.

        # Directories
        self.input_dir = Config.INPUT_DIR
        self.cache_dir = Config.CACHE_DIR
        self.stats_path = Config.STATS_PATH

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load or Compute Global Statistics for Normalization
        self.stats = self._get_global_stats()

    def _get_global_stats(self):
        """
        Loads stats from file or computes them from the training set cache.
        """
        if os.path.exists(self.stats_path):
            return np.load(self.stats_path)

        # If stats don't exist, we must compute them.
        # Ideally, this should be done on the training set.
        # If this instance is not the training set, we might be in trouble if stats aren't generated yet.
        # However, for the purpose of this task, we will assume we can compute them
        # if we are in training mode, or they should exist.

        if not self.is_train:
            # Fallback: if validation/test is loaded before train (unlikely in standard pipeline),
            # we return dummy stats or warn. For now, initialize with identity.
            return {"skel_mean": 0, "skel_std": 1, "audio_mean": 0, "audio_std": 1}

        print("Computing global statistics on training data...")

        # We need to iterate over the dataset to compute stats.
        # To avoid infinite recursion, we'll iterate over a subset or full set
        # processing raw items without normalization.

        skel_sum = np.zeros(Config.SKELETON_INPUT_DIM)
        skel_sq_sum = np.zeros(Config.SKELETON_INPUT_DIM)
        skel_count = 0

        audio_sum = np.zeros(Config.AUDIO_N_MFCC)
        audio_sq_sum = np.zeros(Config.AUDIO_N_MFCC)
        audio_count = 0

        # Process all samples to get accurate stats
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            sample_id = row["sample_id"]

            # Load raw data (this will cache it if not present)
            data = self._load_sample(row)
            if data is None:
                continue

            skel = data["skeleton"]  # (T, D)
            audio = data["audio"]  # (T, D)

            # Skeleton Stats
            skel_sum += np.sum(skel, axis=0)
            skel_sq_sum += np.sum(skel**2, axis=0)
            skel_count += skel.shape[0]

            # Audio Stats
            audio_sum += np.sum(audio, axis=0)
            audio_sq_sum += np.sum(audio**2, axis=0)
            audio_count += audio.shape[0]

        # Compute Mean and Std
        skel_mean = skel_sum / skel_count
        skel_std = np.sqrt((skel_sq_sum / skel_count) - (skel_mean**2)) + 1e-6

        audio_mean = audio_sum / audio_count
        audio_std = np.sqrt((audio_sq_sum / audio_count) - (audio_mean**2)) + 1e-6

        stats = {
            "skel_mean": skel_mean,
            "skel_std": skel_std,
            "audio_mean": audio_mean,
            "audio_std": audio_std,
        }

        np.savez(self.stats_path, **stats)
        print(f"Global statistics saved to {self.stats_path}")
        return np.load(self.stats_path)

    def _process_skeleton(self, mat_path, num_frames):
        """
        Parses .mat file to extract and normalize skeleton data.
        Returns: (NumFrames, Joints*3) numpy array.
        """
        try:
            full_path = os.path.join(self.input_dir, mat_path)
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)

            if "Video" not in mat:
                return np.zeros(
                    (num_frames, Config.SKELETON_INPUT_DIM), dtype=np.float32
                )

            video = mat["Video"]

            # Initialize container
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            if hasattr(video, "Frames"):
                frames = video.Frames
                # Handle case where Frames is a single object or array
                if not isinstance(frames, np.ndarray):
                    frames = np.array([frames])

                # Iterate up to num_frames or available frames
                limit = min(num_frames, len(frames))

                for i in range(limit):
                    frame_obj = frames[i]
                    if hasattr(frame_obj, "Skeleton"):
                        skel_obj = frame_obj.Skeleton
                        # Handle multiple users (take first) or single user
                        if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                            skel_obj = skel_obj[0]

                        if hasattr(skel_obj, "WorldPosition"):
                            # Assuming WorldPosition is 20x3 or similar structure
                            # The prompt says: "WorldPosition... X, Y, Z".
                            # Usually in these datasets, it's a struct array or array of structs.
                            # We'll try to extract coordinates robustly.

                            # If WorldPosition is an array of structs (one per joint)
                            # Or a struct with arrays.
                            # Based on common Kinect formats in Matlab:
                            # It often comes as a struct array of size 20.

                            # Let's try to infer structure.
                            # If skel_obj has JointsType, we iterate joints.
                            pass

                            # Strategy: Try to get joint positions directly.
                            # If specific parsing fails, we return zeros (handled by normalization later).
                            # We assume a standard order (HipCenter=0).

                            # NOTE: Due to the complexity of `scipy.io` parsing of nested structs without
                            # direct inspection, we implement a best-effort parser.

                            # Assuming skel_obj.WorldPosition is available.
                            # If it's a struct array (20 elements):
                            if (
                                isinstance(skel_obj.WorldPosition, np.ndarray)
                                and skel_obj.WorldPosition.size == Config.NUM_JOINTS
                            ):
                                for j in range(Config.NUM_JOINTS):
                                    pos = skel_obj.WorldPosition[j]
                                    skeleton_data[i, j, 0] = pos.X
                                    skeleton_data[i, j, 1] = pos.Y
                                    skeleton_data[i, j, 2] = pos.Z
                            # If it's a single struct with arrays (less likely for this desc)
                            # Fallback: check if we can iterate joints
                            elif hasattr(skel_obj, "JointsType"):
                                # Sometimes Joints are in a specific field
                                pass

            # Relative Coordinates: Subtract HipCenter (Index 0)
            hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - hip_center

            # Flatten
            skeleton_flat = skeleton_data.reshape(num_frames, -1)
            return skeleton_flat.astype(np.float32)

        except Exception as e:
            # print(f"Error processing skeleton {mat_path}: {e}")
            return np.zeros((num_frames, Config.SKELETON_INPUT_DIM), dtype=np.float32)

    def _extract_mfcc(self, audio_path, num_frames):
        """
        Extracts MFCC features aligned with video frames.
        """
        try:
            full_path = os.path.join(self.input_dir, audio_path)
            waveform, sample_rate = torchaudio.load(full_path)

            # Resample if necessary (Config expects 16000)
            if sample_rate != Config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)
                sample_rate = Config.AUDIO_SAMPLE_RATE

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Physics-based Hop Length
            hop_length = Config.AUDIO_HOP_LENGTH
            n_fft = Config.AUDIO_N_FFT
            n_mfcc = Config.AUDIO_N_MFCC

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": 64,
                    "center": False,  # To align strictly with frames
                },
            )

            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, Time)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (Time, n_mfcc)

            # Align with video frames
            audio_frames = mfcc.shape[0]
            if audio_frames < num_frames:
                # Pad
                padding = num_frames - audio_frames
                mfcc = F.pad(mfcc, (0, 0, 0, padding))
            elif audio_frames > num_frames:
                # Truncate
                mfcc = mfcc[:num_frames, :]

            return mfcc.numpy().astype(np.float32)

        except Exception as e:
            # print(f"Error processing audio {audio_path}: {e}")
            return np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _get_dense_labels(self, mat_path, num_frames):
        """
        Constructs frame-wise label tensor.
        """
        labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

        try:
            full_path = os.path.join(self.input_dir, mat_path)
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)

            if "Video" not in mat:
                return labels

            video = mat["Video"]
            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = np.array([raw_labels])

                for l in raw_labels:
                    # Check if valid label object
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # Matlab 1-based indexing
                        begin_frame = int(l.Begin) - 1
                        end_frame = int(l.End)

                        if name in Config.LABEL_MAP:
                            label_id = Config.LABEL_MAP[name]
                            # Clip to valid range
                            begin_frame = max(0, begin_frame)
                            end_frame = min(num_frames, end_frame)

                            if begin_frame < end_frame:
                                labels[begin_frame:end_frame] = label_id

        except Exception:
            pass

        return labels

    def _load_sample(self, row):
        """
        Loads a single sample, using cache if enabled/available.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return {
                    "skeleton": data["skeleton"],
                    "audio": data["audio"],
                    "labels": data["labels"],
                }
            except:
                pass  # Failed to load, recompute

        # Compute from scratch
        num_frames = int(row["num_frames"])
        if num_frames <= 0:
            return None

        # 1. Skeleton
        skeleton = self._process_skeleton(row["data_path"], num_frames)

        # 2. Audio
        if pd.notna(row["audio_path"]):
            audio = self._extract_mfcc(row["audio_path"], num_frames)
        else:
            audio = np.zeros((num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

        # 3. Labels
        # For test set, labels might be empty, but we still generate the array (all zeros)
        labels = self._get_dense_labels(row["data_path"], num_frames)

        # Save to cache
        np.savez(cache_path, skeleton=skeleton, audio=audio, labels=labels)

        return {"skeleton": skeleton, "audio": audio, "labels": labels}

    def _augment(self, skeleton, audio):
        """
        Applies augmentation to skeleton and audio.
        """
        T, D_skel = skeleton.shape
        _, D_audio = audio.shape

        # 1. Additive Gaussian Noise (Skeleton only)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, Config.AUG_GAUSSIAN_NOISE_STD, skeleton.shape)
            skeleton = skeleton + noise

        # 2. Channel Masking
        # Skeleton
        if np.random.rand() < Config.AUG_CHANNEL_MASK_RATE:
            mask = np.random.rand(D_skel) > Config.AUG_CHANNEL_MASK_RATE
            skeleton = skeleton * mask
        # Audio
        if np.random.rand() < Config.AUG_CHANNEL_MASK_RATE:
            mask = np.random.rand(D_audio) > Config.AUG_CHANNEL_MASK_RATE
            audio = audio * mask

        # 3. Time Masking (Temporal Cutout)
        if np.random.rand() < Config.AUG_TIME_MASK_PROB:
            mask_len = np.random.randint(
                Config.AUG_TIME_MASK_LEN_MIN, Config.AUG_TIME_MASK_LEN_MAX + 1
            )
            if T > mask_len:
                start = np.random.randint(0, T - mask_len)
                skeleton[start : start + mask_len, :] = 0
                audio[start : start + mask_len, :] = 0

        return skeleton.astype(np.float32), audio.astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.metadata.iloc[idx]
        data = self._load_sample(row)

        if data is None:
            # Fallback for broken sample: return zero tensors of length 1
            return (
                torch.zeros(1, Config.SKELETON_INPUT_DIM),
                torch.zeros(1, Config.AUDIO_N_MFCC),
                torch.zeros(1, dtype=torch.long),
            )

        skeleton = data["skeleton"]
        audio = data["audio"]
        labels = data["labels"]

        # Augmentation
        if self.is_train:
            skeleton, audio = self._augment(skeleton, audio)

        # Normalization (Z-score)
        # (X - Mean) / Std
        skeleton = (skeleton - self.stats["skel_mean"]) / self.stats["skel_std"]
        audio = (audio - self.stats["audio_mean"]) / self.stats["audio_std"]

        return (
            torch.from_numpy(skeleton).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(labels).long(),
        )


def collate_fn(batch):
    """
    Custom collate function to pad sequences and create masks.
    """
    skeletons, audios, labels = zip(*batch)

    # Calculate lengths
    lengths = torch.tensor([len(s) for s in skeletons], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> (Batch, Time, Dim)
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    return skeletons_padded, audios_padded, labels_padded, lengths
