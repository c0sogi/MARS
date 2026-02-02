import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, tries to load from cache.
                                     If False or cache missing, reprocesses data.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Select metadata file
        if split == "train":
            self.metadata_path = Config.TRAIN_CSV
        elif split == "val":
            self.metadata_path = Config.VAL_CSV
        else:
            self.metadata_path = Config.TEST_CSV

        self.df = pd.read_csv(self.metadata_path)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Stats file path
        self.stats_path = os.path.join(os.path.dirname(Config.CACHE_DIR), "stats.npz")

        # Initialize stats containers
        self.skeleton_mean = None
        self.skeleton_std = None
        self.audio_mean = None
        self.audio_std = None

        # Process/Cache data if needed
        self.valid_sample_ids = self._prepare_data()

        # Load stats for normalization
        self._load_stats()

        # Audio Transform (MFCC)
        # 16000Hz / 20fps = 800 hop length
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLERATE,
            n_mfcc=Config.AUDIO_N_MFCC,
            melkwargs={
                "n_fft": (
                    Config.AUDIO_N_MFCC * 4
                    if Config.AUDIO_N_MFCC * 4 > Config.AUDIO_HOP_LENGTH
                    else 2048
                ),
                "hop_length": Config.AUDIO_HOP_LENGTH,
                "n_mels": 64,
                "center": False,  # To align better with frames
            },
        )

    def _prepare_data(self):
        """
        Iterates through metadata, processes raw files if not cached,
        and computes global stats if processing training set.
        Returns list of valid sample IDs.
        """
        valid_ids = []

        # Containers for stats calculation (only used if reprocessing train)
        skel_accumulator = []
        audio_accumulator = []

        # Check if we need to recompute stats (only if training and not loading cached)
        recompute_stats = (self.split == "train") and (
            not self.load_cached_data or not os.path.exists(self.stats_path)
        )

        for idx, row in self.df.iterrows():
            sample_id = row["sample_id"]
            cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

            # Check cache
            if self.load_cached_data and os.path.exists(cache_path):
                valid_ids.append(sample_id)
                continue

            # Process from scratch
            try:
                # 1. Load Skeleton & Labels
                skeleton, labels_seq, num_frames = self._process_mat(row["data_path"])
                if skeleton is None:
                    continue

                # 2. Load Audio
                audio = self._process_audio(row["audio_path"], num_frames)
                if audio is None:
                    continue

                # 3. Save to Cache
                np.savez_compressed(
                    cache_path, skeleton=skeleton, audio=audio, labels=labels_seq
                )
                valid_ids.append(sample_id)

                # Accumulate for stats
                if recompute_stats:
                    # Subsample to save memory if needed, or accumulate sum/sq_sum
                    # Here we take a random subset of frames to estimate stats
                    if len(skeleton) > 0:
                        indices = np.linspace(
                            0, len(skeleton) - 1, min(len(skeleton), 10)
                        ).astype(int)
                        skel_accumulator.append(skeleton[indices])
                        audio_accumulator.append(audio[indices])

            except Exception as e:
                # print(f"Error processing {sample_id}: {e}")
                continue

        # Save stats if recomputed
        if recompute_stats and skel_accumulator:
            all_skel = np.concatenate(skel_accumulator, axis=0)
            all_audio = np.concatenate(audio_accumulator, axis=0)

            np.savez(
                self.stats_path,
                skeleton_mean=np.mean(all_skel, axis=0),
                skeleton_std=np.std(all_skel, axis=0),
                audio_mean=np.mean(all_audio, axis=0),
                audio_std=np.std(all_audio, axis=0),
            )

        return valid_ids

    def _process_mat(self, mat_rel_path):
        """
        Parses MAT file for Skeleton and Labels.
        Returns:
            skeleton (np.ndarray): (T, Joints*3)
            labels_seq (np.ndarray): (T,) frame-wise labels
            num_frames (int)
        """
        full_path = os.path.join(Config.INPUT_DIR, mat_rel_path)
        try:
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)

            if num_frames == 0:
                return None, None, 0

            # --- Extract Skeleton ---
            # Frames -> Skeleton -> WorldPosition
            frames_data = video.Frames
            if (
                not isinstance(frames_data, np.ndarray)
                or len(frames_data) != num_frames
            ):
                # Fallback or strict check
                return None, None, 0

            # Pre-allocate
            # 20 joints, 3 coords
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            # Joint mapping: Prompt lists 20 joints.
            # We assume the order in 'JointsType' matches the storage order in WorldPosition arrays.
            # Usually WorldPosition is (20,) struct array or similar.
            # Optimization: Check first frame to determine structure

            for i in range(num_frames):
                frame_obj = frames_data[i]
                if hasattr(frame_obj, "Skeleton"):
                    skel_obj = frame_obj.Skeleton
                    # Handle multiple users: take first
                    if isinstance(skel_obj, np.ndarray) and skel_obj.size > 0:
                        skel_obj = skel_obj[0]

                    if hasattr(skel_obj, "WorldPosition"):
                        wp = skel_obj.WorldPosition
                        # wp might be (20,) struct array with X,Y,Z fields
                        # or a 20x3 matrix depending on export version.
                        # Based on prompt "WorldPosition... X value... Y value...", likely struct array.

                        if isinstance(wp, np.ndarray) and wp.size == Config.NUM_JOINTS:
                            # Iterate joints
                            for j in range(Config.NUM_JOINTS):
                                joint = wp[j]
                                skeleton_data[i, j, 0] = joint.X
                                skeleton_data[i, j, 1] = joint.Y
                                skeleton_data[i, j, 2] = joint.Z

            # --- Root Relative ---
            # HipCenter is usually index 0. Prompt lists it first.
            hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
            skeleton_data = skeleton_data - hip_center

            # Flatten: (T, 60)
            skeleton_flat = skeleton_data.reshape(num_frames, -1)

            # --- Extract Labels (Frame-wise) ---
            labels_seq = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

            if hasattr(video, "Labels"):
                raw_labels = video.Labels
                if not isinstance(raw_labels, np.ndarray):
                    raw_labels = [raw_labels]
                elif raw_labels.size == 1:
                    raw_labels = [raw_labels.item()]

                for l in raw_labels:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # Matlab 1-based indexing -> Python 0-based
                        start_f = int(l.Begin) - 1
                        end_f = int(l.End) - 1

                        if name in Config.LABEL_MAP:
                            lid = Config.LABEL_MAP[name]
                            # Clip to valid range
                            start_f = max(0, start_f)
                            end_f = min(num_frames - 1, end_f)
                            if end_f >= start_f:
                                labels_seq[start_f : end_f + 1] = lid

            return skeleton_flat, labels_seq, num_frames

        except Exception as e:
            # print(f"Mat parse error: {e}")
            return None, None, 0

    def _process_audio(self, audio_rel_path, target_frames):
        """
        Loads audio, computes MFCC, aligns to target_frames.
        Returns: (T, n_mfcc)
        """
        full_path = os.path.join(Config.INPUT_DIR, audio_rel_path)
        try:
            # Load with torchaudio
            waveform, sample_rate = torchaudio.load(full_path)

            # Resample if necessary
            if sample_rate != Config.AUDIO_SAMPLERATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.AUDIO_SAMPLERATE
                )
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Compute MFCC
            # (1, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)

            # Transpose to (Time, Features) -> (T_aud, 13)
            mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()

            # Align to Video Frames
            # Simple truncation or padding
            curr_len = mfcc.shape[0]
            if curr_len > target_frames:
                mfcc = mfcc[:target_frames]
            elif curr_len < target_frames:
                pad_amt = target_frames - curr_len
                # Pad with zeros
                mfcc = np.pad(mfcc, ((0, pad_amt), (0, 0)), mode="constant")

            return mfcc

        except Exception as e:
            # print(f"Audio parse error: {e}")
            return None

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.skeleton_mean = torch.tensor(
                stats["skeleton_mean"], dtype=torch.float32
            )
            self.skeleton_std = (
                torch.tensor(stats["skeleton_std"], dtype=torch.float32) + 1e-6
            )
            self.audio_mean = torch.tensor(stats["audio_mean"], dtype=torch.float32)
            self.audio_std = (
                torch.tensor(stats["audio_std"], dtype=torch.float32) + 1e-6
            )
        else:
            # Fallback if stats missing (should not happen if train processed)
            self.skeleton_mean = torch.zeros(Config.NUM_JOINTS * 3)
            self.skeleton_std = torch.ones(Config.NUM_JOINTS * 3)
            self.audio_mean = torch.zeros(Config.AUDIO_N_MFCC)
            self.audio_std = torch.ones(Config.AUDIO_N_MFCC)

    def _augment(self, skeleton, audio, labels):
        """
        Applies Temporal Resampling and Channel Masking.
        Inputs are Tensors.
        """
        # 1. Temporal Resampling
        if Config.TEMPORAL_RESAMPLE_MIN < 1.0 or Config.TEMPORAL_RESAMPLE_MAX > 1.0:
            scale = np.random.uniform(
                Config.TEMPORAL_RESAMPLE_MIN, Config.TEMPORAL_RESAMPLE_MAX
            )
            orig_len = skeleton.shape[0]
            new_len = int(orig_len * scale)

            if new_len > 0:
                # Interpolate Skeleton (T, F) -> (1, F, T) for grid_sample/interpolate
                skel_in = skeleton.unsqueeze(0).transpose(1, 2)  # (1, Feat, T)
                skel_out = F.interpolate(
                    skel_in, size=new_len, mode="linear", align_corners=False
                )
                skeleton = skel_out.transpose(1, 2).squeeze(0)

                # Interpolate Audio
                aud_in = audio.unsqueeze(0).transpose(1, 2)
                aud_out = F.interpolate(
                    aud_in, size=new_len, mode="linear", align_corners=False
                )
                audio = aud_out.transpose(1, 2).squeeze(0)

                # Interpolate Labels (Nearest)
                lbl_in = labels.float().view(1, 1, -1)
                lbl_out = F.interpolate(lbl_in, size=new_len, mode="nearest")
                labels = lbl_out.view(-1).long()

        # 2. Channel Masking
        if np.random.random() < Config.CHANNEL_MASK_PROB:
            # Mask Skeleton channels
            mask_s = torch.rand(skeleton.shape[1]) > 0.1
            skeleton = skeleton * mask_s.float()

            # Mask Audio channels
            mask_a = torch.rand(audio.shape[1]) > 0.1
            audio = audio * mask_a.float()

        return skeleton, audio, labels

    def __len__(self):
        return len(self.valid_sample_ids)

    def __getitem__(self, idx):
        sample_id = self.valid_sample_ids[idx]
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        data = np.load(cache_path)
        skeleton = torch.tensor(data["skeleton"], dtype=torch.float32)
        audio = torch.tensor(data["audio"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.long)

        # Augmentation (Train only)
        if self.split == "train":
            skeleton, audio, labels = self._augment(skeleton, audio, labels)

        # Normalization
        skeleton = (skeleton - self.skeleton_mean) / self.skeleton_std
        audio = (audio - self.audio_mean) / self.audio_std

        return {"skeleton": skeleton, "audio": audio, "labels": labels, "id": sample_id}


def collate_fn(batch):
    """
    Pads sequences and creates masks.
    """
    # Sort by length for pack_padded_sequence (optional but good practice)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels = [x["labels"] for x in batch]
    ids = [x["id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad
    skeletons_pad = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audios_pad = pad_sequence(audios, batch_first=True, padding_value=0.0)
    # Pad labels with 0 (Background)
    labels_pad = pad_sequence(
        labels, batch_first=True, padding_value=Config.LABEL_MAP["background"]
    )

    # Mask: True where valid, False where padded
    # Shape: (B, T)
    max_len = lengths.max()
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "skeleton": skeletons_pad,
        "audio": audios_pad,
        "labels": labels_pad,
        "mask": mask,
        "lengths": lengths,
        "ids": ids,
    }
