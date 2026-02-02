import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
from torch.utils.data import Dataset

# Import configuration
from library.config import (
    INPUT_DIR,
    MFCC_N_MFCC,
    MFCC_HOP_LENGTH,
    MFCC_N_FFT,
    AUDIO_SAMPLE_RATE,
    WORKING_DIR,
    LABEL_MAP,
)


class GestureDataset(Dataset):
    def __init__(self, metadata_path, load_cached_data=True, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            load_cached_data (bool): Whether to use cached pre-processed data.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)
        self.cache_dir = os.path.join(WORKING_DIR, "idea_1", f"cache_{mode}")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # List to store paths of valid processed samples
        self.data_files = []

        # Process and cache data
        self._prepare_data(load_cached_data)

    def _prepare_data(self, load_cached_data):
        """
        Iterates through metadata, processes samples, and caches them.
        """
        valid_files_list_path = os.path.join(self.cache_dir, "file_list.npy")

        # If caching is enabled and the list exists, load it
        if load_cached_data and os.path.exists(valid_files_list_path):
            try:
                self.data_files = np.load(
                    valid_files_list_path, allow_pickle=True
                ).tolist()
                if len(self.data_files) > 0 and os.path.exists(self.data_files[0]):
                    print(
                        f"Loaded {len(self.data_files)} samples from cache for {self.mode}."
                    )
                    return
            except Exception:
                pass  # Fallback to reprocessing

        print(f"Processing {self.mode} dataset...")
        processed_files = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

            # Check if individual file exists if loading cached
            if load_cached_data and os.path.exists(cache_path):
                processed_files.append(cache_path)
                continue

            # Process from scratch
            try:
                features, targets = self._process_sample(row)
                if features is not None:
                    # Save to compressed numpy file
                    np.savez_compressed(
                        cache_path,
                        features=features,
                        targets=targets,
                        sample_id=sample_id,
                    )
                    processed_files.append(cache_path)
            except Exception:
                continue

        self.data_files = processed_files
        # Save the list of valid files
        np.save(valid_files_list_path, np.array(self.data_files))
        print(f"Processed {len(self.data_files)} samples.")

    def _process_sample(self, row):
        """
        Loads and fuses Skeleton and Audio data, and generates labels.
        """
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )

        # 1. Load MAT file
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = video.NumFrames
        except:
            return None, None

        # 2. Process Skeleton
        skeleton_features = self._extract_skeleton(video, num_frames)
        if skeleton_features is None:
            return None, None

        # 3. Process Audio
        audio_features = self._extract_audio(audio_path, num_frames)
        if audio_features is None:
            # Fallback: zero audio features if audio is missing/corrupt but skeleton exists
            audio_features = np.zeros((num_frames, MFCC_N_MFCC), dtype=np.float32)

        # 4. Fuse Features
        # Ensure lengths match (truncate to min length)
        min_len = min(len(skeleton_features), len(audio_features))
        features = np.concatenate(
            [skeleton_features[:min_len], audio_features[:min_len]], axis=1
        )

        # 5. Generate Targets
        targets = np.zeros(min_len, dtype=np.int64)  # Default 0 (Background)

        if self.mode != "test" and hasattr(video, "Labels"):
            labels_raw = video.Labels
            # Handle single label vs list of labels
            if not isinstance(labels_raw, np.ndarray):
                labels_raw = [labels_raw]
            elif labels_raw.size == 1:
                labels_raw = [labels_raw.item()]

            for l in labels_raw:
                try:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # Matlab 1-based indexing -> Python 0-based
                        start_frame = int(l.Begin) - 1
                        end_frame = int(l.End)

                        if name in LABEL_MAP:
                            label_id = LABEL_MAP[name]
                            # Clip to valid range
                            s = max(0, start_frame)
                            e = min(min_len, end_frame)
                            if s < e:
                                targets[s:e] = label_id
                except:
                    continue

        return features.astype(np.float32), targets

    def _extract_skeleton(self, video, num_frames):
        """
        Extracts and normalizes skeleton data.
        Returns: (T, 60) numpy array.
        """
        try:
            frames = video.Frames
            if not isinstance(frames, np.ndarray) or len(frames) == 0:
                return None

            # Pre-allocate (T, 20, 3)
            skeleton_data = np.zeros((len(frames), 20, 3), dtype=np.float32)

            # HipCenter index assumption (standard Kinect index 0)
            hip_index = 0

            for i, frame_obj in enumerate(frames):
                if i >= num_frames:
                    break

                skel_struct = frame_obj.Skeleton
                # Handle multiple users: pick the first one
                if isinstance(skel_struct, np.ndarray):
                    if len(skel_struct) == 0:
                        continue
                    skel_struct = skel_struct[0]

                if hasattr(skel_struct, "WorldPosition"):
                    wp = skel_struct.WorldPosition

                    if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                        skeleton_data[i] = wp
                    # If wp is a struct with X, Y, Z arrays/scalars
                    elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        x = wp.X if isinstance(wp.X, np.ndarray) else np.array([wp.X])
                        y = wp.Y if isinstance(wp.Y, np.ndarray) else np.array([wp.Y])
                        z = wp.Z if isinstance(wp.Z, np.ndarray) else np.array([wp.Z])

                        if len(x) == 20:
                            skeleton_data[i] = np.column_stack((x, y, z))

            # Normalize: Relative to HipCenter
            root_pos = skeleton_data[:, hip_index : hip_index + 1, :]  # (T, 1, 3)
            relative_pos = skeleton_data - root_pos

            # Flatten to (T, 60)
            return relative_pos.reshape(len(frames), -1)

        except Exception:
            return None

    def _extract_audio(self, audio_path, num_frames):
        """
        Extracts MFCC features aligned to video frames.
        Returns: (T, 13) numpy array.
        """
        if not audio_path or not os.path.exists(audio_path):
            return None

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=AUDIO_SAMPLE_RATE,
                n_mfcc=MFCC_N_MFCC,
                melkwargs={
                    "n_fft": MFCC_N_FFT,
                    "hop_length": MFCC_HOP_LENGTH,
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # Shape: (time, n_mfcc)

            return mfcc.numpy()

        except Exception:
            return None

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = np.load(file_path)

        features = torch.from_numpy(data["features"])
        targets = torch.from_numpy(data["targets"])
        sample_id = str(data["sample_id"])

        return features, targets, sample_id


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    """
    features_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    ids = [item[2] for item in batch]

    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)

    # Pad features with 0
    padded_features = torch.nn.utils.rnn.pad_sequence(
        features_list, batch_first=True, padding_value=0.0
    )

    # Pad targets with 0 (Background class)
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets_list, batch_first=True, padding_value=0
    )

    return padded_features, padded_targets, lengths, ids
