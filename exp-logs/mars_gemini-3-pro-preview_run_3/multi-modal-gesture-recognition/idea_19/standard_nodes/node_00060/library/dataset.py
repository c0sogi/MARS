import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library import data_utils


class GestureDataset(Dataset):
    def __init__(self, split_name, augment=False, load_cached=True):
        """
        Args:
            split_name (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
            load_cached (bool): Whether to use cached dataset files.
        """
        self.split_name = split_name
        self.augment = augment

        # Determine CSV path and stride based on split
        if split_name == "train":
            csv_path = Config.TRAIN_CSV
            self.stride = Config.STRIDE_TRAIN
        elif split_name == "val":
            csv_path = Config.VAL_CSV
            self.stride = Config.STRIDE_TEST
        elif split_name == "test":
            csv_path = Config.TEST_CSV
            self.stride = Config.STRIDE_TEST
        else:
            raise ValueError(f"Unknown split: {split_name}")

        # Load raw sequences using library function
        # Returns a list of dicts: {'sample_id', 'skeleton', 'audio', 'labels'}
        self.raw_data = data_utils.load_dataset_and_cache(
            csv_path, split_name, load_cached=load_cached
        )

        # Pre-calculate sliding window indices
        self.windows = []
        self._prepare_windows()

    def _prepare_windows(self):
        """
        Generates a list of (sequence_index, start_frame) tuples.
        Ensures coverage of the entire sequence.
        """
        for seq_idx, item in enumerate(self.raw_data):
            num_frames = item["skeleton"].shape[0]

            # Handle sequences shorter than window size
            if num_frames < Config.WINDOW_SIZE:
                self.windows.append((seq_idx, 0))
                continue

            # Generate start indices
            # Range: 0 to num_frames - window_size, step = stride
            start_indices = list(
                range(0, num_frames - Config.WINDOW_SIZE + 1, self.stride)
            )

            # Ensure the last frames are covered by adding a window ending at the last frame
            if (
                num_frames > Config.WINDOW_SIZE
                and (num_frames - Config.WINDOW_SIZE) % self.stride != 0
            ):
                start_indices.append(num_frames - Config.WINDOW_SIZE)

            # Remove duplicates and sort
            start_indices = sorted(list(set(start_indices)))

            for start in start_indices:
                self.windows.append((seq_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]
        item = self.raw_data[seq_idx]

        # 1. Extract Raw Data Window
        # --------------------------
        full_skel = item["skeleton"]  # (T, 20, 3)
        full_audio = item["audio"]  # (T, 13)
        seq_len = full_skel.shape[0]

        end_frame = start_frame + Config.WINDOW_SIZE

        # Handle padding for short sequences
        if seq_len < Config.WINDOW_SIZE:
            # Pad skeleton with zeros
            skel_window = np.zeros((Config.WINDOW_SIZE, 20, 3), dtype=np.float32)
            skel_window[:seq_len] = full_skel

            # Pad audio
            audio_window = np.zeros(
                (Config.WINDOW_SIZE, Config.AUDIO_INPUT_DIM), dtype=np.float32
            )
            audio_window[:seq_len] = full_audio
        else:
            skel_window = full_skel[start_frame:end_frame]
            audio_window = full_audio[start_frame:end_frame]

        # 2. Feature Engineering (Augmentation + Kinematics + Fusion)
        # -----------------------------------------------------------
        # data_utils.get_feature_vector handles augmentation and kinematics calculation
        # It returns (Window_Size, 193)
        features = data_utils.get_feature_vector(
            skel_window, audio_window, augment=self.augment
        )

        # 3. Label Generation
        # -------------------
        # Initialize labels
        # Class labels: LongTensor (Window,) - Default to Background
        cls_labels = np.full(
            (Config.WINDOW_SIZE,), Config.BACKGROUND_CLASS_ID, dtype=np.int64
        )

        # Boundary labels: FloatTensor (Window,) - Binary, Default to 0
        bnd_labels = np.zeros((Config.WINDOW_SIZE,), dtype=np.float32)

        # Process annotations if they exist (train/val)
        raw_labels = item["labels"]

        if raw_labels and len(raw_labels) > 0:
            for label in raw_labels:
                # Parse label info
                # Metadata contains 1-based indices from Matlab, convert to 0-based
                # Matlab intervals are inclusive [begin, end]
                g_id = int(label["id"])
                g_start = int(label["begin"]) - 1
                g_end = int(label["end"]) - 1

                # --- Classification Labels ---
                # Calculate intersection between gesture and current window
                inter_start = max(start_frame, g_start)
                inter_end = min(start_frame + Config.WINDOW_SIZE - 1, g_end)

                if inter_start <= inter_end:
                    # Map to window-relative indices
                    w_idx_start = inter_start - start_frame
                    w_idx_end = inter_end - start_frame

                    # Assign class ID (slice end is exclusive)
                    cls_labels[w_idx_start : w_idx_end + 1] = g_id

                # --- Boundary Labels ---
                # Mark frames within +/- 1 frame radius of transition points
                transitions = [g_start, g_end]

                for t_point in transitions:
                    for offset in [-1, 0, 1]:
                        p = t_point + offset
                        # Check if point p is within the current window
                        if start_frame <= p < (start_frame + Config.WINDOW_SIZE):
                            w_idx = p - start_frame
                            # Safety check for array bounds
                            if 0 <= w_idx < Config.WINDOW_SIZE:
                                bnd_labels[w_idx] = 1.0

        # 4. Convert to Tensors
        # ---------------------
        features_tensor = torch.from_numpy(features).float()  # (64, 193)
        cls_labels_tensor = torch.from_numpy(cls_labels).long()  # (64,)
        bnd_labels_tensor = torch.from_numpy(bnd_labels).float()  # (64,)

        return {
            "features": features_tensor,
            "cls_labels": cls_labels_tensor,
            "bnd_labels": bnd_labels_tensor,
            "sample_id": item["sample_id"],
            "window_start": start_frame,
        }
