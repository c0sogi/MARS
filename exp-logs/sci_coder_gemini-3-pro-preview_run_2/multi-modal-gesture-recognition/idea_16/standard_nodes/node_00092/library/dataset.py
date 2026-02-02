import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from library.config import Config
from library.preprocessing import process_dataset, compute_velocity


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the GLT-CRCN model.
    Handles loading of multi-modal features, generation of frame-wise targets,
    and physically consistent data augmentation.
    """

    def __init__(self, split="train", augment=False, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.augment = augment
        self.debug = debug
        self.gesture_map = Config.get_gesture_map()

        # Define paths based on split
        if split == "train":
            self.metadata_file = os.path.join(Config.METADATA_DIR, "train.csv")
            self.cache_file = "train_data.npz"
            self.target_cache_file = "train_targets.npy"
        elif split == "val":
            self.metadata_file = os.path.join(Config.METADATA_DIR, "val.csv")
            self.cache_file = "val_data.npz"
            self.target_cache_file = "val_targets.npy"
        else:  # test
            self.metadata_file = os.path.join(Config.METADATA_DIR, "test.csv")
            self.cache_file = "test_data.npz"
            self.target_cache_file = None

        # Load Features (cached via preprocessing module)
        # process_dataset returns {sample_id: {'features': ..., 'labels': ...}}
        self.data_dict = process_dataset(
            self.metadata_file, self.cache_file, load_cached_data=True
        )

        # Load Metadata DataFrame (needed for paths to generate frame targets)
        self.meta_df = pd.read_csv(self.metadata_file)
        # Drop rows with missing data paths to ensure consistency with preprocessing
        self.meta_df.dropna(subset=["data_path"], inplace=True)

        # Filter for debug
        if self.debug:
            self.meta_df = self.meta_df.iloc[:50]
            # Filter data_dict keys
            valid_keys = set(self.meta_df["sample_id"].values)
            self.data_dict = {
                k: v for k, v in self.data_dict.items() if k in valid_keys
            }

        self.sample_ids = self.meta_df["sample_id"].values.tolist()

        # Load or Generate Frame-wise Targets (only for train/val)
        self.frame_targets = {}
        if self.split in ["train", "val"]:
            self._prepare_frame_targets()

    def _prepare_frame_targets(self):
        """
        Loads frame-wise targets from cache or generates them from .mat files.
        """
        target_cache_path = os.path.join(Config.WORKING_DIR, self.target_cache_file)

        if os.path.exists(target_cache_path):
            try:
                self.frame_targets = np.load(
                    target_cache_path, allow_pickle=True
                ).item()
                return
            except Exception:
                pass  # Regenerate if load fails

        # Generate targets
        print(f"Generating frame-wise targets for {self.split} set...")
        for _, row in self.meta_df.iterrows():
            sid = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])

            # Get total frames from features to ensure alignment
            if sid in self.data_dict:
                num_frames = self.data_dict[sid]["features"].shape[0]
            else:
                num_frames = (
                    int(row["num_frames"]) if pd.notna(row["num_frames"]) else 100
                )

            # Default: Background (0)
            targets = np.zeros(num_frames, dtype=np.int64)

            try:
                mat = scipy.io.loadmat(
                    mat_path, squeeze_me=True, struct_as_record=False
                )
                if "Video" in mat:
                    video = mat["Video"]
                    labels_raw = getattr(video, "Labels", [])

                    # Helper to process a single label object
                    def process_label_obj(obj):
                        try:
                            name = obj.Name
                            # Matlab is 1-based, convert to 0-based
                            start = int(obj.Begin) - 1
                            end = int(obj.End) - 1

                            if name in self.gesture_map:
                                gid = self.gesture_map[name]
                                # Clip to valid range
                                start = max(0, start)
                                end = min(num_frames - 1, end)
                                if end >= start:
                                    targets[start : end + 1] = gid
                        except AttributeError:
                            pass

                    if isinstance(labels_raw, np.ndarray):
                        if labels_raw.ndim == 0:
                            process_label_obj(labels_raw.item())
                        else:
                            for l in labels_raw:
                                process_label_obj(l)
                    else:
                        process_label_obj(labels_raw)
            except Exception:
                # If loading fails, targets remain all zeros (background)
                pass

            self.frame_targets[sid] = targets

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(target_cache_path, self.frame_targets)
        print(f"Saved frame targets to {target_cache_path}")

    def _augment_features(self, features):
        """
        Applies Physically Consistent Smooth Noise.
        features: (T, 85) -> [Pos(36), Vel(36), Audio(13)]
        """
        T = features.shape[0]

        # 1. Extract Position: (T, 36) -> (T, 12, 3)
        pos_flat = features[:, :36]
        pos = pos_flat.reshape(T, 12, 3)

        # 2. Generate Gaussian Noise
        noise = np.random.normal(0, Config.NOISE_STD, pos.shape)

        # 3. Apply Temporal Low-Pass Filter
        # Simple box filter approximation or uniform kernel
        kernel_size = Config.TEMPORAL_FILTER_WIDTH
        kernel = np.ones(kernel_size) / kernel_size

        # Apply smoothing along time axis (axis 0) for each joint/coord
        smooth_noise = np.zeros_like(noise)
        for j in range(12):
            for c in range(3):
                smooth_noise[:, j, c] = convolve1d(
                    noise[:, j, c], kernel, mode="nearest"
                )

        # 4. Add noise to positions
        new_pos = pos + smooth_noise

        # 5. Re-compute Velocity from noisy positions to ensure consistency
        # compute_velocity expects (T, 12, 3)
        new_vel = compute_velocity(new_pos)  # Returns (T, 12, 3)

        # 6. Re-assemble
        new_pos_flat = new_pos.reshape(T, 36)
        new_vel_flat = new_vel.reshape(T, 36)
        audio = features[:, 72:]

        augmented_features = np.concatenate([new_pos_flat, new_vel_flat, audio], axis=1)
        return augmented_features.astype(np.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        data = self.data_dict[sid]

        features = data["features"]  # (T, 85)

        # Apply Augmentation
        if self.augment:
            features = self._augment_features(features)

        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()

        # Get Targets
        if self.split in ["train", "val"]:
            # Frame-wise targets
            targets = self.frame_targets.get(
                sid, np.zeros(features.shape[0], dtype=np.int64)
            )
            # Ensure length match (in case of mismatch between .mat frames and extracted features)
            if len(targets) != features.shape[0]:
                min_len = min(len(targets), features.shape[0])
                targets = targets[:min_len]
                features_tensor = features_tensor[:min_len]

            targets_tensor = torch.from_numpy(targets).long()

            # Sequence labels (for metric calculation/debugging)
            seq_labels = torch.from_numpy(data["labels"]).long()

            return {
                "sample_id": sid,
                "features": features_tensor,
                "targets": targets_tensor,
                "seq_labels": seq_labels,
            }
        else:
            # Test mode
            return {"sample_id": sid, "features": features_tensor}

    @staticmethod
    def collate_fn(batch):
        """
        Pads sequences to the maximum length in the batch.
        """
        sample_ids = [item["sample_id"] for item in batch]
        features = [item["features"] for item in batch]

        # Pad features
        lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
        padded_features = torch.nn.utils.rnn.pad_sequence(
            features, batch_first=True, padding_value=0.0
        )

        # Generate Mask (B, T)
        max_len = padded_features.shape[1]
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(
            1
        )

        batch_out = {
            "sample_ids": sample_ids,
            "features": padded_features,
            "mask": mask,
            "lengths": lengths,
        }

        if "targets" in batch[0]:
            targets = [item["targets"] for item in batch]
            # Pad targets with -1 (ignore index) or 0 (background).
            # Using -1 is safer for CrossEntropyLoss ignore_index.
            padded_targets = torch.nn.utils.rnn.pad_sequence(
                targets, batch_first=True, padding_value=-1
            )
            batch_out["targets"] = padded_targets

            # Also collect sequence labels (list of tensors)
            batch_out["seq_labels"] = [item["seq_labels"] for item in batch]

        return batch_out
