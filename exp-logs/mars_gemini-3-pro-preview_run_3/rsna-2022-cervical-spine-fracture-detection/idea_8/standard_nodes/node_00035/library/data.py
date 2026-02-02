import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings

from library.config import Config
from library.dicom_utils import load_scan, get_z_position

# Attempt import for header reading (metadata mapping)
try:
    import pydicom
except ImportError:
    pydicom = None


class CervicalSpineDataset(Dataset):
    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use disk caching.
        """
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_METADATA)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_METADATA)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load Bounding Boxes (only for train/val)
        self.box_df = None
        if mode in ["train", "val"] and os.path.exists(Config.BOUNDING_BOXES_PATH):
            self.box_df = pd.read_csv(Config.BOUNDING_BOXES_PATH)

        # Pre-group boxes by StudyInstanceUID for O(1) access
        self.boxes_by_study = {}
        if self.box_df is not None:
            # Filter to only relevant columns
            # Assuming columns: StudyInstanceUID, x, y, width, height, slice_number
            # We map fracture types based on the slice?
            # The competition boxes usually don't label the vertebra directly in the box row
            # but the task implies we need to know which vertebra is fractured.
            # However, the prompt says: "Map the provided fracture bounding boxes... If a bounding box for 'C1' exists..."
            # The provided box csv usually just has geometry.
            # We will infer the vertebra from the segmentation or just use the box presence as a generic fracture signal
            # OR we use the global labels to infer.
            # Strict reading: "If a bounding box for 'C1' exists".
            # Since the box file often doesn't have the class, we will use the global labels to gate the box targets
            # or assume the box indicates a fracture on that slice.
            # For this implementation, we treat any box on a slice as a positive target for the fractured vertebrae
            # present in that study, or simply as a binary "fracture present on slice" signal.
            # Given the output is (64, 7), we'll assign the box to all positive global classes for that study
            # (Weakly supervised localization refinement).
            for study_id, group in self.box_df.groupby("StudyInstanceUID"):
                self.boxes_by_study[study_id] = group

    def __len__(self):
        return len(self.df)

    def _get_slice_map(self, study_dir):
        """
        Maps InstanceNumber to Sorted Index (Z-order).
        Returns: dict {instance_number: sorted_index}
        """
        if not pydicom:
            return {}

        files = glob.glob(os.path.join(study_dir, "*.dcm"))
        if not files:
            # Try recursive or other patterns if needed, but keeping simple as per utils
            files = glob.glob(os.path.join(study_dir, "*"))

        slices = []
        for f in files:
            try:
                # Fast header read
                dcm = pydicom.dcmread(f, stop_before_pixels=True)
                z = get_z_position(dcm)
                inst = getattr(dcm, "InstanceNumber", -1)
                slices.append((z, int(inst)))
            except:
                continue

        # Sort by Z
        slices.sort(key=lambda x: x[0])

        # Map InstanceNumber -> Index in sorted volume
        return {s[1]: i for i, s in enumerate(slices)}

    def _process_and_cache(self, study_id, image_path, labels):
        """
        Loads volume, samples slices, generates targets, and caches to disk.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")
        full_image_path = os.path.join(Config.INPUT_DIR, image_path)

        # 1. Try Load from Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                return data
            except Exception:
                pass  # Corrupt cache, recompute

        # 2. Compute
        # A. Get Slice Mapping (for boxes)
        slice_map = {}
        if self.mode in ["train", "val"]:
            slice_map = self._get_slice_map(full_image_path)

        # B. Load Volume (D, H, W) float32 [0, 1]
        volume = load_scan(full_image_path, resize_to=Config.IMAGE_SIZE)

        depth = volume.shape[0]
        seq_len = Config.SEQ_LEN

        if depth == 0:
            # Handle empty/corrupt scans
            volume = np.zeros((seq_len, *Config.IMAGE_SIZE), dtype=np.float32)
            indices = np.arange(seq_len)
            depth = seq_len
        else:
            # C. Uniform Sampling
            # We want exactly SEQ_LEN slices
            indices = np.linspace(0, depth - 1, seq_len).astype(int)

        # D. Construct 2.5D Stacks
        # Output: (Seq, 3, H, W)
        stacks = np.zeros(
            (seq_len, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.float32
        )

        for i, idx in enumerate(indices):
            # Channels: z-1, z, z+1
            # Clamp to [0, depth-1]
            idx_prev = max(0, idx - 1)
            idx_curr = idx
            idx_next = min(depth - 1, idx + 1)

            if depth > 0:
                stacks[i, 0] = volume[idx_prev]
                stacks[i, 1] = volume[idx_curr]
                stacks[i, 2] = volume[idx_next]

        # E. Generate Box Targets (Sparse Mask)
        # Shape: (Seq, 7)
        box_targets = np.zeros((seq_len, 7), dtype=np.float32)
        has_box = 0.0

        if self.mode in ["train", "val"] and study_id in self.boxes_by_study:
            study_boxes = self.boxes_by_study[study_id]

            # Identify which of the sampled indices correspond to a box
            # We iterate through boxes, find their volume index, and check if that index
            # is close to one of our sampled indices.

            # Invert slice_map for lookup: Index -> InstanceNumber (not needed directly)
            # We need: Box(InstanceNumber) -> VolumeIndex

            for _, row in study_boxes.iterrows():
                slice_num = int(row["slice_number"])
                if slice_num in slice_map:
                    vol_idx = slice_map[slice_num]

                    # Find if this vol_idx is in our sampled 'indices'
                    # Since we sample, we might miss the exact slice.
                    # We assign the box to the nearest sampled slice.
                    dist = np.abs(indices - vol_idx)
                    min_dist_idx = np.argmin(dist)

                    # Threshold: only assign if reasonably close (e.g. within sampling stride)
                    stride = depth / seq_len
                    if dist[min_dist_idx] < max(2.0, stride):
                        # This sampled slice contains a fracture
                        # Which subtype? The boxes don't say.
                        # We use the global labels to turn on relevant columns.
                        # If global C1 is 1, and we have a box, we assume it *might* be C1.
                        # This is a heuristic for "Hybrid Supervised".
                        # If labels are provided:
                        if labels is not None:
                            for c_idx, col in enumerate(
                                ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                            ):
                                if labels[col] == 1:
                                    box_targets[min_dist_idx, c_idx] = 1.0
                        else:
                            # Fallback if no labels (shouldn't happen in train): set all to 1?
                            # Or just ignore.
                            pass

                        has_box = 1.0

        # Convert stacks to uint8 for storage
        stacks_uint8 = (stacks * 255).astype(np.uint8)

        data = {"images": stacks_uint8, "box_targets": box_targets, "has_box": has_box}

        # Save to cache
        if self.load_cached_data:
            np.save(cache_path, data)

        return data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Get Global Labels
        # [C1...C7, patient_overall]
        targets = np.zeros(8, dtype=np.float32)
        labels_dict = None

        if self.mode != "test":
            cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            targets = row[cols].values.astype(np.float32)
            labels_dict = row[cols].to_dict()

        # Load Data (Images + Box Targets)
        data = self._process_and_cache(study_id, row["image_path"], labels_dict)

        # Unpack
        images = data["images"].astype(np.float32) / 255.0  # (Seq, 3, H, W)
        box_targets = data["box_targets"]  # (Seq, 7)
        has_box = np.float32(data["has_box"])

        # Augmentation
        # We need to apply the same transform to all slices in the sequence
        if self.transform:
            seq, c, h, w = images.shape
            # Reshape to (H, W, Seq*C) to apply 2D transform consistently
            images_reshaped = images.transpose(2, 3, 0, 1).reshape(h, w, seq * c)

            augmented = self.transform(image=images_reshaped)["image"]

            # Reshape back to (Seq, C, H, W)
            # Albumentations returns tensor if ToTensorV2 is used, or numpy
            if isinstance(augmented, torch.Tensor):
                # (C_total, H, W) -> (Seq, C, H, W)
                images = augmented.view(seq, c, h, w)
            else:
                images = augmented.reshape(h, w, seq, c).transpose(2, 3, 0, 1)
                images = torch.from_numpy(images)
        else:
            images = torch.from_numpy(images)

        # Convert targets to tensors
        targets = torch.tensor(targets, dtype=torch.float32)
        box_targets = torch.tensor(box_targets, dtype=torch.float32)
        has_box = torch.tensor(has_box, dtype=torch.float32)

        return images, targets, box_targets, has_box, study_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
                A.Normalize(
                    mean=(0.5,), std=(0.5,)
                ),  # Normalize to roughly [-1, 1] or centered
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test.
    """
    # Train
    train_ds = CervicalSpineDataset(
        mode="train", transform=get_transforms("train"), load_cached_data=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_ds = CervicalSpineDataset(
        mode="val", transform=get_transforms("val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test
    test_ds = CervicalSpineDataset(
        mode="test", transform=get_transforms("test"), load_cached_data=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
