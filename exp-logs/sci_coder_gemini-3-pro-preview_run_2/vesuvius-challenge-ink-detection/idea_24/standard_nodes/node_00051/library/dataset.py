import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import PATHS, SPECIALIST_SETTINGS, AUGMENTATION_PARAMS, SLAB_PARAMS
from library.data_utils import get_fragment_3ch_slab


class InkDataset(Dataset):
    def __init__(self, metadata, specialist_mode, split="train", transform=None):
        """
        Dataset for Vesuvius Ink Detection using Matched-Depth Specialist Ensemble strategy.

        Args:
            metadata: pandas DataFrame or path to csv containing patch metadata.
            specialist_mode: 'High', 'Mid', or 'Low'. Determines Z-range.
            split: 'train', 'val', or 'test'.
            transform: Optional albumentations transform. If None, default is used based on split.
        """
        super().__init__()

        # Validate specialist mode
        if specialist_mode not in SPECIALIST_SETTINGS:
            raise ValueError(
                f"Invalid specialist_mode '{specialist_mode}'. Must be one of {list(SPECIALIST_SETTINGS.keys())}"
            )

        self.mode_settings = SPECIALIST_SETTINGS[specialist_mode]
        self.split = split
        self.z_start = self.mode_settings["z_start"]
        self.z_end = self.mode_settings["z_end"]

        # Load Metadata
        if isinstance(metadata, str):
            self.df = pd.read_csv(metadata)
        else:
            self.df = metadata

        # Preload Fragments into Memory
        # We load the full projected slabs once to maximize throughput.
        self.fragment_images = {}
        self.fragment_labels = {}

        # Identify unique fragments to load
        unique_fragments = self.df["fragment_id"].unique()

        # Determine base directory for labels
        if split == "test":
            base_dir = PATHS.TEST_FRAGMENTS
        else:
            base_dir = PATHS.TRAIN_FRAGMENTS

        print(
            f"Initializing InkDataset ({split}) for Specialist '{specialist_mode}' (Z: {self.z_start}-{self.z_end})"
        )

        for frag_id in unique_fragments:
            # 1. Load Image Slab (Cached)
            # Map 'val' split to 'train' for data loading as they share the same source directory
            data_split = "train" if split in ["train", "val"] else "test"

            # Load the specific specialist view using the utility function
            # This handles projection, normalization, and disk caching
            image_slab = get_fragment_3ch_slab(
                fragment_id=str(frag_id),
                split=data_split,
                z_start=self.z_start,
                z_end=self.z_end,
                slab_params=SLAB_PARAMS,
                load_cached_data=True,
            )
            self.fragment_images[frag_id] = image_slab

            # 2. Load Labels (if available and not test)
            if split != "test":
                label_path = os.path.join(base_dir, str(frag_id), "inklabels.png")
                if os.path.exists(label_path):
                    # Load as grayscale
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if label_img is not None:
                        # Binarize and convert to float32 (0.0, 1.0)
                        label_img = (label_img > 0).astype(np.float32)
                        self.fragment_labels[frag_id] = label_img
                    else:
                        print(f"Warning: Failed to load label for fragment {frag_id}")

        # Setup Transforms
        if transform is not None:
            self.transform = transform
        else:
            # Default Transform Logic
            transforms_list = []

            # Geometric Augmentations (Train only)
            if split == "train" and AUGMENTATION_PARAMS.get("geometric", False):
                transforms_list.extend(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ]
                )

            # Note: Z-jitter and Intensity augmentations are explicitly excluded per MDSE protocol.

            # Convert to Tensor (HWC -> CHW)
            transforms_list.append(ToTensorV2())

            self.transform = A.Compose(transforms_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]

        # Coordinates
        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        # Retrieve full fragment data from memory
        full_image = self.fragment_images[frag_id]

        # Crop Image Patch
        # Metadata generation ensures valid bounds, but simple slicing handles edges gracefully
        image_patch = full_image[y : y + h, x : x + w, :]

        # Retrieve and Crop Label Patch
        label_patch = None
        if self.split != "test" and frag_id in self.fragment_labels:
            full_label = self.fragment_labels[frag_id]
            label_patch = full_label[y : y + h, x : x + w]

        # Apply Transforms
        if self.transform:
            if label_patch is not None:
                # Augment both image and mask
                augmented = self.transform(image=image_patch, mask=label_patch)
                image_tensor = augmented["image"]
                label_tensor = augmented["mask"]

                # Ensure label has channel dimension (1, H, W)
                label_tensor = label_tensor.unsqueeze(0)

                return image_tensor, label_tensor
            else:
                # Augment image only
                augmented = self.transform(image=image_patch)
                image_tensor = augmented["image"]
                return image_tensor

        # Fallback (should not be reached with default init)
        image_tensor = torch.from_numpy(image_patch.transpose(2, 0, 1))
        if label_patch is not None:
            label_tensor = torch.from_numpy(label_patch).unsqueeze(0)
            return image_tensor, label_tensor

        return image_tensor
