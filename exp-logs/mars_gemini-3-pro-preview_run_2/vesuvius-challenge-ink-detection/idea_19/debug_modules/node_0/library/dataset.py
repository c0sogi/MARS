import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import random
from torch.utils.data import Dataset
from library.config import Config
from library.utils import generate_multiview_tensor, load_or_process_data


def get_transforms(split):
    """
    Returns the Albumentations transform pipeline based on the split.
    Adheres to the protocol: Geometric only, no intensity augmentation.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    else:
        return A.Compose([])


def _process_dataset(metadata_df):
    """
    Deterministic data processing function to be used with caching.
    Generates multi-view tensors for every patch in the metadata.
    """
    data_list = []

    # Pre-calculate paths to avoid repeated joins
    input_dir = Config.INPUT_DIR

    print(f"Processing {len(metadata_df)} patches...")

    for idx, row in metadata_df.iterrows():
        # 1. Resolve Paths
        # Metadata paths are relative to input/
        mask_path = os.path.join(input_dir, row["mask_path"])
        label_path = os.path.join(input_dir, row["label_path"])
        volume_dir = os.path.join(input_dir, row["volume_path"])

        # 2. Load Label and Mask
        # Load as grayscale
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if label_img is None or mask_img is None:
            print(f"Warning: Missing image for row {idx}. Skipping.")
            continue

        # Crop to patch coordinates
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Helper to crop safely
        def crop_img(img, x, y, w, h):
            img_h, img_w = img.shape
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            crop = img[y:y2, x:x2]
            # Pad if necessary
            ph = h - crop.shape[0]
            pw = w - crop.shape[1]
            if ph > 0 or pw > 0:
                crop = np.pad(
                    crop, ((0, ph), (0, pw)), mode="constant", constant_values=0
                )
            return crop

        label_patch = crop_img(label_img, x, y, w, h)
        mask_patch = crop_img(mask_img, x, y, w, h)

        # Normalize labels to 0/1
        label_patch = (label_patch > 0).astype(np.float32)
        mask_patch = (mask_patch > 0).astype(np.float32)

        # 3. Generate Multi-View Tensors
        # We generate all 3 views (A, B, C) now so we can sample efficiently during training
        views_data = {}
        for view_name, start_z in Config.TRAIN_VIEWS.items():
            # generate_multiview_tensor returns (3, H, W) torch tensor
            tensor_th = generate_multiview_tensor(volume_dir, start_z, x, y, w, h)

            # Convert to numpy (H, W, 3) for storage and Albumentations
            tensor_np = tensor_th.numpy().transpose(1, 2, 0)
            views_data[view_name] = tensor_np

        # 4. Store
        data_list.append(
            {
                "fragment_id": row["fragment_id"],
                "x": x,
                "y": y,
                "views": views_data,
                "label": label_patch,
                "mask": mask_patch,
            }
        )

    return data_list


class InkDataset(Dataset):
    """
    Dataset class for Vesuvius Ink Detection.
    Implements 'Safe-Zone Multi-View Sampling' and caching.
    """

    def __init__(self, metadata_df, split="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing patch metadata.
            split (str): 'train' or 'val'. Determines augmentation and sampling strategy.
            load_cached_data (bool): Whether to use disk caching for processed tensors.
        """
        self.split = split
        self.transform = get_transforms(split)

        # Unique cache filename based on split and length to avoid collisions
        # (e.g. if we use a subset for debugging)
        cache_name = f"dataset_{split}_{len(metadata_df)}.npy"

        # Load or generate data
        # We pass the dataframe to the processing function via kwargs logic or closure
        # Here using lambda/partial is cleaner for the utility function signature
        self.data = load_or_process_data(
            file_name=cache_name,
            process_func=lambda: _process_dataset(metadata_df),
            load_cached_data=load_cached_data,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- Safe-Zone Multi-View Sampling ---
        if self.split == "train":
            # Randomly select one view (A, B, or C) to enforce translation invariance
            view_id = random.choice(list(Config.TRAIN_VIEWS.keys()))
        else:
            # For validation, deterministically use View B (Center)
            view_id = "B"

        image = item["views"][view_id]  # Shape: (H, W, 3)
        label = item["label"]  # Shape: (H, W)
        mask = item["mask"]  # Shape: (H, W)

        # --- Augmentation ---
        # Albumentations expects (H, W, C)
        augmented = self.transform(image=image, mask=label)
        image_aug = augmented["image"]
        label_aug = augmented["mask"]

        # Also transform the valid-pixel mask if geometric augs were applied
        # We can re-use the replay or just apply same transform if it's deterministic per call.
        # Since A.Compose is random, we should have passed both masks.
        # Let's redo transform call correctly to handle multiple masks.
        # Re-calling transform:
        # A.Compose supports 'image', 'mask', and 'masks' (list).

        if self.split == "train":
            # Re-apply transform logic to ensure mask and label get same geometric transform
            # We use the 'masks' argument for additional masks
            res = self.transform(image=image, masks=[label, mask])
            image_aug = res["image"]
            label_aug = res["masks"][0]
            mask_aug = res["masks"][1]
        else:
            # No random transform, just pass through
            mask_aug = mask

        # --- To Tensor ---
        # Convert Image: (H, W, 3) -> (3, H, W)
        image_tensor = torch.from_numpy(image_aug).float().permute(2, 0, 1)

        # Convert Label/Mask: (H, W) -> (1, H, W)
        label_tensor = torch.from_numpy(label_aug).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_aug).float().unsqueeze(0)

        return image_tensor, label_tensor, mask_tensor
