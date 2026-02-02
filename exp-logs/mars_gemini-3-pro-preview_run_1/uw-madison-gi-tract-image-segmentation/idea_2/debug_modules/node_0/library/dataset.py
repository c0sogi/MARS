import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def get_transforms(data="train"):
    """
    Returns albumentations transforms for training, validation, or testing.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class UWGIDataset(Dataset):
    """
    Dataset class for 2.5D MRI Segmentation.
    Loads 3 adjacent slices (z-1, z, z+1) to form a 3-channel image.
    Applies percentile normalization and augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Ensure slice is integer for arithmetic operations
        # The metadata might have it as string '0001', convert to int
        self.df["slice_int"] = self.df["slice"].astype(int)

        # Create a lookup for file paths: (case, day, slice_int) -> file_path
        # This allows O(1) retrieval of adjacent slices
        self.lookup = self.df.set_index(["case", "day", "slice_int"])[
            "file_path"
        ].to_dict()
        self.available_keys = set(self.lookup.keys())

    def __len__(self):
        return len(self.df)

    def normalize(self, img):
        """
        Applies percentile normalization (1st - 99th percentile).
        """
        min_val = np.percentile(img, Config.PERCENTILE_MIN)
        max_val = np.percentile(img, Config.PERCENTILE_MAX)

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = np.clip(img, 0, 1)
        return img.astype(np.float32)

    def load_slice_img(self, case, day, slice_idx):
        """
        Loads a single slice image, handles bit depth, and normalizes it.
        Returns None if slice does not exist.
        """
        key = (case, day, slice_idx)
        if key in self.available_keys:
            path = os.path.join(Config.INPUT_DIR, self.lookup[key])
            # Load image unchanged (preserves bit depth)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                return None

            # Convert to float32
            img = img.astype(np.float32)

            return self.normalize(img)
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case = row["case"]
        day = row["day"]
        current_slice = row["slice_int"]

        # 2.5D Strategy: Load z-1, z, z+1
        # We form a 3-channel image where channels correspond to spatial depth
        slices_to_load = [current_slice - 1, current_slice, current_slice + 1]
        imgs = []

        # Load center slice first to use as fallback
        center_img = self.load_slice_img(case, day, current_slice)
        if center_img is None:
            # Fallback for safety, though valid dataframe rows should have files
            center_img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

        for s_idx in slices_to_load:
            img = self.load_slice_img(case, day, s_idx)
            if img is None:
                # Boundary condition: replicate center slice if neighbor missing
                img = center_img
            imgs.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(imgs, axis=-1)

        # Resize to model input size (Width, Height) for cv2
        img_stack = cv2.resize(
            img_stack,
            (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        mask_stack = None
        if self.mode in ["train", "val"]:
            # Initialize mask: (H, W, Classes)
            mask_stack = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], Config.NUM_CLASSES),
                dtype=np.float32,
            )

            for i, cls_name in enumerate(Config.CLASSES):
                rle = row[cls_name]
                if isinstance(rle, str) and rle != "":
                    # Decode mask at original resolution
                    mask = rle_decode(rle, (row["height"], row["width"]))

                    # Resize mask to model input size (Nearest Neighbor to preserve binary nature)
                    mask = cv2.resize(
                        mask,
                        (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    mask_stack[:, :, i] = mask

        # Apply Augmentations
        if self.transforms:
            if mask_stack is not None:
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]
            else:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]

        # Return tensors
        # ToTensorV2 converts (H, W, C) -> (C, H, W)
        if self.mode in ["train", "val"]:
            return img_stack, mask_stack, row["id"]
        else:
            return img_stack, row["id"]
