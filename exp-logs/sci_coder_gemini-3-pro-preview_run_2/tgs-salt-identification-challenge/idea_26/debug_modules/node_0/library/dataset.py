import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import rle_decode, pad_image


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation Task.
    Handles Train, Val, Test, and Pseudo-label modes.
    Implements caching, padding, augmentation, and depth standardization.
    """

    def __init__(
        self,
        mode,
        csv_path=None,
        depth_mean=None,
        depth_std=None,
        soft_masks=None,
        load_cached_data=True,
    ):
        """
        Args:
            mode (str): One of 'train', 'val', 'test', 'pseudo'.
            csv_path (str): Path to the metadata CSV file.
            depth_mean (float, optional): Mean for depth standardization.
            depth_std (float, optional): Std for depth standardization.
            soft_masks (dict, optional): Dictionary {id: np.ndarray} of soft masks for pseudo labeling.
            load_cached_data (bool): Whether to load data from numpy cache.
        """
        self.mode = mode
        self.soft_masks = soft_masks

        # Determine CSV path if not provided based on mode
        if csv_path is None:
            if mode == "train":
                csv_path = Config.TRAIN_CSV
            elif mode == "val":
                csv_path = Config.VAL_CSV
            elif mode == "test":
                csv_path = Config.TEST_CSV
            elif mode == "pseudo":
                # Pseudo usually combines train and test, or just test.
                # For this implementation, we expect the caller to provide the specific CSV
                # (likely the test csv or a combined one).
                # Defaulting to TEST_CSV if not provided for safety.
                csv_path = Config.TEST_CSV

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Debug mode: reduce dataset size
        if Config.DEBUG:
            self.df = self.df.head(Config.BATCH_SIZE * 2)

        # Caching Logic
        cache_prefix = f"{mode}_{len(self.df)}"
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.images = self._load_data(
            "images", cache_prefix, load_cached_data, self._load_images_from_disk
        )

        self.depths = self._load_data(
            "depths", cache_prefix, load_cached_data, self._load_depths_from_disk
        )

        self.ids = self._load_data(
            "ids", cache_prefix, load_cached_data, self._load_ids_from_disk
        )

        # Load masks only if not test mode (pseudo mode might use soft_masks passed in init)
        if mode in ["train", "val"]:
            self.masks = self._load_data(
                "masks", cache_prefix, load_cached_data, self._load_masks_from_disk
            )
        else:
            self.masks = None

        # Depth Standardization
        if depth_mean is None or depth_std is None:
            # If not provided, compute from current depths (usually done on training set)
            self.depth_mean = np.mean(self.depths)
            self.depth_std = np.std(self.depths)
        else:
            self.depth_mean = depth_mean
            self.depth_std = depth_std

        # Augmentation Pipeline
        self.transforms = self._get_transforms()

    def _load_data(self, name, prefix, load_cache, load_fn):
        """Generic caching wrapper."""
        filename = os.path.join(self.cache_dir, f"{prefix}_{name}.npy")

        if load_cache and os.path.exists(filename):
            try:
                return np.load(filename, allow_pickle=True)
            except Exception:
                pass  # Fallback to loading from disk if cache is corrupt

        data = load_fn()
        np.save(filename, data)
        return data

    def _load_images_from_disk(self):
        images = []
        for path in self.df["image_path"]:
            full_path = os.path.join(Config.INPUT_ROOT, path)
            # Load as grayscale (1 channel)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {full_path}")
            images.append(img)
        return np.array(images)

    def _load_masks_from_disk(self):
        masks = []
        for rle in self.df["rle_mask"]:
            # Decode RLE to 101x101 mask
            # Handle NaN/float RLEs
            if pd.isna(rle):
                rle = ""
            mask = rle_decode(str(rle), shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH))
            masks.append(mask)
        return np.array(masks)

    def _load_depths_from_disk(self):
        return self.df["z"].values.astype(np.float32)

    def _load_ids_from_disk(self):
        return self.df["id"].values

    def _get_transforms(self):
        # Calculate 1-channel mean/std from ImageNet 3-channel stats
        mean_1ch = sum(Config.MEAN) / 3.0
        std_1ch = sum(Config.STD) / 3.0

        if self.mode in ["train", "pseudo"]:
            return A.Compose(
                [
                    # Elastic Transform (Mandatory per strategy)
                    A.ElasticTransform(
                        alpha=Config.ELASTIC_ALPHA,
                        sigma=Config.ELASTIC_SIGMA,
                        alpha_affine=Config.ELASTIC_ALPHA_AFFINE,
                        p=0.5,
                    ),
                    # Rigid Transform (Conservative)
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=Config.RIGID_P,
                    ),
                    # Horizontal Flip
                    A.HorizontalFlip(p=0.5),
                    # Normalize & Tensor
                    A.Normalize(
                        mean=(mean_1ch,), std=(std_1ch,), max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(
                        mean=(mean_1ch,), std=(std_1ch,), max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        image = self.images[idx]
        depth_val = self.depths[idx]
        img_id = self.ids[idx]

        # 2. Pad Image (101x101 -> 128x128)
        # We pad before augmentation to ensure consistent spatial dimensions for the network
        image_padded = pad_image(image)

        # 3. Handle Mask
        mask_padded = None

        if self.mode == "pseudo" and self.soft_masks is not None:
            # Load soft mask from dictionary
            if img_id in self.soft_masks:
                mask = self.soft_masks[img_id]
                mask_padded = pad_image(mask)
            else:
                # Fallback if ID missing (should not happen in correct pipeline)
                mask_padded = np.zeros(
                    (Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.float32
                )

        elif self.masks is not None:
            mask = self.masks[idx]
            mask_padded = pad_image(mask)

        # 4. Augmentation
        # Albumentations expects HWC, so we add a channel dimension to the grayscale image
        # image_padded shape: (128, 128) -> (128, 128, 1)
        image_aug_in = image_padded[..., np.newaxis]

        if mask_padded is not None:
            augmented = self.transforms(image=image_aug_in, mask=mask_padded)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

            # Ensure mask is (1, H, W)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.ndim == 3 and mask_tensor.shape[2] == 1:
                # ToTensorV2 usually outputs (C, H, W) for image, but mask might be (H, W)
                # If mask came out as (H, W), unsqueeze. If (1, H, W) or (H, W, 1), adjust.
                pass

            # For BCE/Lovasz, we typically want float masks.
            # If original mask was uint8 (0/1), convert to float.
            mask_tensor = mask_tensor.float()

        else:
            augmented = self.transforms(image=image_aug_in)
            image_tensor = augmented["image"]
            mask_tensor = torch.zeros((1, Config.IMG_HEIGHT, Config.IMG_WIDTH)).float()

        # 5. Depth Standardization
        depth_std_val = (depth_val - self.depth_mean) / (self.depth_std + 1e-8)
        depth_tensor = torch.tensor([depth_std_val], dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "depth": depth_tensor,
            "id": img_id,
        }
