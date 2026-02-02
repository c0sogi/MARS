import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    transforms = []

    # 1. Padding to 128x128 (Divisibility by 32)
    # Reflection padding is preferred for texture synthesis to avoid border artifacts
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0,
        )
    )

    if mode == "train":
        # 2. Mandatory Non-Rigid Augmentation (Elastic)
        # Simulates organic plasticity of salt structures
        transforms.append(
            A.ElasticTransform(
                alpha=Config.ELASTIC_ALPHA,
                sigma=Config.ELASTIC_SIGMA,
                alpha_affine=None,
                p=Config.ELASTIC_PROB,
            )
        )

        # 3. Rigid Augmentation (ShiftScaleRotate)
        # Conservative rigid transforms to preserve boundary precision
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=Config.RIGID_AUG_PROB,
            )
        )

        # 4. Standard Flips
        transforms.append(A.HorizontalFlip(p=0.5))

    # 5. Normalization
    # Normalize pixel values to [0, 1] range based on dataset stats
    transforms.append(
        A.Normalize(
            mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD, max_pixel_value=255.0, p=1.0
        )
    )

    # 6. ToTensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    """
    Standard dataset for Labeled Training and Validation data.
    Handles loading, caching, and preprocessing of images, masks, and depths.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train' or 'val'.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Select metadata file
        if mode == "train":
            self.meta_path = Config.TRAIN_METADATA
        elif mode == "val":
            self.meta_path = Config.VAL_METADATA
        else:
            raise ValueError(f"Invalid mode for SaltDataset: {mode}")

        # Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        self.df = pd.read_csv(self.meta_path)

        # Cache paths
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.images_cache_path = os.path.join(self.cache_dir, f"{mode}_images.npy")
        self.masks_cache_path = os.path.join(self.cache_dir, f"{mode}_masks.npy")
        self.depths_cache_path = os.path.join(self.cache_dir, f"{mode}_depths.npy")
        self.ids_cache_path = os.path.join(self.cache_dir, f"{mode}_ids.npy")

        # Load Data
        self._load_data()

        # Transforms
        self.transforms = get_transforms(mode=mode)

    def _load_data(self):
        """
        Handles caching logic for images, masks, depths, and IDs.
        """
        # Check if cache exists
        cache_exists = (
            os.path.exists(self.images_cache_path)
            and os.path.exists(self.masks_cache_path)
            and os.path.exists(self.depths_cache_path)
            and os.path.exists(self.ids_cache_path)
        )

        if self.load_cached_data and cache_exists:
            try:
                # print(f"Loading cached {self.mode} data from {self.cache_dir}...")
                self.images = np.load(self.images_cache_path)
                self.masks = np.load(self.masks_cache_path)
                self.depths = np.load(self.depths_cache_path)
                self.ids = np.load(self.ids_cache_path, allow_pickle=True)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        # print(f"Processing {self.mode} data from scratch...")

        images_list = []
        masks_list = []
        depths_list = []
        ids_list = []

        for idx, row in self.df.iterrows():
            # Load Image
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Load Mask
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            # Binarize mask (standardize to 0-1)
            mask = (mask > 127).astype(np.uint8)

            # Depth
            z = row["z"]

            # ID
            img_id = row["id"]

            images_list.append(img)
            masks_list.append(mask)
            depths_list.append(z)
            ids_list.append(img_id)

        # Convert to numpy arrays
        self.images = np.array(images_list, dtype=np.uint8)  # (N, H, W)
        self.masks = np.array(masks_list, dtype=np.uint8)  # (N, H, W)
        self.depths = np.array(depths_list, dtype=np.float32)
        self.ids = np.array(ids_list)

        # Save to cache
        np.save(self.images_cache_path, self.images)
        np.save(self.masks_cache_path, self.masks)
        np.save(self.depths_cache_path, self.depths)
        np.save(self.ids_cache_path, self.ids)
        # print(f"Saved {self.mode} data to cache.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W)
        mask = self.masks[idx]  # (H, W)
        depth = self.depths[idx]  # Scalar
        img_id = self.ids[idx]

        # Apply transforms
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Ensure channel dimension for image: (H, W) -> (1, H, W)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Ensure channel dimension for mask: (H, W) -> (1, H, W)
        # Convert to float for loss compatibility
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

        # Normalize Depth: (z - mean) / std
        depth_norm = (depth - Config.DEPTH_MEAN) / Config.DEPTH_STD
        depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "depth": depth_tensor,
            "id": img_id,
        }


class PseudoLabelDataset(Dataset):
    """
    Dataset for Unlabeled Test Data, optionally with Soft Pseudo-Labels.
    Used for Stage 2 (Inference) and Stage 3 (Noisy Student Training).
    """

    def __init__(self, soft_labels=None, load_cached_data=True):
        """
        Args:
            soft_labels (dict, optional): Dictionary mapping ID -> Soft Mask (np.ndarray).
                                          If None, acts as inference dataset.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.soft_labels = soft_labels
        self.load_cached_data = load_cached_data
        self.mode = "test"

        # Metadata
        self.meta_path = Config.TEST_METADATA
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        self.df = pd.read_csv(self.meta_path)

        # Cache paths
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.images_cache_path = os.path.join(self.cache_dir, "test_images.npy")
        self.depths_cache_path = os.path.join(self.cache_dir, "test_depths.npy")
        self.ids_cache_path = os.path.join(self.cache_dir, "test_ids.npy")

        # Load Data
        self._load_data()

        # Transforms
        # If soft_labels are present, we are in Stage 3 (Training) -> Use Train Transforms
        # If soft_labels are None, we are in Stage 2 (Inference) -> Use Test Transforms
        transform_mode = "train" if self.soft_labels is not None else "test"
        self.transforms = get_transforms(mode=transform_mode)

    def _load_data(self):
        """
        Handles caching logic for test images, depths, and IDs.
        """
        cache_exists = (
            os.path.exists(self.images_cache_path)
            and os.path.exists(self.depths_cache_path)
            and os.path.exists(self.ids_cache_path)
        )

        if self.load_cached_data and cache_exists:
            try:
                # print(f"Loading cached test data from {self.cache_dir}...")
                self.images = np.load(self.images_cache_path)
                self.depths = np.load(self.depths_cache_path)
                self.ids = np.load(self.ids_cache_path, allow_pickle=True)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # print("Processing test data from scratch...")
        images_list = []
        depths_list = []
        ids_list = []

        for idx, row in self.df.iterrows():
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            z = row["z"]
            img_id = row["id"]

            images_list.append(img)
            depths_list.append(z)
            ids_list.append(img_id)

        self.images = np.array(images_list, dtype=np.uint8)
        self.depths = np.array(depths_list, dtype=np.float32)
        self.ids = np.array(ids_list)

        np.save(self.images_cache_path, self.images)
        np.save(self.depths_cache_path, self.depths)
        np.save(self.ids_cache_path, self.ids)
        # print("Saved test data to cache.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img_id = self.ids[idx]

        # Determine Target
        if self.soft_labels is not None and img_id in self.soft_labels:
            # Load soft mask (assumed to be same spatial dims as image)
            mask = self.soft_labels[img_id]
        else:
            # Dummy mask for inference (zeros)
            mask = np.zeros_like(image, dtype=np.float32)

        # Augmentation
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Ensure channel dims
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Mask tensor (soft labels are float)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Depth Injection Strategy:
        # For Pseudo-Labeling/Inference, we use the "Generalist" mode.
        # We inject the Mean Depth (which normalizes to 0) to force the model to rely on texture.
        depth_val = Config.DEPTH_MEAN
        depth_norm = (depth_val - Config.DEPTH_MEAN) / Config.DEPTH_STD  # Result is 0.0
        depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,  # Soft labels (float) or dummy zeros
            "depth": depth_tensor,
            "id": img_id,
        }
