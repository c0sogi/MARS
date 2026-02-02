import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_dicom_slab


class WITSNetDataset(Dataset):
    """
    Dataset for WITS-Net that implements Wide-Field Stratified Slab Sampling.
    It extracts 3 slabs (Lower, Center, Upper) per subject and treats them as
    independent instances, effectively tripling the dataset size.
    """

    def __init__(self, metadata, mode="train", transform=None):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing subject paths and labels.
            mode (str): 'train', 'val', or 'test'. Used for caching naming.
            transform (A.Compose): Albumentations transforms.
        """
        self.metadata = metadata
        self.mode = mode
        self.transform = transform

        # Define cache paths
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_images_path = os.path.join(
            self.cache_dir, f"cache_{mode}_images.npy"
        )
        self.cache_ids_path = os.path.join(self.cache_dir, f"cache_{mode}_ids.npy")
        self.cache_targets_path = os.path.join(
            self.cache_dir, f"cache_{mode}_targets.npy"
        )

        # Load data
        self.images, self.ids, self.targets = self._load_data()

    def _load_data(self):
        """
        Loads data from cache if available, otherwise processes DICOMs and creates cache.
        """
        # Check if cache exists
        cache_exists = (
            os.path.exists(self.cache_images_path)
            and os.path.exists(self.cache_ids_path)
            and os.path.exists(self.cache_targets_path)
        )

        if Config.LOAD_CACHED_DATA and cache_exists:
            # print(f"Loading cached data for {self.mode} from {self.cache_dir}...")
            images = np.load(self.cache_images_path)
            ids = np.load(self.cache_ids_path)
            targets = np.load(self.cache_targets_path)
            return images, ids, targets

        # Process data from scratch
        # print(f"Processing data for {self.mode} (Cache miss or disabled)...")
        images_list = []
        ids_list = []
        targets_list = []

        # Use only the center slice (offset 0)
        # This avoids redundancy (Cite solution_lesson_node_00018) and uses a robust
        # deterministic heuristic (Cite solution_lesson_node_00015)
        offset = 0

        for idx, row in self.metadata.iterrows():
            # For debugging, break early if needed
            if Config.DEBUG and idx >= Config.DEBUG_SUBSET_SIZE:
                break

            braTS21ID = row["BraTS21ID"]
            # Handle target: Test set might not have MGMT_value
            mgmt_value = row.get("MGMT_value", -1)

            # Construct full paths
            # Metadata paths are relative to input dir
            flair_path = os.path.join(Config.INPUT_DIR, row["flair_path"])
            t1wce_path = os.path.join(Config.INPUT_DIR, row["t1wce_path"])
            t2w_path = os.path.join(Config.INPUT_DIR, row["t2w_path"])

            # Load slabs: returns (Depth, H, W) -> (1, H, W)
            slab_flair = load_dicom_slab(flair_path, offset, depth=Config.SLAB_DEPTH)
            slab_t1wce = load_dicom_slab(t1wce_path, offset, depth=Config.SLAB_DEPTH)
            slab_t2w = load_dicom_slab(t2w_path, offset, depth=Config.SLAB_DEPTH)

            # Stack modalities along channel dimension
            # Result shape: (3, H, W) -> (1+1+1, H, W)
            combined_slab = np.concatenate([slab_flair, slab_t1wce, slab_t2w], axis=0)

            # Transpose to (H, W, C) for Albumentations and storage
            # Shape: (H, W, 3)
            combined_slab_hwc = np.transpose(combined_slab, (1, 2, 0))

            images_list.append(combined_slab_hwc)
            ids_list.append(braTS21ID)
            targets_list.append(mgmt_value)

        # Convert to numpy arrays
        images = np.array(images_list, dtype=np.float32)
        ids = np.array(ids_list, dtype=np.int32)
        targets = np.array(targets_list, dtype=np.float32)

        # Save to cache
        np.save(self.cache_images_path, images)
        np.save(self.cache_ids_path, ids)
        np.save(self.cache_targets_path, targets)

        return images, ids, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 9)
        target = self.targets[idx]
        braTS21ID = self.ids[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # If no transform provided, just convert to tensor manually
            # Transpose (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Ensure target is a tensor
        target = torch.tensor(target, dtype=torch.float32)

        return image, target, braTS21ID


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specific mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # Geometric augmentations applied to all 9 channels simultaneously
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No geometric distortion
        return A.Compose([ToTensorV2()])


def get_dataloader(
    metadata, mode="train", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create a DataLoader for WITS-Net.
    """
    transform = get_transforms(mode)
    dataset = WITSNetDataset(metadata, mode=mode, transform=transform)

    shuffle = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )
