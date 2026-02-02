import os
import io
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library import config
from library import utils


class ProductDataset(Dataset):
    def __init__(self, subset, transform=None, debug_size=None):
        """
        Args:
            subset (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            debug_size (int, optional): If provided, limits the dataset size for debugging.
        """
        self.subset = subset
        self.transform = transform

        # Select Metadata File
        if subset == "train":
            self.meta_path = config.TRAIN_METADATA
        elif subset == "val":
            self.meta_path = config.VAL_METADATA
        elif subset == "test":
            self.meta_path = config.TEST_METADATA
        else:
            raise ValueError("Subset must be 'train', 'val', or 'test'")

        # Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        if debug_size is not None:
            self.df = self.df.iloc[:debug_size].reset_index(drop=True)

        # Load Hierarchy Mappings
        self.mappings = config.get_hierarchy_mappings()
        self.cat_to_idx = self.mappings["cat_to_idx"]
        self.idx_to_l1 = self.mappings["idx_to_l1"]
        self.idx_to_l2 = self.mappings["idx_to_l2"]

        # Pre-process labels for Train/Val
        if subset != "test":
            # Map category_id to class_idx (Level 3 target)
            # We use map and fillna to handle potential mismatches safely, though metadata should be clean
            self.df["class_idx"] = self.df["category_id"].map(self.cat_to_idx)

            # Drop rows with invalid categories (should be 0)
            if self.df["class_idx"].isnull().any():
                self.df = self.df.dropna(subset=["class_idx"])
                self.df["class_idx"] = self.df["class_idx"].astype(int)

            # Extract labels as numpy arrays for fast indexing
            self.labels_l3 = self.df["class_idx"].values.astype(np.int64)

            # Map L3 index to L1 and L2 indices
            # Using list comprehension with the lookup dicts is efficient enough for init
            self.labels_l1 = np.array(
                [self.idx_to_l1[x] for x in self.labels_l3], dtype=np.int64
            )
            self.labels_l2 = np.array(
                [self.idx_to_l2[x] for x in self.labels_l3], dtype=np.int64
            )

        # Store input directory base path
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Retrieve Image Data
        # We construct the full path. Metadata contains relative path (e.g., "train.bson")
        bson_path = os.path.join(self.input_dir, row["bson_file_path"])
        offset = row["bson_offset"]

        # Open file, seek, and extract.
        # Opening inside __getitem__ relies on OS page cache for performance.
        try:
            with open(bson_path, "rb") as f:
                img_bytes_list = utils.extract_images_from_bson(f, offset)
        except Exception as e:
            # Fallback for corrupt read: return empty list, handled below
            img_bytes_list = []

        # Decode Images
        images = []
        for b_data in img_bytes_list:
            try:
                img = Image.open(io.BytesIO(b_data)).convert("RGB")
                images.append(img)
            except Exception:
                continue

        # Handle Missing Images (Fallback to Black Image)
        if len(images) == 0:
            images = [Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))]

        # 2. Pad / Normalize Image Count
        # We enforce a fixed size of 4 images per product for batching.
        # We cycle the existing images to fill the slots.
        # e.g., [A, B] -> [A, B, A, B]. This is robust for Global Max Pooling.
        num_imgs = len(images)
        if num_imgs < 4:
            images = (images * 4)[:4]
        else:
            images = images[:4]

        # 3. Apply Transforms
        t_images = []
        if self.transform:
            for img in images:
                t_images.append(self.transform(img))
        else:
            default_t = transforms.ToTensor()
            for img in images:
                t_images.append(default_t(img))

        # Stack into (4, 3, H, W)
        img_tensor = torch.stack(t_images)

        # 4. Return Data
        if self.subset == "test":
            sample_id = row["sample_id"]
            return img_tensor, sample_id
        else:
            l1 = self.labels_l1[idx]
            l2 = self.labels_l2[idx]
            l3 = self.labels_l3[idx]
            return img_tensor, (l1, l2, l3)


def get_dataloaders(debug_size=None):
    """
    Creates and returns the training and validation DataLoaders.
    """
    # Standard ImageNet Normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training Transforms: Resize -> Flip -> Tensor -> Normalize
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Validation Transforms: Resize -> Tensor -> Normalize
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Create Datasets
    train_ds = ProductDataset(
        subset="train", transform=train_transform, debug_size=debug_size
    )
    val_ds = ProductDataset(
        subset="val", transform=val_transform, debug_size=debug_size
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates and returns the test DataLoader.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_ds = ProductDataset(subset="test", transform=test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
