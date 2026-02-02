import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import library.config as config
import library.utils as utils

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ProductDataset(Dataset):
    """
    PyTorch Dataset that reads product images directly from BSON files using
    a metadata lookup table (CSV). It handles variable image counts by padding
    to a fixed size and retrieves hierarchical labels.
    """

    def __init__(
        self, metadata, bson_path, hierarchy_mapper=None, transform=None, is_test=False
    ):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing 'bson_offset', 'bson_length', etc.
            bson_path (str): Path to the binary BSON file.
            hierarchy_mapper (HierarchyMapper): Utility to map category_ids to L1/L2/L3 indices.
            transform (callable, optional): Image transformations.
            is_test (bool): If True, returns (images, sample_id). If False, returns (images, l1, l2, l3).
        """
        self.metadata = metadata
        self.bson_path = bson_path
        self.hierarchy_mapper = hierarchy_mapper
        self.transform = transform
        self.is_test = is_test

        # File handle is initialized lazily per worker to support multiprocessing
        self.bson_file = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Open file if not already open (one handle per worker)
        if self.bson_file is None:
            self.bson_file = open(self.bson_path, "rb")

        # Retrieve record metadata
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read images from BSON
        try:
            images = utils.read_bson_images(self.bson_file, offset, length)
        except Exception:
            # Fallback for potentially corrupt records
            images = []

        # Handle edge case: no images found (create black image)
        if not images:
            images = [Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))]

        # Apply transformations to each image
        if self.transform:
            images = [self.transform(img.convert("RGB")) for img in images]

        # Pad or truncate to fixed size of 4 images per product
        # This allows stacking into a single tensor (4, C, H, W) for batching
        target_count = 4
        current_count = len(images)

        if current_count < target_count:
            # Pad by repeating existing images
            # e.g., [Img1, Img2] -> [Img1, Img2, Img1, Img2]
            for i in range(target_count - current_count):
                images.append(images[i % current_count])
        elif current_count > target_count:
            # Truncate
            images = images[:target_count]

        # Stack into a single tensor: (4, 3, H, W)
        images_tensor = torch.stack(images)

        if self.is_test:
            sample_id = row["sample_id"]
            return images_tensor, sample_id
        else:
            category_id = row["category_id"]

            # Retrieve hierarchical labels
            labels = self.hierarchy_mapper.get_labels(category_id)
            if labels is None:
                # Should not happen given valid metadata, but safe fallback
                l1, l2, l3 = 0, 0, 0
            else:
                l1 = labels["l1"]
                l2 = labels["l2"]
                l3 = labels["l3"]

            return images_tensor, l1, l2, l3


def get_dataloaders(
    debug=False,
    subset_size=2000,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
):
    """
    Constructs and returns DataLoaders for training, validation, and testing.

    Args:
        debug (bool): If True, loads only a small subset of data for debugging.
        subset_size (int): Number of samples to load in debug mode.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader, hierarchy_mapper)
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA)
    val_meta = pd.read_csv(config.VAL_METADATA)
    test_meta = pd.read_csv(config.TEST_METADATA)

    # Debugging: Subsample data
    if debug:
        print(f"Debug mode: limiting datasets to {subset_size} samples.")
        train_meta = train_meta.iloc[:subset_size]
        val_meta = val_meta.iloc[:subset_size]
        test_meta = test_meta.iloc[:subset_size]

    # 2. Initialize Hierarchy Mapper
    mapper = utils.HierarchyMapper(load_cached_data=True)

    # 3. Define Transforms
    # Train: Resize -> RandomFlip -> Tensor -> Normalize
    # We use ImageNet stats as we will likely use a pretrained backbone
    train_transform = T.Compose(
        [
            T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Eval: Resize -> Tensor -> Normalize
    eval_transform = T.Compose(
        [
            T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # 4. Instantiate Datasets
    train_dataset = ProductDataset(
        train_meta,
        config.TRAIN_BSON,
        hierarchy_mapper=mapper,
        transform=train_transform,
        is_test=False,
    )

    val_dataset = ProductDataset(
        val_meta,
        config.TRAIN_BSON,
        hierarchy_mapper=mapper,
        transform=eval_transform,
        is_test=False,
    )

    test_dataset = ProductDataset(
        test_meta,
        config.TEST_BSON,
        hierarchy_mapper=None,
        transform=eval_transform,
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm/Training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, mapper
