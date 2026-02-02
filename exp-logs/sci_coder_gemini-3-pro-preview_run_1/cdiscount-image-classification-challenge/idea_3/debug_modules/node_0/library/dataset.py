import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

from library.config import Config
from library.utils import extract_images_from_bson, load_category_hierarchy


# ==========================================
# Transforms
# ==========================================
def get_transforms(is_train=True):
    """
    Returns the image transformations.
    """
    # Base transforms
    t_list = [
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        # ToTensor is handled inside extract_images_from_bson or immediately after
        # But extract_images_from_bson returns tensors if we pass a transform,
        # or we can pass a composition here.
        # library.utils.extract_images_from_bson uses TF.to_tensor(img) internally if no transform passed.
        # However, we want to apply Normalize.
    ]

    # We will apply these transforms to the PIL image inside the extraction function
    # or apply them to the tensor afterwards.
    # The utils function signature: extract_images_from_bson(buffer, transform=None)
    # If transform is None, it returns ToTensor().

    # Let's define a composite transform that takes a PIL image and returns a Tensor
    transform_steps = [
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(transform_steps)


# ==========================================
# Dataset
# ==========================================
class ProductDataset(Dataset):
    def __init__(self, metadata_path, bson_path, is_test=False, transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            bson_path (str): Path to the BSON file.
            is_test (bool): Whether this is the test set (no labels).
            transform (callable, optional): Transform to apply to images.
        """
        self.metadata_path = metadata_path
        self.bson_path = bson_path
        self.is_test = is_test
        self.transform = transform

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Debugging: Reduce dataset size
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

        # Hierarchy Mapping (for Train/Val)
        self.hierarchy_map = None
        if not self.is_test:
            # Load hierarchy mapping: category_id -> l1_idx, l2_idx, l3_idx
            # load_category_hierarchy returns a DF indexed by category_id
            self.hierarchy_map = load_category_hierarchy(load_cached_data=True)

            # Ensure all categories in metadata exist in hierarchy map
            # (They should, based on generation logic)

        # File handle for BSON (lazy initialization)
        self.file_handle = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Record
        row = self.df.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        sample_id = row["sample_id"]

        # 2. Lazy File Open (One handle per worker)
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        # 3. Read BSON Document
        self.file_handle.seek(offset)
        doc_bytes = self.file_handle.read(length)

        # 4. Extract Images
        # Pass the transform to be applied directly to PIL images
        images = extract_images_from_bson(doc_bytes, transform=self.transform)

        # Handle case with 0 images (rare/corrupt) -> Return black image
        if len(images) == 0:
            # Create a dummy black image tensor
            # Shape: (3, H, W)
            dummy = torch.zeros(
                (3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
            )
            images = [dummy]

        # 5. Prepare Output
        if self.is_test:
            return {"images": images, "sample_id": sample_id}  # List[Tensor]  # int
        else:
            # Retrieve Labels
            cat_id = row["category_id"]

            # Lookup hierarchical indices
            # hierarchy_map is indexed by category_id
            try:
                h_row = self.hierarchy_map.loc[cat_id]
                l1_target = h_row["l1_idx"]
                l2_target = h_row["l2_idx"]
                l3_target = h_row["l3_idx"]
            except KeyError:
                # Fallback for safety (should not happen if metadata is consistent)
                l1_target, l2_target, l3_target = 0, 0, 0

            return {
                "images": images,  # List[Tensor]
                "l1_target": l1_target,  # int
                "l2_target": l2_target,  # int
                "l3_target": l3_target,  # int
                "sample_id": sample_id,  # int
            }

    def __del__(self):
        # Close file handle if it exists
        if self.file_handle is not None:
            self.file_handle.close()


# ==========================================
# Collate Function
# ==========================================
def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.

    Args:
        batch: List of dictionaries returned by __getitem__.

    Returns:
        dict: Collated batch with flattened images and batch indices.
    """
    # Check if this is a test batch
    is_test = "l1_target" not in batch[0]

    batch_images = []
    batch_indices = []
    sample_ids = []

    if not is_test:
        l1_targets = []
        l2_targets = []
        l3_targets = []

    for i, sample in enumerate(batch):
        imgs = sample["images"]  # List of tensors
        num_imgs = len(imgs)

        # Append images to flat list
        batch_images.extend(imgs)

        # Create batch indices for these images (all belong to product 'i')
        # e.g. if product 0 has 2 images -> [0, 0]
        batch_indices.extend([i] * num_imgs)

        sample_ids.append(sample["sample_id"])

        if not is_test:
            l1_targets.append(sample["l1_target"])
            l2_targets.append(sample["l2_target"])
            l3_targets.append(sample["l3_target"])

    # Stack images into a single large tensor (N_total, C, H, W)
    batch_images_tensor = torch.stack(batch_images)

    # Convert indices to tensor
    batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)

    # Convert sample_ids
    sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)

    output = {
        "images": batch_images_tensor,
        "batch_index": batch_indices_tensor,
        "sample_ids": sample_ids_tensor,
    }

    if not is_test:
        output["l1_target"] = torch.tensor(l1_targets, dtype=torch.long)
        output["l2_target"] = torch.tensor(l2_targets, dtype=torch.long)
        output["l3_target"] = torch.tensor(l3_targets, dtype=torch.long)

    return output


# ==========================================
# DataLoader Factory
# ==========================================
def create_dataloaders():
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Transforms
    train_transform = get_transforms(is_train=True)
    eval_transform = get_transforms(is_train=False)

    # Datasets
    train_dataset = ProductDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        is_test=False,
        transform=train_transform,
    )

    val_dataset = ProductDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,  # Val is subset of train.bson
        is_test=False,
        transform=eval_transform,
    )

    test_dataset = ProductDataset(
        metadata_path=Config.TEST_METADATA,
        bson_path=Config.TEST_BSON,
        is_test=True,
        transform=eval_transform,
    )

    # DataLoaders
    # We use pin_memory=True for faster transfer to GPU
    # persistent_workers=True keeps the workers alive, avoiding repeated file opening overhead

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    return train_loader, val_loader, test_loader
