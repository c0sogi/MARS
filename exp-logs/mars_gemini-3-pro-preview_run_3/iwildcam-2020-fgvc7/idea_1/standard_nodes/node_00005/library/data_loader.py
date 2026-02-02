import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import load_detector_bboxes


class CroppedSpeciesDataset(Dataset):
    """
    Dataset class that loads images, crops them to the animal bounding box,
    and applies transformations.
    """

    def __init__(self, metadata_df, bbox_dict, transform=None, is_test=False):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image metadata.
            bbox_dict (dict): Dictionary mapping image_id to [x, y, w, h] (normalized).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): If True, returns (image, image_id). If False, returns (image, label).
        """
        self.df = metadata_df
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])

        # Construct file path
        # metadata file_path is relative to input directory
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using cv2 for efficiency
        img = cv2.imread(img_path)

        if img is None:
            # Handle missing/corrupt images by creating a black placeholder
            # This ensures the dataloader doesn't crash
            img = np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop to bounding box if available
        if img_id in self.bbox_dict:
            bbox = self.bbox_dict[img_id]  # [x, y, w, h] normalized
            h_img, w_img, _ = img.shape

            # Convert normalized coordinates to pixels
            x = int(bbox[0] * w_img)
            y = int(bbox[1] * h_img)
            w = int(bbox[2] * w_img)
            h = int(bbox[3] * h_img)

            # Clip coordinates to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            # Validate crop dimensions
            if w > 0 and h > 0:
                img = img[y : y + h, x : x + w]

        # Convert numpy array to PIL Image for torchvision transforms
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_id
        else:
            # category_id is the target label
            label = int(row["category_id"])
            return img, label


def get_transforms(split="train"):
    """
    Returns the data transformations for the specified split.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        # Cite solution_lesson_node_00004: Enhancing Generalization via Advanced Augmentation
        return transforms.Compose(
            [
                transforms.Resize(Config.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Val and Test
        return transforms.Compose(
            [
                transforms.Resize(Config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    limit_data=None,
):
    """
    Creates training and validation DataLoaders.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached bounding boxes.
        limit_data (int, optional): Limit dataset size for debugging.

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # 2. Load Bounding Boxes
    bbox_dict = load_detector_bboxes(load_cached_data=load_cached_data)

    # 3. Filter Data
    # We only train on images where the detector found something significant.
    # Low confidence images are treated as "Empty" by the rule-based stage and excluded here
    # to balance the training distribution.
    train_filtered = train_df[
        train_df["max_detection_conf"] >= Config.CONF_THRESHOLD
    ].copy()
    val_filtered = val_df[val_df["max_detection_conf"] >= Config.CONF_THRESHOLD].copy()

    if limit_data:
        train_filtered = train_filtered.iloc[:limit_data]
        val_filtered = val_filtered.iloc[:limit_data]

    print(
        f"Training Data: {len(train_filtered)} samples (Filtered from {len(train_df)})"
    )
    print(f"Validation Data: {len(val_filtered)} samples (Filtered from {len(val_df)})")

    # 4. Create Datasets
    train_dataset = CroppedSpeciesDataset(
        train_filtered, bbox_dict, transform=get_transforms("train"), is_test=False
    )

    val_dataset = CroppedSpeciesDataset(
        val_filtered, bbox_dict, transform=get_transforms("val"), is_test=False
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates the test DataLoader for high-confidence images and returns a DataFrame for low-confidence ones.

    Returns:
        test_loader: DataLoader for images with max_detection_conf >= threshold.
        low_conf_df: DataFrame containing images with max_detection_conf < threshold (to be predicted as 0).
    """
    # 1. Load Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Load Bounding Boxes
    bbox_dict = load_detector_bboxes(load_cached_data=load_cached_data)

    # 3. Split Test Data
    high_conf_mask = test_df["max_detection_conf"] >= Config.CONF_THRESHOLD
    test_high_conf = test_df[high_conf_mask].copy()
    test_low_conf = test_df[~high_conf_mask].copy()

    print(f"Test Data: {len(test_df)} total.")
    print(f"  - High Confidence (Model): {len(test_high_conf)}")
    print(f"  - Low Confidence (Rule-based): {len(test_low_conf)}")

    # 4. Create Dataset for High Confidence Images
    test_dataset = CroppedSpeciesDataset(
        test_high_conf, bbox_dict, transform=get_transforms("test"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, test_low_conf
