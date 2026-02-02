import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.utils import get_label_map, parse_ground_truth

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")


def get_transforms(train=True):
    """
    Returns the Albumentations transform pipeline.

    Args:
        train (bool): Whether to include training augmentations (ColorJitter).

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms_list = []

    # Resize logic: min_size=1024, max_size=2048
    # SmallestMaxSize ensures the short edge is 1024.
    # LongestMaxSize ensures the long edge does not exceed 2048.
    transforms_list.append(A.SmallestMaxSize(max_size=1024))
    transforms_list.append(A.LongestMaxSize(max_size=2048))

    if train:
        # Photometric Augmentation
        transforms_list.append(
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)
        )

    # Normalize to [0, 1] as expected by torchvision models (which then normalize with ImageNet stats internally)
    # We use mean=0, std=1, max_pixel_value=255 to simply scale 0-255 to 0-1.
    transforms_list.append(
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0)
    )
    transforms_list.append(ToTensorV2())

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class KuzushijiDataset(Dataset):
    def __init__(self, mode="train", transforms=None):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transforms (A.Compose): Albumentations transforms.
        """
        self.mode = mode
        self.transforms = transforms

        # Load Metadata
        if mode == "train":
            csv_path = os.path.join(METADATA_DIR, "train.csv")
        elif mode == "val":
            csv_path = os.path.join(METADATA_DIR, "val.csv")
        elif mode == "test":
            csv_path = os.path.join(METADATA_DIR, "test.csv")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.df = pd.read_csv(csv_path)

        # Load Label Map
        # utils.get_label_map handles caching and returns 1-based IDs (0 is background)
        self.char_to_int, self.int_to_char = get_label_map(UNICODE_MAP_PATH)

        # Create unique integer IDs for images (useful for COCO eval)
        # We map the unique string image_id to an integer
        unique_ids = self.df["image_id"].unique()
        self.image_id_map = {uid: i for i, uid in enumerate(unique_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id_str = row["image_id"]

        # Construct full image path
        # Metadata contains relative path e.g., "train_images/..."
        # We need to join with INPUT_DIR
        rel_path = row["file_path"]
        img_path = os.path.join(INPUT_DIR, rel_path)

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse Targets
        label_str = row.get("labels", "")
        boxes, labels = parse_ground_truth(label_str, self.char_to_int)

        # Clip boxes to image dimensions before transforms to prevent Albumentations errors
        if len(boxes) > 0:
            h, w = image.shape[:2]
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            # Filter invalid boxes
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

        # Apply Transforms
        if self.transforms:
            # Albumentations requires bboxes to be list of lists
            if len(boxes) > 0:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = np.array(transformed["bboxes"], dtype=np.float32)
                labels = np.array(transformed["labels"], dtype=np.int64)
            else:
                # Handle empty boxes case
                transformed = self.transforms(image=image, bboxes=[], labels=[])
                image = transformed["image"]
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty((0,), dtype=np.int64)
        else:
            # Convert to tensor if no transforms provided (fallback)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare Target Dict
        target = {}

        # Boxes: [N, 4]
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Clip boxes to image dimensions to avoid errors
            _, h, w = image.shape
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)

            # Filter out invalid boxes (area <= 0)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = torch.as_tensor(labels, dtype=torch.int64)[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels

        # Image ID
        img_int_id = self.image_id_map.get(image_id_str, idx)
        target["image_id"] = torch.tensor([img_int_id])

        # Area
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        # Iscrowd
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.uint8)

        return image, target


def collate_fn(batch):
    """
    Custom collate function for object detection.
    """
    return tuple(zip(*batch))
