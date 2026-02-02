import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import read_dicom, load_dataset_dataframe


def get_transforms(split="train"):
    """
    Returns the Albumentations transforms for the specific split.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.0
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.0
            ),
        )


class ChestXRayDataset(Dataset):
    def __init__(self, split="train", transform=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Custom transforms.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.split = split
        self.df = load_dataset_dataframe(split=split, load_cached_data=load_cached_data)

        # If no transform provided, use default
        if transform is None:
            self.transform = get_transforms(split)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        # 1. Read Image
        # read_dicom returns a numpy array (H, W) in uint8 (0-255)
        img = read_dicom(file_path, fix_monochrome=True)

        # Handle read failure
        if img is None:
            if self.split == "test":
                # For test set, return a black image to ensure we generate a prediction
                img = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
                )
            else:
                # For train/val, return None to be filtered by collate_fn
                return None, None, None

        # Convert to RGB (3 channels) for model compatibility
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Prepare Boxes and Labels
        boxes = []
        labels = []

        # Test set has no labels
        if self.split != "test":
            # Get boxes from the dataframe
            # load_dataset_dataframe ensures 'boxes' is a list of dicts (or empty list)
            raw_boxes = row.get("boxes", [])
            class_id = row.get("class_id", 0)  # 0 is Negative

            # If class_id > 0, we expect boxes. If class_id == 0, boxes should be empty.
            if isinstance(raw_boxes, list) and len(raw_boxes) > 0:
                for box in raw_boxes:
                    # Parse dictionary format {'x': ..., 'y': ..., 'width': ..., 'height': ...}
                    if isinstance(box, dict):
                        x = float(box["x"])
                        y = float(box["y"])
                        w = float(box["width"])
                        h = float(box["height"])

                        x_min = x
                        y_min = y
                        x_max = x + w
                        y_max = y + h

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        # 3. Apply Transforms
        if self.transform:
            # Albumentations handles empty boxes correctly if fields are provided
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # 4. Convert to Tensor Targets
        target = {}
        # Convert boxes to FloatTensor
        # Cite debug_lesson_5: Ensure correct shape (0, 4) for empty boxes
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        # Convert labels to Int64Tensor
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        # Calculate area (useful for evaluation metrics like COCO)
        if len(boxes) > 0:
            area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
            target["area"] = area
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target["area"] = torch.as_tensor([], dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor([], dtype=torch.int64)

        return img, target, image_id
