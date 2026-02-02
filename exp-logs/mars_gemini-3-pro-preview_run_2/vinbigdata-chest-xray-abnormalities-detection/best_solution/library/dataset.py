import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

# Optional imports for robust header reading
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from skimage import io as skimage_io

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

logger = get_logger("dataset")


class VinBigDataset(Dataset):
    """
    PyTorch Dataset for VinBigData Chest X-ray Abnormalities Detection.

    Handles:
    - Loading pre-processed PNG images (cached).
    - Reading original DICOM dimensions for bbox scaling via rasterio.
    - Parsing annotations and handling 'No finding' class.
    - Applying augmentations via Albumentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, class_id, bbox).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Unique images
        self.image_ids = self.df["image_id"].unique()

        # Pre-group annotations for faster access during training
        if self.mode in ["train", "val"]:
            self.annotations = self.df.groupby("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # --- 1. Determine File Paths ---
        # Get the path to the cached PNG
        if self.mode in ["train", "val"]:
            records = self.annotations.get_group(image_id)
            file_path = records.iloc[0]["file_path"]
        else:
            # Test mode
            file_path = self.df[self.df["image_id"] == image_id].iloc[0]["file_path"]

        # --- 2. Load Image ---
        # Load PNG (uint8)
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- 3. Get Original Dimensions ---
        # We need original dimensions to calculate scaling factors because
        # the cached images are resized to Config.IMG_SIZE (512), but bboxes are in original coords.

        # Construct path to original DICOM
        orig_path_train = os.path.join(Config.TRAIN_DICOM_DIR, f"{image_id}.dicom")
        orig_path_test = os.path.join(Config.TEST_DICOM_DIR, f"{image_id}.dicom")

        if os.path.exists(orig_path_train):
            orig_path = orig_path_train
        elif os.path.exists(orig_path_test):
            orig_path = orig_path_test
        else:
            # Should not happen if metadata is correct
            raise FileNotFoundError(f"Original DICOM not found for {image_id}")

        # Read metadata (fast if possible)
        orig_h, orig_w = None, None

        # Strategy 1: pydicom (Header only - fast)
        if HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(orig_path, stop_before_pixels=True)
                orig_h, orig_w = ds.Rows, ds.Columns
            except Exception:
                pass

        # Strategy 2: rasterio
        if orig_h is None:
            try:
                with rasterio.open(orig_path) as src:
                    orig_h, orig_w = src.height, src.width
            except Exception:
                pass

        # Strategy 3: skimage (Slow - reads full image)
        if orig_h is None and HAS_SKIMAGE:
            try:
                # Only read if absolutely necessary
                full_img = skimage_io.imread(orig_path)
                orig_h, orig_w = full_img.shape[:2]
            except Exception:
                pass

        # Fallback: Use current image size (Cite debug_lesson_1: Ensure Fallback Data Dimensions Match Metadata)
        # Note: If we hit this, bboxes will be scaled by 1.0. If original was larger,
        # boxes will be out of bounds. We rely on clipping/filtering downstream to handle this safely.
        if orig_h is None:
            logger.warning(
                f"Could not read original dims for {image_id}. Using current dims (Boxes may be wrong)."
            )
            orig_h, orig_w = image.shape[:2]

        current_h, current_w = image.shape[:2]

        scale_w = current_w / orig_w
        scale_h = current_h / orig_h

        # --- 4. Process Targets ---
        boxes = []
        labels = []

        if self.mode in ["train", "val"]:
            # Filter out 'No finding' (Class 14)
            valid_records = records[records["class_id"] != 14]

            if len(valid_records) > 0:
                # Extract raw boxes
                raw_boxes = valid_records[
                    ["x_min", "y_min", "x_max", "y_max"]
                ].values.astype(np.float32)

                # Scale boxes
                raw_boxes[:, 0] *= scale_w
                raw_boxes[:, 2] *= scale_w
                raw_boxes[:, 1] *= scale_h
                raw_boxes[:, 3] *= scale_h

                # Clip boxes to image boundaries
                raw_boxes[:, 0] = np.clip(raw_boxes[:, 0], 0, current_w)
                raw_boxes[:, 2] = np.clip(raw_boxes[:, 2], 0, current_w)
                raw_boxes[:, 1] = np.clip(raw_boxes[:, 1], 0, current_h)
                raw_boxes[:, 3] = np.clip(raw_boxes[:, 3], 0, current_h)

                # Filter invalid boxes (area <= 0)
                keep = (raw_boxes[:, 2] > raw_boxes[:, 0]) & (
                    raw_boxes[:, 3] > raw_boxes[:, 1]
                )

                if keep.any():
                    boxes = raw_boxes[keep]
                    # Map class IDs: Dataset 0-13 -> Model 1-14
                    cls_ids = valid_records["class_id"].values[keep]
                    labels = cls_ids + 1

        # --- 5. Apply Transforms ---
        if self.transforms:
            if len(boxes) > 0:
                # Albumentations requires boxes as list
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["labels"], dtype=torch.int64)
            else:
                # Transform image only
                transformed = self.transforms(image=image, bboxes=[], labels=[])
                image = transformed["image"]
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Just convert image to tensor if no transforms provided (fallback)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # --- 6. Construct Target Dict ---
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Calculate area (needed for evaluation)
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # Metadata for inference/submission
        target["scale_x"] = scale_w
        target["scale_y"] = scale_h
        target["orig_w"] = orig_w
        target["orig_h"] = orig_h
        target["img_id_str"] = image_id

        return image, target


def get_transforms(data_split="train"):
    """
    Returns Albumentations transforms for train or validation/test.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # ShiftScaleRotate helps with slight misalignments
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.2
                ),
                # Cite solution_lesson_node_00006: CoarseDropout regularization
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2
                ),
                A.Normalize(
                    mean=Config.NORM_MEAN, std=Config.NORM_STD, max_pixel_value=255.0
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=Config.NORM_MEAN, std=Config.NORM_STD, max_pixel_value=255.0
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Stacks images into a single tensor, but keeps targets as a list of dictionaries.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)
