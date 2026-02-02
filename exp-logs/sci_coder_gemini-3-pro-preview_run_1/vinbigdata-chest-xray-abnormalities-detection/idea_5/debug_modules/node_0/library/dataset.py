import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.dicom_loader import get_image_tensor

# =========================================================================
# CenterNet Helper Functions
# =========================================================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the object size.
    Derived from: CornerNet (https://arxiv.org/abs/1808.01244)
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a Gaussian distribution on the heatmap at the specified center.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# =========================================================================
# Dataset Class
# =========================================================================


class VinBigDataset(Dataset):
    """
    Dataset class for VinBigData Chest X-ray Abnormalities Detection.
    Prepares data for a CenterNet-based architecture.
    """

    def __init__(self, csv_path, mode="train", load_cached_data=True, transforms=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy image files.
            transforms (albumentations.Compose): Custom transforms (optional).
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.df = pd.read_csv(csv_path)

        # Group by image_id to handle multiple findings per image
        if mode != "test":
            self.image_ids = self.df["image_id"].unique()
            self.group_df = self.df.groupby("image_id")
        else:
            self.image_ids = self.df["image_id"].unique()
            # For test, we might not have grouping if it's sample submission,
            # but structure is same.
            self.group_df = self.df.groupby("image_id")

        # Output stride for CenterNet (Input 640 -> Output 160 => Stride 4)
        self.output_stride = 4
        self.output_size = Config.IMG_SIZE // self.output_stride
        self.num_classes = Config.NUM_CLASSES

        # Define Augmentations
        if transforms:
            self.transforms = transforms
        else:
            if mode == "train":
                self.transforms = A.Compose(
                    [
                        # Geometric Augmentations
                        A.ShiftScaleRotate(
                            shift_limit=0.0625,
                            scale_limit=0.1,
                            rotate_limit=10,
                            p=0.5,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                        ),
                        # Photometric Augmentations (No CLAHE)
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ],
                    bbox_params=A.BboxParams(
                        format="pascal_voc",
                        label_fields=["class_labels"],
                        min_visibility=Config.MIN_VISIBILITY,
                    ),
                )
            else:
                # Validation/Test: Resize (handled by loader/resize logic) + Normalize
                # Note: The loader returns the image. We usually resize it to Config.IMG_SIZE.
                # Since we need to resize image AND boxes, we use Albumentations Resize here.
                self.transforms = A.Compose(
                    [
                        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ],
                    bbox_params=A.BboxParams(
                        format="pascal_voc", label_fields=["class_labels"]
                    ),
                )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get all annotations for this image
        if image_id in self.group_df.groups:
            rows = self.group_df.get_group(image_id)
        else:
            # Fallback for test if ID not found (shouldn't happen with correct meta)
            rows = pd.DataFrame()

        # 1. Load Image
        # file_path in metadata is like "train/id.dicom"
        # We need to prepend INPUT_DIR
        rel_path = rows.iloc[0]["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # get_image_tensor returns (img, h, w)
        # img is HxWx3 uint8
        img, orig_h, orig_w = get_image_tensor(
            full_path,
            cache_dir=Config.CACHE_DIR,
            load_cached_data=self.load_cached_data,
        )

        # 2. Prepare Boxes and Labels
        bboxes = []
        labels = []

        # Global label: 1 if Findings present (class != 14), 0 if No Finding (class == 14)
        # Default to 0 (No Finding)
        has_finding = 0.0

        if self.mode != "test":
            for _, row in rows.iterrows():
                cls_id = int(row["class_id"])

                if cls_id == 14:
                    # No finding
                    continue

                has_finding = 1.0

                # Extract box
                x_min = float(row["x_min"])
                y_min = float(row["y_min"])
                x_max = float(row["x_max"])
                y_max = float(row["y_max"])

                # Clip boxes to image dimensions to avoid errors
                x_min = max(0, min(x_min, orig_w - 1))
                y_min = max(0, min(y_min, orig_h - 1))
                x_max = max(x_min + 1, min(x_max, orig_w))
                y_max = max(y_min + 1, min(y_max, orig_h))

                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(cls_id)
        else:
            # Test mode: dummy boxes to satisfy Albumentations
            # We will ignore targets anyway
            bboxes = [[0, 0, 1, 1]]
            labels = [14]

        # 3. Apply Augmentations
        # If train mode, we might resize via RandomCrop or Resize if not handled.
        # The provided config assumes fixed IMG_SIZE.
        # We need to ensure the input image is resized to IMG_SIZE.
        # For training, ShiftScaleRotate handles geometry, but we should Resize first
        # or rely on the transform pipeline.
        # To be safe and consistent, we prepend Resize to the pipeline if the image isn't 640x640.
        # However, A.Compose is defined in __init__.
        # Let's assume the input image to transforms needs to be resized to IMG_SIZE
        # if the transform chain doesn't explicitly do it.
        # My train transform has ShiftScaleRotate but no Resize.
        # I will add Resize to the beginning of the chain in __init__?
        # No, ShiftScaleRotate doesn't resize the canvas necessarily.
        # Let's modify the pipeline logic slightly:
        # We will resize the image to IMG_SIZE before passing to augmentations
        # to ensure consistent tensor size.

        # Manual Resize before augmentation to ensure fixed input size
        # Albumentations Resize handles boxes correctly.
        pre_resize = A.Compose(
            [A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        transformed = pre_resize(image=img, bboxes=bboxes, labels=labels)
        img_resized = transformed["image"]
        bboxes_resized = transformed["bboxes"]
        labels_resized = transformed["labels"]

        # Apply main augmentations
        augmented = self.transforms(
            image=img_resized, bboxes=bboxes_resized, class_labels=labels_resized
        )
        image_tensor = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        # 4. Generate CenterNet Targets
        # Heatmap: [C, H/4, W/4]
        # Size: [2, H/4, W/4]
        # Offset: [2, H/4, W/4]
        # Mask: [H/4, W/4] (Used to mask loss calculation)

        hm = np.zeros(
            (self.num_classes, self.output_size, self.output_size), dtype=np.float32
        )
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((self.output_size, self.output_size), dtype=np.float32)

        if self.mode != "test":
            for box, cls_id in zip(aug_bboxes, aug_labels):
                x1, y1, x2, y2 = box

                # Map to output feature map scale
                x1 = x1 / self.output_stride
                y1 = y1 / self.output_stride
                x2 = x2 / self.output_stride
                y2 = y2 / self.output_stride

                h, w = y2 - y1, x2 - x1

                if h > 0 and w > 0:
                    radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                    radius = max(0, int(radius))

                    # Center coordinates
                    ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # Check bounds
                    if (
                        ct_int[0] >= 0
                        and ct_int[0] < self.output_size
                        and ct_int[1] >= 0
                        and ct_int[1] < self.output_size
                    ):

                        # Draw Gaussian on heatmap for the specific class
                        draw_gaussian(hm[cls_id], ct_int, radius)

                        # Regression targets
                        wh[0, ct_int[1], ct_int[0]] = w
                        wh[1, ct_int[1], ct_int[0]] = h

                        reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                        reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]

                        reg_mask[ct_int[1], ct_int[0]] = 1

        # Convert targets to tensors
        target_heatmap = torch.from_numpy(hm)
        target_size = torch.from_numpy(wh)
        target_offset = torch.from_numpy(reg)
        target_mask = torch.from_numpy(reg_mask)
        global_label = torch.tensor([has_finding], dtype=torch.float32)

        # Original dimensions for post-processing
        # We need this to map predictions back to the original image size
        original_dims = torch.tensor([orig_h, orig_w], dtype=torch.float32)

        return {
            "image": image_tensor,
            "target_heatmap": target_heatmap,
            "target_size": target_size,
            "target_offset": target_offset,
            "target_mask": target_mask,
            "global_label": global_label,
            "original_dims": original_dims,
            "image_id": image_id,
        }
