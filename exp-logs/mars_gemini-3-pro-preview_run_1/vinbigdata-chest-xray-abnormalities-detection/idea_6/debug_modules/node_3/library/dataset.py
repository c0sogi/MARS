import torch
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_image_and_dimensions


def gaussian2D(shape, sigma=1):
    """Generates a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draws a 2D Gaussian on the heatmap at the specified center."""
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


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the object size.
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


class ThoracicDataset(Dataset):
    def __init__(self, mode="train", transform=None, debug=False):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, uses a small subset of data.
        """
        self.mode = mode
        self.debug = debug

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_META_PATH)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_META_PATH)
        else:
            self.df = pd.read_csv(Config.TEST_META_PATH)

        # Debugging: Subset
        if self.debug or Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Group by Image ID for Train/Val to aggregate boxes
        if mode in ["train", "val"]:
            self.image_ids = self.df["image_id"].unique()
            self.group = self.df.groupby("image_id")
        else:
            self.image_ids = self.df["image_id"].unique()
            self.group = None

        # Setup Transforms
        if transform is None:
            self.transform = self.get_transforms(mode)
        else:
            self.transform = transform

    def get_transforms(self, mode):
        """Returns the augmentation pipeline based on the mode."""
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=Config.BRIGHTNESS_LIMIT,
                        contrast_limit=Config.CONTRAST_LIMIT,
                        p=0.5,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    min_visibility=Config.MIN_VISIBILITY,
                    label_fields=["class_labels"],
                ),
            )
        else:
            return A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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

        # 1. Determine File Path
        if self.mode in ["train", "val"]:
            # Retrieve the first row for this image to get the file path
            row = self.group.get_group(image_id).iloc[0]
            file_path = row["file_path"]
        else:
            # For test set, direct lookup
            row = self.df[self.df["image_id"] == image_id].iloc[0]
            file_path = row["file_path"]

        # 2. Load Image (Coupled Loading)
        # Returns float32 image (0-1) and original dimensions
        # Caching is handled internally by the utility
        img, orig_h, orig_w = get_image_and_dimensions(
            image_id, file_path, load_cached_data=True
        )

        # Convert to RGB (H, W) -> (H, W, 3)
        img = np.stack([img] * 3, axis=-1)

        # 3. Get Annotations (Boxes & Labels)
        boxes = []
        labels = []

        if self.mode in ["train", "val"]:
            anns = self.group.get_group(image_id)
            for _, ann in anns.iterrows():
                class_id = ann["class_id"]
                # Filter out "No finding" (Class 14) for detection targets
                if class_id != Config.NO_FINDING_CLASS_ID:
                    x_min = float(ann["x_min"])
                    y_min = float(ann["y_min"])
                    x_max = float(ann["x_max"])
                    y_max = float(ann["y_max"])

                    # Cite debug_lesson_3: Synchronize Dependent Metadata When Triggering Data Fallbacks
                    # Clamp boxes to the actual loaded image dimensions to prevent Albumentations errors
                    x_min = max(0, min(x_min, orig_w))
                    y_min = max(0, min(y_min, orig_h))
                    x_max = max(0, min(x_max, orig_w))
                    y_max = max(0, min(y_max, orig_h))

                    # Ensure valid box area
                    if (x_max > x_min) and (y_max > y_min):
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        # 4. Apply Transforms
        # Handle empty boxes gracefully for Albumentations
        if len(boxes) == 0:
            transformed = self.transform(image=img, bboxes=[], class_labels=[])
        else:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)

        img_tensor = transformed["image"]
        trans_boxes = transformed["bboxes"]
        trans_labels = transformed["class_labels"]

        # 5. Generate CenterNet Targets
        output_h = Config.IMG_SIZE // Config.DOWNSAMPLE_RATIO
        output_w = Config.IMG_SIZE // Config.DOWNSAMPLE_RATIO

        # Classes 0-13 are findings. Class 14 is "No finding".
        # Heatmap has channels for findings only.
        num_finding_classes = Config.NUM_CLASSES - 1

        hm = np.zeros((num_finding_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg_mask = np.zeros((1, output_h, output_w), dtype=np.float32)

        # Global Label Logic:
        # 1.0 if "No Finding" (Class 14 or empty), 0.0 if findings exist.
        # We base this on the transformed labels (what is visible in the crop).
        has_finding = len(trans_labels) > 0
        global_label = np.array([0.0 if has_finding else 1.0], dtype=np.float32)

        if has_finding:
            for i, box in enumerate(trans_boxes):
                if i >= Config.MAX_DETECTIONS_PER_IMAGE:
                    break

                cls_id = int(trans_labels[i])
                x_min, y_min, x_max, y_max = box

                # Scale boxes to output resolution
                x_min = x_min / Config.DOWNSAMPLE_RATIO
                y_min = y_min / Config.DOWNSAMPLE_RATIO
                x_max = x_max / Config.DOWNSAMPLE_RATIO
                y_max = y_max / Config.DOWNSAMPLE_RATIO

                h, w = y_max - y_min, x_max - x_min

                if h > 0 and w > 0:
                    radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                    radius = max(0, int(radius))

                    # Center
                    ct = np.array(
                        [(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32
                    )
                    ct_int = ct.astype(np.int32)

                    # Check bounds
                    if (
                        ct_int[0] >= 0
                        and ct_int[0] < output_w
                        and ct_int[1] >= 0
                        and ct_int[1] < output_h
                    ):
                        # Draw Gaussian on Heatmap
                        draw_umich_gaussian(hm[cls_id], ct_int, radius)

                        # Size Target (Width, Height)
                        wh[0, ct_int[1], ct_int[0]] = w
                        wh[1, ct_int[1], ct_int[0]] = h

                        # Offset Target (Discretization Error)
                        reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                        reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]

                        # Mask for regression loss
                        reg_mask[0, ct_int[1], ct_int[0]] = 1

        target = {
            "heatmap": torch.from_numpy(hm),
            "size": torch.from_numpy(wh),
            "offset": torch.from_numpy(reg),
            "mask": torch.from_numpy(reg_mask),
            "global_label": torch.from_numpy(global_label),
        }

        return img_tensor, target, image_id
