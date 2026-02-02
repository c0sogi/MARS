import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.dicom_utils import load_image_and_metadata


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius of the Gaussian kernel for a given object size.
    Based on the CornerNet implementation.
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


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap at the specified center.
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


class VinBigDataDataset(Dataset):
    def __init__(self, split="train", debug=False, debug_size=500):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            debug (bool): If True, use a smaller subset of data.
            debug_size (int): Number of images to use in debug mode.
        """
        self.split = split
        self.debug = debug
        self.output_stride = 4  # CenterNet default
        self.num_classes = Config.NUM_CLASSES  # 14 classes (excluding "No finding")
        self.max_objs = 100  # Max objects per image for tensor padding

        # Load Metadata
        if split == "train":
            self.meta_path = Config.TRAIN_META_PATH
            self.df = pd.read_csv(self.meta_path)
        elif split == "val":
            self.meta_path = Config.VAL_META_PATH
            self.df = pd.read_csv(self.meta_path)
        elif split == "test":
            self.meta_path = Config.TEST_META_PATH
            self.df = pd.read_csv(self.meta_path)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Group by image_id to handle multiple findings per image
        self.image_ids = self.df["image_id"].unique()

        if self.debug:
            self.image_ids = self.image_ids[:debug_size]
            print(f"[{split.upper()}] Debug mode: Using {len(self.image_ids)} images.")

        # Pre-group annotations for faster access
        # Create a dictionary mapping image_id to a DataFrame of its annotations
        self.annotations = {
            img_id: group for img_id, group in self.df.groupby("image_id")
        }

        # Define Augmentations
        if split == "train":
            self.transforms = A.Compose(
                [
                    A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomBrightnessContrast(p=0.5),
                    A.CLAHE(p=0.5),
                    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.3,
                ),
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
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

        # Retrieve annotations for this image
        # If test set, annotations might be dummy or empty, handled gracefully
        if image_id in self.annotations:
            annos = self.annotations[image_id]
        else:
            # Should not happen for train/val based on metadata construction
            annos = pd.DataFrame(
                columns=["class_id", "x_min", "y_min", "x_max", "y_max", "file_path"]
            )

        # 1. Load Image
        # Get relative path from the first row of annotations
        relative_path = annos.iloc[0]["file_path"]

        # Use centralized loader (handles caching and parsing)
        # Returns HxWx3 uint8 image and original (H, W)
        image, (orig_h, orig_w) = load_image_and_metadata(
            image_id, relative_path, cache_dir=Config.CACHE_DIR, load_cached_data=True
        )

        # 2. Prepare Boxes for Augmentation
        bboxes = []
        labels = []

        # Check if we have valid annotations (not test set with placeholders)
        if "class_id" in annos.columns:
            for _, row in annos.iterrows():
                # x_min, y_min, x_max, y_max
                box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]

                # Clip boxes to image dimensions to avoid Albumentations errors
                box[0] = max(0, min(box[0], orig_w - 1))
                box[1] = max(0, min(box[1], orig_h - 1))
                box[2] = max(box[0] + 1, min(box[2], orig_w))
                box[3] = max(box[1] + 1, min(box[3], orig_h))

                bboxes.append(box)
                labels.append(row["class_id"])

        # If no boxes (e.g. pure test set without dummy rows), add a dummy for transform
        # Albumentations requires at least one box if bbox_params are set?
        # Actually, it handles empty lists fine, but we need to pass the lists.

        # 3. Apply Transforms
        transformed = self.transforms(image=image, bboxes=bboxes, class_labels=labels)
        image_tensor = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        # 4. Generate CenterNet Targets
        # Feature map dimensions
        h_feat = Config.IMG_SIZE // self.output_stride
        w_feat = Config.IMG_SIZE // self.output_stride

        # Initialize targets
        hm = np.zeros((self.num_classes, h_feat, w_feat), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # Global classification target: 1 if any finding (class 0-13), 0 if all are "No finding" (14)
        # Default to 0 (No finding)
        global_target = 0.0

        num_objs = 0

        for i, (box, label) in enumerate(zip(aug_bboxes, aug_labels)):
            label = int(label)

            # Skip "No finding" class (ID 14) for detection targets
            if label == 14:
                continue

            # If we are here, we have a finding
            global_target = 1.0

            # Coordinates in the resized image (512x512)
            x1, y1, x2, y2 = box

            # Map to feature map coordinates
            # Center coordinates
            ct_x = (x1 + x2) / 2
            ct_y = (y1 + y2) / 2

            # Integer center on feature map
            ct_x_idx = int(ct_x / self.output_stride)
            ct_y_idx = int(ct_y / self.output_stride)

            # Ensure within bounds
            ct_x_idx = np.clip(ct_x_idx, 0, w_feat - 1)
            ct_y_idx = np.clip(ct_y_idx, 0, h_feat - 1)

            # Object Size on feature map (or original scale, usually original scale / stride in implementation)
            # Here we use the scale relative to the feature map for consistency with loss
            h_obj = (y2 - y1) / self.output_stride
            w_obj = (x2 - x1) / self.output_stride

            if h_obj > 0 and w_obj > 0:
                # Gaussian Radius
                radius = gaussian_radius((math.ceil(h_obj), math.ceil(w_obj)))
                radius = max(0, int(radius))

                # Draw Gaussian on Heatmap
                # Label corresponds to channel index (0-13)
                draw_umich_gaussian(hm[label], (ct_x_idx, ct_y_idx), radius)

                # Fill regression targets
                if num_objs < self.max_objs:
                    wh[num_objs] = [w_obj, h_obj]

                    # Offset (discretization error)
                    reg[num_objs] = [
                        (ct_x / self.output_stride) - ct_x_idx,
                        (ct_y / self.output_stride) - ct_y_idx,
                    ]

                    # Index in flattened feature map
                    ind[num_objs] = ct_y_idx * w_feat + ct_x_idx

                    reg_mask[num_objs] = 1
                    num_objs += 1

        # Convert targets to tensors
        target_dict = {
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "global_target": torch.tensor(global_target, dtype=torch.float32),
            # Metadata for evaluation
            "original_shape": torch.tensor([orig_h, orig_w], dtype=torch.int32),
            "image_id": image_id,
        }

        return image_tensor, target_dict
