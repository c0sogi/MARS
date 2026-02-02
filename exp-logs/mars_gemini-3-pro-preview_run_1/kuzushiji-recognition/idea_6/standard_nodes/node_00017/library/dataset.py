import os
import cv2
import numpy as np
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import (
    TRAIN_IMGS_DIR,
    TEST_IMGS_DIR,
    UNICODE_MAP_PATH,
    IMG_SIZE,
    DOWN_RATIO,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_DETECTIONS,
    WORKING_DIR,
)
from library.utils import load_and_parse_metadata

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_class_mapping(csv_path=UNICODE_MAP_PATH):
    """
    Creates a mapping from Unicode label to integer index.
    """
    df = pd.read_csv(csv_path)
    # Ensure consistent ordering
    chars = df["Unicode"].values
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_id, id_to_char


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size.
    Derived from CornerNet/CenterNet.
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
    Draws a 2D Gaussian on the heatmap.
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
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# =============================================================================
# AUGMENTATIONS
# =============================================================================


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for train or validation/test.
    """
    if split == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Resize and Pad
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
        )
    else:
        return A.Compose(
            [
                # Resize and Pad
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
        )


# =============================================================================
# DATASET CLASS
# =============================================================================


class KuzushijiDataset(Dataset):
    def __init__(self, metadata_path, split="train", debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            split (str): 'train', 'val', or 'test'.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.data = load_and_parse_metadata(metadata_path)

        if debug:
            self.data = self.data[:100]

        self.char_to_id, self.id_to_char = get_class_mapping()
        self.transforms = get_transforms(split)

        # Determine image directory based on split
        if split == "test":
            self.img_dir = TEST_IMGS_DIR
        else:
            self.img_dir = TRAIN_IMGS_DIR

        # Output grid size
        self.output_h = IMG_SIZE // DOWN_RATIO
        self.output_w = IMG_SIZE // DOWN_RATIO
        self.max_objs = MAX_DETECTIONS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item["image_id"]
        file_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (should not happen with correct metadata)
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        original_shape = (h, w)

        # Prepare boxes and labels
        bboxes = []
        labels = []

        if self.split != "test":
            for ann in item["annotations"]:
                label_str = ann["label"]
                if label_str in self.char_to_id:
                    cls_id = self.char_to_id[label_str]
                    bbox = ann["bbox"]  # [x, y, w, h]

                    # Albumentations expects [x, y, w, h] for 'coco' format
                    # Ensure bbox is within image boundaries
                    x, y, bw, bh = bbox
                    # Clip to image
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    bw = max(1, min(bw, w - x))
                    bh = max(1, min(bh, h - y))

                    bboxes.append([x, y, bw, bh])
                    labels.append(cls_id)

        # Apply Transforms
        if self.split == "test":
            # For test, we need dummy boxes to satisfy BboxParams if we use them,
            # but usually we just transform the image.
            # However, our get_transforms defines bbox_params.
            # We pass empty lists.
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            inp_image = transformed["image"]
        else:
            transformed = self.transforms(
                image=image, bboxes=bboxes, class_labels=labels
            )
            inp_image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # Initialize Targets
        # Heatmap: (1, H, W) - Class agnostic objectness
        hm = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)

        # Dense regression targets
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        cls_ids = np.zeros((self.max_objs), dtype=np.int64)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # Fill Targets
        if self.split != "test":
            num_objs = min(len(bboxes), self.max_objs)

            for k in range(num_objs):
                bbox = bboxes[k]
                cls_id = labels[k]

                x, y, w_box, h_box = bbox

                # Map to output grid
                ct_x = (x + w_box / 2) / DOWN_RATIO
                ct_y = (y + h_box / 2) / DOWN_RATIO

                ct_x_int = int(ct_x)
                ct_y_int = int(ct_y)

                # Check bounds
                if 0 <= ct_x_int < self.output_w and 0 <= ct_y_int < self.output_h:
                    radius = gaussian_radius(
                        (math_ceil(h_box / DOWN_RATIO), math_ceil(w_box / DOWN_RATIO))
                    )
                    radius = max(0, int(radius))

                    # Draw Gaussian on heatmap (channel 0)
                    draw_umich_gaussian(hm[0], (ct_x_int, ct_y_int), radius)

                    # Fill regression targets
                    wh[k] = [1.0 * w_box / DOWN_RATIO, 1.0 * h_box / DOWN_RATIO]
                    reg[k] = [ct_x - ct_x_int, ct_y - ct_y_int]
                    ind[k] = ct_y_int * self.output_w + ct_x_int
                    cls_ids[k] = cls_id
                    reg_mask[k] = 1

        # Convert to tensors
        target = {
            "hm": torch.from_numpy(hm),
            "reg": torch.from_numpy(reg),
            "wh": torch.from_numpy(wh),
            "cls_ids": torch.from_numpy(cls_ids),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
        }

        return {
            "image": inp_image,
            "target": target,
            "image_id": image_id,
            "original_shape": original_shape,
        }


def math_ceil(x):
    return int(np.ceil(x))
