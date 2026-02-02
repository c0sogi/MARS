import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import read_dicom_image

# =============================================================================
# Gaussian Utilities for CenterNet Targets
# =============================================================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on object size and IoU overlap.
    Derived from CornerNet/CenterNet logic.
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


# =============================================================================
# Dataset Class
# =============================================================================


class ChestXRayDataset(Dataset):
    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to load/save cached grouped metadata.
        """
        self.mode = mode
        self.transform = transform
        self.data = self._load_data(load_cached_data)

        # CenterNet Hyperparameters
        self.num_classes = Config.NUM_CLASSES
        self.img_size = Config.IMAGE_SIZE
        self.stride = 4  # Output stride of the model
        self.output_size = self.img_size // self.stride
        self.max_objs = 100  # Max objects per image for tensor sizing

    def _load_data(self, load_cached_data):
        """
        Loads metadata and groups it by image_id.
        Caches the result to disk to speed up subsequent initializations.
        """
        cache_file = os.path.join(Config.CACHE_DIR, f"grouped_{self.mode}.npy")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True).tolist()
                return data
            except Exception:
                pass  # Fallback to processing from scratch

        # 2. Determine source file
        if self.mode == "train":
            df = pd.read_csv(Config.TRAIN_META_PATH)
        elif self.mode == "val":
            df = pd.read_csv(Config.VAL_META_PATH)
        else:
            df = pd.read_csv(Config.TEST_META_PATH)

        # 3. Group by image_id
        grouped_data = []

        if self.mode in ["train", "val"]:
            # Group by image_id to gather all boxes for an image
            # We filter for relevant columns to reduce memory usage
            # Columns: image_id, class_id, x_min, y_min, x_max, y_max, file_path

            # Ensure class_id is int
            df["class_id"] = df["class_id"].astype(int)

            groups = df.groupby("image_id")

            for img_id, group in groups:
                # Get file path from the first row
                file_path = group.iloc[0]["file_path"]

                boxes = []
                labels = []

                for _, row in group.iterrows():
                    cls_id = row["class_id"]

                    # Store all annotations.
                    # We will filter "No finding" (14) during __getitem__ for box targets,
                    # but we keep it here to correctly determine global_label.

                    # BBox format: x_min, y_min, x_max, y_max
                    b = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
                    boxes.append(b)
                    labels.append(cls_id)

                grouped_data.append(
                    {
                        "image_id": img_id,
                        "file_path": file_path,
                        "boxes": np.array(boxes, dtype=np.float32),
                        "labels": np.array(labels, dtype=np.int64),
                    }
                )
        else:
            # Test mode: One row per image in test_meta.csv
            for _, row in df.iterrows():
                grouped_data.append(
                    {
                        "image_id": row["image_id"],
                        "file_path": row["file_path"],
                        "boxes": np.zeros((0, 4), dtype=np.float32),  # No GT boxes
                        "labels": np.zeros((0,), dtype=np.int64),
                    }
                )

        # 4. Save to cache
        if load_cached_data:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            np.save(cache_file, grouped_data)

        return grouped_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(Config.INPUT_DIR, sample["file_path"])

        # 1. Load Image
        # Returns float32 image and (orig_h, orig_w)
        img, (orig_h, orig_w) = read_dicom_image(img_path)

        # Min-Max Normalize to [0, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Stack to 3 channels (H, W, 3) for EfficientNet
        img = np.stack([img, img, img], axis=-1)

        # 2. Prepare Boxes and Labels
        boxes = sample["boxes"].copy()
        labels = sample["labels"].copy()

        # Filter out "No finding" (Class 14) for box targets
        # Class 14 boxes are 1x1 placeholders. We don't want to regress them.
        valid_mask = labels != Config.NO_FINDING_CLASS_ID

        # Determine Global Label: 1 if finding exists (any class != 14), 0 otherwise
        # If valid_mask has any True, then there is a finding.
        has_finding = np.any(valid_mask)
        global_label = 1.0 if has_finding else 0.0

        target_boxes = boxes[valid_mask]
        target_labels = labels[valid_mask]

        # 3. Augmentation
        if self.transform:
            # Albumentations requires boxes to be a list
            # We pass all boxes (even if empty) to handle resizing
            if len(target_boxes) > 0:
                transformed = self.transform(
                    image=img, bboxes=target_boxes, class_labels=target_labels
                )
                img = transformed["image"]
                target_boxes = np.array(transformed["bboxes"], dtype=np.float32)
                target_labels = np.array(transformed["class_labels"], dtype=np.int64)
            else:
                # Just transform image
                transformed = self.transform(image=img, bboxes=[], class_labels=[])
                img = transformed["image"]
                target_boxes = np.array([], dtype=np.float32)
                target_labels = np.array([], dtype=np.int64)

        # 4. Generate CenterNet Targets
        hm = np.zeros(
            (self.num_classes, self.output_size, self.output_size), dtype=np.float32
        )
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = min(len(target_boxes), self.max_objs)

        for k in range(num_objs):
            bbox = target_boxes[k]
            cls_id = target_labels[k]

            # Map to output feature map size
            # bbox is xmin, ymin, xmax, ymax
            # Downsample by stride
            bbox = bbox / self.stride

            x1, y1, x2, y2 = bbox
            h, w = y2 - y1, x2 - x1

            if h > 0 and w > 0:
                radius = gaussian_radius((math_ceil(h), math_ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # Draw Gaussian on Heatmap
                # Ensure class ID is valid (0-13)
                if 0 <= cls_id < self.num_classes:
                    draw_umich_gaussian(hm[cls_id], ct_int, radius)

                # Regression Targets
                wh[k] = 1.0 * w, 1.0 * h
                reg[k] = ct - ct_int
                ind[k] = ct_int[1] * self.output_size + ct_int[0]
                reg_mask[k] = 1

        # 5. Return Dictionary
        return {
            "image": img,  # (3, H, W) tensor
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "global_label": torch.tensor([global_label], dtype=torch.float32),
            "image_id": sample["image_id"],
            "original_shape": torch.tensor(
                [orig_h, orig_w], dtype=torch.int32
            ),  # For inference rescaling
        }


def math_ceil(x):
    return int(np.ceil(x))


# =============================================================================
# Transforms
# =============================================================================


def get_train_transforms():
    return A.Compose(
        [
            # Geometric Augmentations
            # min_visibility=0.3 ensures we don't learn from boxes that are mostly cropped out
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            # Photometric Augmentations (No CLAHE as per strategy)
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            # Resize & Normalize
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.3, label_fields=["class_labels"]
        ),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


# =============================================================================
# Data Loader Factory
# =============================================================================


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.
    """

    # Train Set
    train_dataset = ChestXRayDataset(
        mode="train", transform=get_train_transforms(), load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Set
    val_dataset = ChestXRayDataset(
        mode="val", transform=get_val_transforms(), load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test Set
    test_dataset = ChestXRayDataset(
        mode="test", transform=get_val_transforms(), load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
