import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    NUM_CLASSES,
)
from library.utils import get_image_data, seed_everything

# =============================================================================
# AUGMENTATION UTILS
# =============================================================================


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Geometric Augmentations with strict visibility constraint
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentations (No CLAHE as per strategy)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Resize and Normalize
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_ids"],
                min_visibility=0.3,  # Filter out boxes that are largely cropped out
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_ids"]),
        )


# =============================================================================
# TARGET GENERATION UTILS (CenterNet Style)
# =============================================================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on object size.
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


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a Gaussian blob on the heatmap at the specified center.
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
# DATASET CLASS
# =============================================================================


class ThoracicDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            cache_dir (str): Directory to save/load cached numpy images.
        """
        self.df = df
        self.mode = mode
        self.cache_dir = cache_dir
        self.transforms = get_transforms(mode)

        # Group annotations by image_id
        # For test set, we might not have annotations, but the structure holds.
        self.image_ids = self.df["image_id"].unique()
        self.grouped = self.df.groupby("image_id")

        # Output stride of the model (EfficientNetB0 + BiFPN P3-P7 -> Upsampled to P2/Stride 4)
        self.output_stride = 4
        self.output_size = IMG_SIZE // self.output_stride

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        group = self.grouped.get_group(image_id)

        # 1. Load Image
        # file_path is strictly relative to input dir in metadata
        # We need to reconstruct full path.
        # The metadata generation script stored 'file_path' like 'train/xxxx.dicom'
        rel_path = group.iloc[0]["file_path"]
        full_path = os.path.join("./input", rel_path)

        # Determine specific cache subdirectory
        if self.cache_dir:
            if "train" in rel_path:
                sub_cache = os.path.join(self.cache_dir, "train")
            elif "val" in rel_path:
                sub_cache = os.path.join(self.cache_dir, "val")
            else:
                sub_cache = os.path.join(self.cache_dir, "test")
        else:
            sub_cache = CACHE_DIR  # Fallback

        try:
            image, (orig_h, orig_w) = get_image_data(
                image_id, full_path, sub_cache, load_cached_data=True
            )
            valid_image = True
        except Exception as e:
            print(
                f"Warning: Could not load image {image_id} ({full_path}). Using fallback. Error: {e}"
            )
            # Fallback: Black image (1024x1024)
            orig_h, orig_w = 1024, 1024
            image = np.zeros((orig_h, orig_w), dtype=np.uint8)
            valid_image = False

        # Convert to RGB for Albumentations (it expects 3 channels usually)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 2. Prepare Annotations
        boxes = []
        class_ids = []

        # Class 14 is "No finding". We filter these out for detection targets.
        # But we keep track for the Global Classification Head.
        has_finding = 0

        if self.mode != "test" and valid_image:
            for _, row in group.iterrows():
                cid = row["class_id"]
                if cid != 14:
                    has_finding = 1
                    # Pascal VOC format: [x_min, y_min, x_max, y_max]
                    # Ensure coordinates are within bounds
                    x_min = max(0, row["x_min"])
                    y_min = max(0, row["y_min"])
                    x_max = min(orig_w, row["x_max"])
                    y_max = min(orig_h, row["y_max"])

                    # Basic validity check
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        class_ids.append(cid)

        # 3. Apply Augmentations
        # Albumentations requires dummy boxes if none exist to handle the pipe correctly
        if len(boxes) == 0:
            # Pass dummy box for transform, then ignore result
            # We use a dummy box [0, 0, 1, 1] with a dummy class
            transformed = self.transforms(
                image=image, bboxes=[[0, 0, 1, 1]], class_ids=[999]
            )
            image_tensor = transformed["image"]
            boxes_aug = []
            class_ids_aug = []
        else:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_ids=class_ids
            )
            image_tensor = transformed["image"]
            boxes_aug = transformed["bboxes"]
            class_ids_aug = transformed["class_ids"]

        # 4. Generate Targets (Anchor-Free)
        # Dimensions:
        # Heatmap: [NUM_CLASSES, H/4, W/4]
        # Size: [2, H/4, W/4] (Width, Height)
        # Offset: [2, H/4, W/4] (x-offset, y-offset)

        hm = np.zeros(
            (NUM_CLASSES, self.output_size, self.output_size), dtype=np.float32
        )
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)

        # Global classification target
        global_label = torch.tensor([has_finding], dtype=torch.float32)

        if self.mode != "test":
            for box, cls_id in zip(boxes_aug, class_ids_aug):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = box

                # Resize box to feature map scale
                x1_s = x1 / self.output_stride
                y1_s = y1 / self.output_stride
                x2_s = x2 / self.output_stride
                y2_s = y2 / self.output_stride

                h, w = y2_s - y1_s, x2_s - x1_s

                if h > 0 and w > 0:
                    radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                    radius = max(0, int(radius))

                    # Center
                    ct = np.array(
                        [(x1_s + x2_s) / 2, (y1_s + y2_s) / 2], dtype=np.float32
                    )
                    ct_int = ct.astype(np.int32)

                    # Bounds check
                    if (
                        ct_int[0] >= 0
                        and ct_int[0] < self.output_size
                        and ct_int[1] >= 0
                        and ct_int[1] < self.output_size
                    ):

                        # 1. Heatmap
                        draw_gaussian(hm[cls_id], ct_int, radius)

                        # 2. Size (Width, Height) - Absolute scale in resized image coords
                        # Note: We store raw width/height. Model predicts raw width/height.
                        # Some implementations use log(w), but raw is specified by strategy.
                        # Storing w, h relative to the input image size (0-640 range).
                        # Since x1, x2 are in 0-640 range (from albumentations),
                        # w, h here are in feature map range (0-160).
                        # To match "Size Head... Values [0, W]", let's store in Input Scale (0-640).
                        wh[0, ct_int[1], ct_int[0]] = x2 - x1
                        wh[1, ct_int[1], ct_int[0]] = y2 - y1

                        # 3. Offset (Discretization Error)
                        reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                        reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]

                        # 4. Mask (indicates presence of object for regression loss)
                        reg_mask[0, ct_int[1], ct_int[0]] = 1

        # Convert to tensors
        target = {
            "heatmap": torch.from_numpy(hm),
            "size": torch.from_numpy(wh),
            "offset": torch.from_numpy(reg),
            "mask": torch.from_numpy(reg_mask),
            "global_label": global_label,
        }

        # Return original shape for rescaling predictions during inference
        original_shape = torch.tensor([orig_h, orig_w], dtype=torch.int32)

        return image_tensor, target, image_id, original_shape


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================


def create_dataloaders():
    """
    Creates Train, Validation, and Test DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(SEED)

    # Load Metadata
    if os.path.exists(TRAIN_META_PATH):
        df_train = pd.read_csv(TRAIN_META_PATH)
    else:
        df_train = None

    if os.path.exists(VAL_META_PATH):
        df_val = pd.read_csv(VAL_META_PATH)
    else:
        df_val = None

    if os.path.exists(TEST_META_PATH):
        df_test = pd.read_csv(TEST_META_PATH)
    else:
        df_test = None

    # Create Datasets
    train_ds = (
        ThoracicDataset(df_train, mode="train", cache_dir=CACHE_DIR)
        if df_train is not None
        else None
    )
    val_ds = (
        ThoracicDataset(df_val, mode="val", cache_dir=CACHE_DIR)
        if df_val is not None
        else None
    )
    test_ds = (
        ThoracicDataset(df_test, mode="test", cache_dir=CACHE_DIR)
        if df_test is not None
        else None
    )

    # Create Loaders
    train_loader = (
        DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        if train_ds
        else None
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        if val_ds
        else None
    )

    test_loader = (
        DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        if test_ds
        else None
    )

    return train_loader, val_loader, test_loader
