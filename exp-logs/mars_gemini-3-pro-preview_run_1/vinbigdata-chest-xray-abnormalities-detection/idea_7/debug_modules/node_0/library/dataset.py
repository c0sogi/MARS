import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import read_dicom_binary


def gaussian_radius(det_size, min_overlap=0.7):
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
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


class ThoracicDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, transform=None):
        self.split = split
        self.load_cached_data = load_cached_data

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_META_PATH)
            self.mode = "train"
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_META_PATH)
            self.mode = "val"
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_META_PATH)
            self.mode = "test"

        self.image_ids = self.df["image_id"].unique()

        # Setup Cache Dir
        self.cache_dir = os.path.join(Config.CACHE_DIR, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Transforms
        if transform is None:
            self.transforms = self.get_transforms(self.mode)
        else:
            self.transforms = transform

    def get_transforms(self, mode):
        if mode == "train":
            return A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomBrightnessContrast(p=0.5),  # No CLAHE
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=Config.MIN_VISIBILITY,
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

    def load_data(self, image_id, file_path):
        cache_path = os.path.join(self.cache_dir, f"{image_id}.npy")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                return data["image"], data["orig_h"], data["orig_w"]
            except Exception:
                pass  # Fallback to load from source

        # Load from source
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        img = read_dicom_binary(full_path, fix_monochrome=Config.INVERT_MONOCHROME1)

        # Img is (H, W, 1)
        h, w = img.shape[:2]

        # Save to cache
        if self.load_cached_data:
            np.save(cache_path, {"image": img, "orig_h": h, "orig_w": w})

        return img, h, w

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get annotations for this image
        records = self.df[self.df["image_id"] == image_id]
        file_path = records.iloc[0]["file_path"]

        # Load Image
        img, orig_h, orig_w = self.load_data(image_id, file_path)

        # Convert to 3 channels for EfficientNet
        img = np.repeat(img, 3, axis=-1)

        boxes = []
        labels = []

        # Process boxes
        if self.mode != "test":
            for _, row in records.iterrows():
                cid = row["class_id"]
                # Skip "No finding" boxes for detection targets
                if cid == Config.CLASS_ID_NO_FINDING:
                    continue

                # Box
                x_min, y_min, x_max, y_max = (
                    row["x_min"],
                    row["y_min"],
                    row["x_max"],
                    row["y_max"],
                )

                # Clip to image boundaries (safety)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(orig_w, x_max)
                y_max = min(orig_h, y_max)

                # Filter degenerate boxes
                if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(cid)

        # Augment
        transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)

        img_tensor = transformed["image"]
        boxes_aug = transformed["bboxes"]
        labels_aug = transformed["class_labels"]

        # --- Generate Targets ---
        # Output stride 4
        output_h = Config.IMG_SIZE // 4
        output_w = Config.IMG_SIZE // 4

        # Heatmap: (Num_Classes, H, W) -> (14, H, W)
        hm = np.zeros((Config.NUM_CLASSES, output_h, output_w), dtype=np.float32)

        # Regression: (2, H, W) -> Width, Height
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)

        # Offset: (2, H, W) -> x-int(x), y-int(y)
        reg = np.zeros((2, output_h, output_w), dtype=np.float32)

        # Mask: (H, W) -> 1 if object present
        reg_mask = np.zeros((output_h, output_w), dtype=np.float32)

        # Global Classification Target
        # 1 if No Finding, 0 if Finding
        global_target = 1.0  # Default to No Finding
        if self.mode != "test":
            has_finding = (records["class_id"] != Config.CLASS_ID_NO_FINDING).any()
            global_target = 0.0 if has_finding else 1.0

        if self.mode != "test":
            for box, label in zip(boxes_aug, labels_aug):
                x_min, y_min, x_max, y_max = box

                # Map to output stride
                x_min = x_min / 4
                y_min = y_min / 4
                x_max = x_max / 4
                y_max = y_max / 4

                # Center
                ct_x = (x_min + x_max) / 2
                ct_y = (y_min + y_max) / 2
                ct_x_int = int(ct_x)
                ct_y_int = int(ct_y)

                # Box Size
                b_w = x_max - x_min
                b_h = y_max - y_min

                # Bounds Check
                if (
                    ct_x_int < 0
                    or ct_x_int >= output_w
                    or ct_y_int < 0
                    or ct_y_int >= output_h
                ):
                    continue

                if b_w > 0 and b_h > 0:
                    radius = gaussian_radius((math.ceil(b_h), math.ceil(b_w)))
                    radius = max(0, int(radius))

                    # Draw Gaussian on Heatmap for this class
                    if label < Config.NUM_CLASSES:
                        draw_gaussian(hm[label], (ct_x_int, ct_y_int), radius)

                    # Regression Targets
                    wh[0, ct_y_int, ct_x_int] = b_w
                    wh[1, ct_y_int, ct_x_int] = b_h

                    # Offset Targets
                    reg[0, ct_y_int, ct_x_int] = ct_x - ct_x_int
                    reg[1, ct_y_int, ct_x_int] = ct_y - ct_y_int

                    reg_mask[ct_y_int, ct_x_int] = 1

        # Return Dict
        target = {
            "heatmap": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(reg),
            "reg_mask": torch.from_numpy(reg_mask),
            "global_label": torch.tensor(global_target, dtype=torch.float32),
        }

        return {
            "image": img_tensor,
            "target": target,
            "image_id": image_id,
            "original_dim": torch.tensor([orig_h, orig_w], dtype=torch.int32),
        }
