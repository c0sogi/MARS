import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import read_dicom_binary, get_original_dimensions


# -----------------------------------------------------------------------------
# Gaussian Utilities for CenterNet
# -----------------------------------------------------------------------------
def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute gaussian radius for a bounding box.
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D gaussian on the heatmap at the given center.
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


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def get_transforms(split):
    if split == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS_LIMIT,
                    contrast_limit=Config.AUG_CONTRAST_LIMIT,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_visibility=Config.AUG_MIN_VISIBILITY,
                label_fields=["class_labels"],
            ),
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ThoraxDataset(Dataset):
    def __init__(self, df, split="train", output_stride=4):
        self.df = df
        self.split = split
        self.output_stride = output_stride
        self.image_ids = self.df["image_id"].unique()
        self.transforms = get_transforms(split)

        # Group annotations by image_id
        self.annotations = self.df.groupby("image_id")

        # Cache setup
        self.cache_dir = os.path.join(Config.CACHE_DIR, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-load original dimensions map
        self._orig_dims_map = get_original_dimensions(self.df)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 1. Load Image (H, W, 3)
        image = self._load_image(image_id)

        # 2. Get Annotations
        # Cite debug_lesson_3: Synchronize Dependent Metadata When Triggering Data Fallbacks
        if image is None:
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
            anns = pd.DataFrame(
                columns=["class_id", "x_min", "y_min", "x_max", "y_max"]
            )
        elif image_id in self.annotations.groups:
            anns = self.annotations.get_group(image_id)
        else:
            anns = pd.DataFrame(
                columns=["class_id", "x_min", "y_min", "x_max", "y_max"]
            )

        bboxes = []
        labels = []
        has_finding = False

        # Filter annotations
        for _, row in anns.iterrows():
            cid = int(row["class_id"])
            # Class 14 is 'No finding', used for global label but not detection
            if cid != 14:
                has_finding = True
                bboxes.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
                labels.append(cid)

        # Global Label: 1 if "No finding", 0 if findings present
        global_target = 1.0 if not has_finding else 0.0

        # 3. Rescale Bboxes to 512x512
        orig_w, orig_h = self._orig_dims_map.get(str(image_id), (1024, 1024))

        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            scale_x = Config.IMAGE_SIZE[1] / orig_w
            scale_y = Config.IMAGE_SIZE[0] / orig_h

            bboxes[:, [0, 2]] *= scale_x
            bboxes[:, [1, 3]] *= scale_y

            # Clip
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, Config.IMAGE_SIZE[1] - 1)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, Config.IMAGE_SIZE[0] - 1)

        # 4. Augmentations
        if self.transforms:
            try:
                augmented = self.transforms(
                    image=image, bboxes=bboxes, class_labels=labels
                )
                image = augmented["image"]
                bboxes = augmented["bboxes"]
                labels = augmented["class_labels"]
            except ValueError:
                # Fallback: keep image, drop boxes
                bboxes = []
                labels = []
                # Re-apply only image transforms
                t_img = A.Compose(
                    [
                        t
                        for t in self.transforms.transforms
                        if not isinstance(t, (A.BboxParams, A.Compose))
                    ]
                )
                # We can't easily extract just image transforms from Compose with BboxParams in one line
                # So we just manually normalize/tensorize if augmentation failed
                # But usually this doesn't happen with valid data.
                # Simplest fallback:
                image = self._load_image(image_id)  # Reload raw
                image = A.Normalize()(image=image)["image"]
                image = ToTensorV2()(image=image)["image"]

        # 5. Generate CenterNet Targets
        output_h = Config.IMAGE_SIZE[0] // self.output_stride
        output_w = Config.IMAGE_SIZE[1] // self.output_stride

        # 14 finding classes
        num_findings = 14
        hm = np.zeros((num_findings, output_h, output_w), dtype=np.float32)
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((2, output_h, output_w), dtype=np.float32)

        # Indices for sparse loss
        K = 100
        ind = np.zeros((K), dtype=np.int64)
        reg_mask = np.zeros((K), dtype=np.uint8)

        draw_ct = 0

        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox

            h, w = y_max - y_min, x_max - x_min
            if h <= 0 or w <= 0:
                continue

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            ct_int_x = int(center_x / self.output_stride)
            ct_int_y = int(center_y / self.output_stride)

            if (
                ct_int_x < 0
                or ct_int_x >= output_w
                or ct_int_y < 0
                or ct_int_y >= output_h
            ):
                continue

            cls_id = int(label)
            if cls_id >= num_findings:
                continue

            radius = gaussian_radius(
                (np.ceil(h / self.output_stride), np.ceil(w / self.output_stride))
            )
            radius = max(0, int(radius))

            draw_gaussian(hm[cls_id], (ct_int_x, ct_int_y), radius)

            wh[0, ct_int_y, ct_int_x] = w / self.output_stride
            wh[1, ct_int_y, ct_int_x] = h / self.output_stride

            reg[0, ct_int_y, ct_int_x] = (center_x / self.output_stride) - ct_int_x
            reg[1, ct_int_y, ct_int_x] = (center_y / self.output_stride) - ct_int_y

            if draw_ct < K:
                ind[draw_ct] = ct_int_y * output_w + ct_int_x
                reg_mask[draw_ct] = 1
                draw_ct += 1

        targets = {
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "global_label": torch.tensor([global_target], dtype=torch.float32),
        }

        return image, targets, image_id

    def _load_image(self, image_id):
        cache_path = os.path.join(self.cache_dir, f"{image_id}.npy")

        if Config.USE_CACHE and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except:
                pass

        # Resolve path
        if image_id in self.annotations.groups:
            rel_path = self.annotations.get_group(image_id).iloc[0]["file_path"]
        else:
            # Fallback lookup
            row = self.df[self.df["image_id"] == image_id]
            if not row.empty:
                rel_path = row.iloc[0]["file_path"]
            else:
                # Should not happen given dataset construction
                raise ValueError(f"Image ID {image_id} not found in dataframe")

        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read DICOM
        img = read_dicom_binary(full_path, fix_monochrome=Config.FIX_MONOCHROME1)

        if img is None:
            return None

        # Resize to 512x512
        img = cv2.resize(
            img,
            (Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0]),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize to 8-bit [0, 255] for consistency
        if img.max() > 0:
            img = (img.astype(np.float32) / img.max() * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Convert to RGB (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if Config.USE_CACHE:
            np.save(cache_path, img)

        return img


def get_dataloaders(train_df, val_df, test_df=None):
    train_ds = ThoraxDataset(train_df, split="train")
    val_ds = ThoraxDataset(val_df, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = None
    if test_df is not None:
        test_ds = ThoraxDataset(test_df, split="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
