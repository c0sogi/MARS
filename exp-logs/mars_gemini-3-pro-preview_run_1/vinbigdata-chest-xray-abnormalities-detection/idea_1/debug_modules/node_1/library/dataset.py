import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# =========================================================================
# Helper Functions for CenterNet Targets
# =========================================================================


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
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


# =========================================================================
# Dataset Class
# =========================================================================


class VinBigDataDataset(Dataset):
    def __init__(self, split="train", debug=False, load_cached_data=False):
        self.split = split
        self.debug = debug
        self.load_cached_data = load_cached_data
        self.num_classes = Config.NUM_CLASSES
        self.img_size = Config.IMG_SIZE
        self.down_ratio = Config.DOWN_RATIO
        self.max_objects = Config.MAX_OBJECTS

        # Cache directory setup
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache", split)
        if self.load_cached_data:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_META_PATH)
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_META_PATH)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_META_PATH)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Debug Mode
        if self.debug:
            unique_ids = self.df["image_id"].unique()
            sample_ids = unique_ids[: Config.DEBUG_SAMPLE_SIZE]
            self.df = self.df[self.df["image_id"].isin(sample_ids)].copy()

        # Group annotations by image_id
        self.image_ids = self.df["image_id"].unique()
        self.annotations = self.df.groupby("image_id")

        # Define Transforms
        self.transforms = A.Compose(
            [
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.5 if split == "train" else 0),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id, file_path):
        # Cache Check
        cache_path = os.path.join(self.cache_dir, f"{image_id}.npy")
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to loading from source if cache is corrupt

        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        img = None

        # Strategy 1: Pydicom
        try:
            import pydicom

            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array

            # Handle Photometric Interpretation if needed (simple inversion)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.amax(img) - img
        except (ImportError, Exception):
            pass

        # Strategy 2: OpenCV
        if img is None:
            try:
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            except Exception:
                pass

        # Fallback
        if img is None:
            # Return black image to prevent crash, though this shouldn't happen with valid data
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Normalization (Percentile)
        img = img.astype(np.float32)
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        else:
            img = img * 0  # Constant image

        # Convert to 3 channels for ResNet backbone
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save to cache if enabled
        if self.load_cached_data:
            try:
                np.save(cache_path, img)
            except Exception:
                pass

        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get annotations for this image
        try:
            anns = self.annotations.get_group(image_id)
            file_path = anns.iloc[0]["file_path"]
        except KeyError:
            # Should not happen if logic is correct
            file_path = f"{self.split}/{image_id}.dicom"
            anns = pd.DataFrame()

        # Load Image
        img = self.load_image(image_id, file_path)
        h_orig, w_orig, _ = img.shape

        # Extract Boxes and Classes
        bboxes = []
        labels = []

        if self.split != "test" and not anns.empty:
            for _, row in anns.iterrows():
                # Filter out "No finding" (Class 14) for detection targets
                if row["class_id"] == 14:
                    continue

                # Bounding box
                x_min, y_min = row["x_min"], row["y_min"]
                x_max, y_max = row["x_max"], row["y_max"]

                # Clip to image boundaries (metadata might have errors)
                # Note: We don't know original DICOM size here easily without reading it.
                # However, Albumentations handles clipping if we set check_each_transform=False or similar.
                # For safety, we rely on Albumentations to handle resizing.

                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(row["class_id"])

        # Apply Transforms
        if len(bboxes) > 0:
            transformed = self.transforms(image=img, bboxes=bboxes, class_labels=labels)
            img_tensor = transformed["image"]
            bboxes_trans = transformed["bboxes"]
            labels_trans = transformed["class_labels"]
        else:
            # No boxes (or test set)
            transformed = self.transforms(image=img, bboxes=[], class_labels=[])
            img_tensor = transformed["image"]
            bboxes_trans = []
            labels_trans = []

        # =====================================================================
        # Generate CenterNet Targets
        # =====================================================================
        output_h = self.img_size // self.down_ratio
        output_w = self.img_size // self.down_ratio

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)

        num_objs = min(len(bboxes_trans), self.max_objects)

        for k in range(num_objs):
            bbox = bboxes_trans[k]
            cls_id = int(labels_trans[k])

            # Resize box to output resolution
            bbox_out = np.array(bbox) / self.down_ratio

            # Calculate center and dimensions
            h, w = bbox_out[3] - bbox_out[1], bbox_out[2] - bbox_out[0]

            if h > 0 and w > 0:
                radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2, (bbox_out[1] + bbox_out[3]) / 2],
                    dtype=np.float32,
                )
                ct_int = ct.astype(np.int32)

                # Ensure center is within bounds
                if (0 <= ct_int[0] < output_w) and (0 <= ct_int[1] < output_h):
                    draw_umich_gaussian(hm[cls_id], ct_int, radius)

                    wh[k] = 1.0 * w, 1.0 * h
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1

        target = {
            "image": img_tensor,
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "image_id": image_id,
        }

        return target
