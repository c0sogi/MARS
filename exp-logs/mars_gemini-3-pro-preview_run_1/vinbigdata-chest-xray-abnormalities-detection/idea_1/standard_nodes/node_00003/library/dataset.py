import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
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

    def read_dicom_manual(self, file_path):
        """
        Manually parses DICOM file to extract image data when pydicom is missing.
        Supports JPEG, JPEG2000, and Raw (Implicit/Explicit VR).
        """
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except Exception:
            return None

        def read_u16(b, offset):
            return int.from_bytes(b[offset : offset + 2], "little")

        def read_u32(b, offset):
            return int.from_bytes(b[offset : offset + 4], "little")

        # 1. Search for Pixel Data Tag (7FE0,0010) -> E0 7F 10 00
        idx_pixels = data.find(b"\xe0\x7f\x10\x00")

        pixel_data_start = 0
        is_encapsulated = False

        if idx_pixels != -1:
            # Check VR (Explicit)
            vr = data[idx_pixels + 4 : idx_pixels + 6]
            if vr in [b"OB", b"OW", b"UN"]:
                # Explicit VR: Tag(4) + VR(2) + Res(2) + Len(4)
                length = read_u32(data, idx_pixels + 8)
                pixel_data_start = idx_pixels + 12
            else:
                # Implicit VR: Tag(4) + Len(4)
                length = read_u32(data, idx_pixels + 4)
                pixel_data_start = idx_pixels + 8

            if length == 0xFFFFFFFF:
                is_encapsulated = True

        # 2. Try JPEG/JP2 Extraction (Encapsulated or Global Search)
        # If encapsulated, search from pixel_data_start. Else, search globally (fallback).
        search_start = pixel_data_start if is_encapsulated else 0
        search_area = data[search_start:]

        # JPEG (FF D8)
        jpeg_idx = search_area.find(b"\xff\xd8")
        if jpeg_idx != -1:
            img_bytes = search_area[jpeg_idx:]
            # Decode
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

        # JP2 (Signature: 00 00 00 0C 6A 50 20 20)
        jp2_idx = search_area.find(b"\x00\x00\x00\x0c\x6a\x50\x20\x20")
        if jp2_idx != -1:
            img_bytes = search_area[jp2_idx:]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

        # J2K (Codestream: FF 4F FF 51)
        j2k_idx = search_area.find(b"\xff\x4f\xff\x51")
        if j2k_idx != -1:
            img_bytes = search_area[j2k_idx:]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

        # 3. Try Raw Uncompressed (Implicit/Explicit VR)
        # We need Rows (0028,0010) and Columns (0028,0011)
        rows = 0
        cols = 0

        # Helper to find tag value (uint16)
        def find_tag_u16(tag_bytes):
            idx = data.find(tag_bytes)
            if idx != -1:
                # Check Explicit VR (US/SS)
                if data[idx + 4 : idx + 6] in [b"US", b"SS"]:
                    return read_u16(data, idx + 8)
                else:
                    # Implicit
                    return read_u16(data, idx + 8)
            return 0

        rows = find_tag_u16(b"\x28\x00\x10\x00")
        cols = find_tag_u16(b"\x28\x00\x11\x00")

        if rows > 0 and cols > 0 and idx_pixels != -1 and not is_encapsulated:
            # Determine Bit Depth (Bits Stored 0028,0101)
            bits_stored = find_tag_u16(b"\x28\x00\x01\x01")
            dtype = np.uint16 if bits_stored > 8 else np.uint8
            item_size = 2 if dtype == np.uint16 else 1

            expected_bytes = rows * cols * item_size

            if pixel_data_start + expected_bytes <= len(data):
                arr = np.frombuffer(
                    data[pixel_data_start : pixel_data_start + expected_bytes],
                    dtype=dtype,
                )
                img = arr.reshape((rows, cols))

                # Photometric Interpretation (MONOCHROME1 = Invert)
                # Search for MONOCHROME1 in header
                if b"MONOCHROME1" in data[:4096]:
                    img = np.max(img) - img

                return img

        return None

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

        # Strategy 1: Rasterio (Primary for DICOM when pydicom is missing)
        # Cite solution_lesson_node_00001: Fail loudly if data loading fails.
        try:
            with rasterio.open(full_path) as src:
                img = src.read(1)
        except Exception:
            pass

        # Strategy 2: OpenCV (Fallback)
        if img is None:
            try:
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                # If loaded as multi-channel, convert or pick one
                if img is not None:
                    if len(img.shape) == 3:
                        img = img[:, :, 0]  # Take first channel
            except Exception:
                pass

        # Strategy 3: Manual DICOM Parsing (Robust Fallback)
        if img is None:
            try:
                img = self.read_dicom_manual(full_path)
                if img is not None:
                    if len(img.shape) == 3:
                        # Convert RGB/BGR to Gray
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        # Critical Failure Check
        if img is None:
            raise FileNotFoundError(
                f"Failed to load image at {full_path}. Ensure supported libraries are installed."
            )

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

        bboxes = []
        labels = []

        if img is None:
            # Fallback for failed loads: Black image, no boxes
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            # bboxes and labels remain empty
        else:
            # Normal flow
            if self.split != "test" and not anns.empty:
                for _, row in anns.iterrows():
                    # Filter out "No finding" (Class 14) for detection targets
                    if row["class_id"] == 14:
                        continue

                    # Bounding box
                    x_min, y_min = row["x_min"], row["y_min"]
                    x_max, y_max = row["x_max"], row["y_max"]

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
