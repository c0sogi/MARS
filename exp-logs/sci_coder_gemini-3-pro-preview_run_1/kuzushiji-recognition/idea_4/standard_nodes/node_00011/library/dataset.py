import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from library.config import Config
from library.utils import parse_ground_truth


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the object size.
    Derived from: CornerNet (Law and Deng, ECCV 2018)
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


class KuzushijiDataset(Dataset):
    def __init__(self, metadata_df, mode="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image paths and labels.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load parsed annotations from cache.
        """
        self.df = metadata_df
        self.mode = mode
        self.input_size = Config.IMG_SIZE
        self.output_size = self.input_size // 4  # Stride 4 for CenterNet
        self.max_objs = Config.MAX_DETECTIONS

        # Load Unicode Translation Map to build Class Index
        self.unicode_map = pd.read_csv(Config.UNICODE_MAP_PATH)
        # Ensure we cover all characters in the map
        self.chars = self.unicode_map["Unicode"].dropna().unique()
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.num_classes = len(self.chars)

        # Pre-process/Parse Annotations with Caching
        self.anns = {}
        if self.mode in ["train", "val"]:
            self._load_parsed_annotations(load_cached_data)

        # Define Augmentations
        # Strictly Geometric Augmentations as requested
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.input_size),
                    A.PadIfNeeded(
                        min_height=self.input_size,
                        min_width=self.input_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5,
                    ),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            )
        else:
            # Validation/Test: Resize and Pad only
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.input_size),
                    A.PadIfNeeded(
                        min_height=self.input_size,
                        min_width=self.input_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            )

    def _load_parsed_annotations(self, load_cached_data):
        """
        Parses label strings into structured dictionaries.
        Implements caching using .npy files.
        """
        cache_filename = f"parsed_anns_{self.mode}.npy"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        if load_cached_data and os.path.exists(cache_path):
            try:
                self.anns = np.load(cache_path, allow_pickle=True).item()
                return
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

        # Compute from scratch
        print(f"Parsing annotations for {self.mode} set...")
        parsed_data = {}
        for _, row in self.df.iterrows():
            img_id = row["image_id"]
            label_str = row["labels"]

            # Use library utility to parse string
            anns_list = parse_ground_truth(label_str)

            # Convert unicode labels to integer indices
            valid_anns = []
            for ann in anns_list:
                char = ann["label"]
                if char in self.char_to_idx:
                    ann["category_id"] = self.char_to_idx[char]
                    # Convert to list for albumentations [x, y, w, h]
                    ann["bbox"] = [ann["x"], ann["y"], ann["w"], ann["h"]]
                    valid_anns.append(ann)

            parsed_data[img_id] = valid_anns

        self.anns = parsed_data

        # Save to cache
        try:
            np.save(cache_path, parsed_data)
            print(f"Saved parsed annotations to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images (though EDA showed none)
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Get original dimensions
        h, w, c = img.shape

        # Prepare Annotations
        bboxes = []
        category_ids = []

        if self.mode in ["train", "val"] and image_id in self.anns:
            ann_list = self.anns[image_id]
            for ann in ann_list:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])

        # Apply Augmentations
        # Dummy category_id for test set to satisfy Albumentations if bboxes exist (unlikely for test)
        if self.mode == "test":
            augmented = self.transform(image=img, bboxes=[], category_ids=[])
        else:
            augmented = self.transform(
                image=img, bboxes=bboxes, category_ids=category_ids
            )

        img_aug = augmented["image"]
        bboxes_aug = augmented["bboxes"]
        cat_ids_aug = augmented["category_ids"]

        # Normalize Image
        img_tensor = img_aug.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)  # HWC -> CHW

        # Prepare Targets for CenterNet
        # Heatmap: (1, H/4, W/4) - Class Agnostic Objectness
        # Classification: (MaxObjs) - Class IDs
        # Regression: (MaxObjs, 2) - Offsets
        # WH: (MaxObjs, 2) - Size

        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        cat = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = min(len(bboxes_aug), self.max_objs)

        for k in range(num_objs):
            bbox = bboxes_aug[k]
            cls_id = cat_ids_aug[k]

            x, y, w, h_box = bbox

            # Map to output stride
            ct = np.array([x + w / 2, y + h_box / 2], dtype=np.float32)
            ct_out = ct / 4.0

            ct_int = ct_out.astype(np.int32)

            # Check bounds
            if not (
                0 <= ct_int[0] < self.output_size and 0 <= ct_int[1] < self.output_size
            ):
                continue

            # Gaussian Radius
            radius = gaussian_radius((math_h := h_box / 4, math_w := w / 4))
            radius = max(0, int(radius))

            # Draw Gaussian on Heatmap (Class Agnostic)
            draw_umich_gaussian(hm[0], ct_int, radius)

            # Populate Regression Targets
            wh[k] = [1.0 * w / 4, 1.0 * h_box / 4]
            reg[k] = [ct_out[0] - ct_int[0], ct_out[1] - ct_int[1]]
            ind[k] = ct_int[1] * self.output_size + ct_int[0]
            cat[k] = cls_id
            reg_mask[k] = 1

        # Return Dictionary
        ret = {
            "image": img_tensor,
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "cat": torch.from_numpy(cat),
            "reg_mask": torch.from_numpy(reg_mask),
            "image_id": image_id,
            "orig_h": h,
            "orig_w": w,
        }

        return ret
