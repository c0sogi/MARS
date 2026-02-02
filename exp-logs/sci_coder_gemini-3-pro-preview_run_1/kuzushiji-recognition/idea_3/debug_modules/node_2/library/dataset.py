import os
import cv2
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size.
    Standard formula from CornerNet/CenterNet.
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


def gaussian_2d(shape, sigma=1):
    """Generates a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in (shape, shape)]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a Gaussian kernel on the heatmap at the specified center.
    Modifies heatmap in-place.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d(diameter, sigma=diameter / 6)

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


class KuzushijiDataset(Dataset):
    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Optional override for transforms.
            load_cached_data (bool): Whether to use cached parsed annotations.
        """
        self.mode = mode
        self.config = Config

        # Determine paths based on mode
        if mode == "train":
            self.metadata_path = self.config.TRAIN_METADATA_PATH
            self.img_dir = self.config.TRAIN_IMG_DIR
        elif mode == "val":
            self.metadata_path = self.config.VAL_METADATA_PATH
            self.img_dir = (
                self.config.TRAIN_IMG_DIR
            )  # Val images are in train_images folder
        else:
            self.metadata_path = self.config.TEST_METADATA_PATH
            self.img_dir = self.config.TEST_IMG_DIR

        # Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)

        # Load Class Mapping
        self.char2id, self.id2char = self.config.get_class_mapping()
        self.num_classes = len(self.char2id)

        # Parse Annotations (with caching)
        # Test set does not have ground truth labels to parse
        if mode in ["train", "val"]:
            self.annotations = self._load_or_parse_annotations(load_cached_data)
        else:
            self.annotations = {}

        # Setup Transforms
        if transform is None:
            self.transform = self._get_transforms()
        else:
            self.transform = transform

        # CenterNet specific configurations
        self.down_ratio = 4  # Swin-B FPN output stride
        self.max_objs = self.config.MAX_DETECTIONS

    def _get_transforms(self):
        """
        Returns Albumentations transforms.
        Strictly geometric for training (no noise/blur), resizing/padding for val/test.
        """
        # Base: Resize longest edge to IMG_SIZE, then pad to square
        base_transforms = [
            A.LongestMaxSize(max_size=self.config.IMG_SIZE),
            A.PadIfNeeded(
                min_height=self.config.IMG_SIZE,
                min_width=self.config.IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,  # Black padding
                position="center",
            ),
        ]

        if self.mode == "train":
            # Geometric Augmentations: Shift, Scale, Rotate
            # We avoid photometric distortions to preserve ink stroke details
            aug_transforms = [
                A.ShiftScaleRotate(
                    shift_limit=self.config.AUG_TRANSLATE,
                    scale_limit=(
                        self.config.AUG_SCALE_RANGE[0] - 1,
                        self.config.AUG_SCALE_RANGE[1] - 1,
                    ),
                    rotate_limit=self.config.AUG_ROTATION,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                )
            ]
            transforms = A.Compose(
                aug_transforms
                + base_transforms
                + [
                    A.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["class_ids"]),
            )
        else:
            # Val/Test: Deterministic resizing
            bbox_params = (
                A.BboxParams(format="coco", label_fields=["class_ids"])
                if self.mode == "val"
                else None
            )
            transforms = A.Compose(
                base_transforms
                + [
                    A.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
                    ToTensorV2(),
                ],
                bbox_params=bbox_params,
            )

        return transforms

    def _load_or_parse_annotations(self, load_cached_data):
        """
        Parses label strings into structured data.
        Caches result to ./working/idea_3/ to speed up future inits.
        """
        cache_file = f"{self.mode}_parsed_anns.npy"
        cache_path = os.path.join(self.config.WORK_DIR, cache_file)

        # Ensure working directory exists
        os.makedirs(self.config.WORK_DIR, exist_ok=True)

        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"[{self.mode}] Loading cached annotations from {cache_path}")
                return np.load(cache_path, allow_pickle=True).item()
            except Exception as e:
                print(f"[{self.mode}] Failed to load cache: {e}. Reparsing.")

        print(f"[{self.mode}] Parsing annotations...")
        parsed_data = {}

        for idx, row in self.metadata.iterrows():
            img_id = row["image_id"]
            label_str = row["labels"]

            bboxes = []
            class_ids = []

            if isinstance(label_str, str) and label_str.strip():
                parts = label_str.strip().split(" ")
                # Format: Unicode X Y W H ...
                for i in range(0, len(parts), 5):
                    try:
                        unicode_char = parts[i]
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        w = int(parts[i + 3])
                        h = int(parts[i + 4])

                        if unicode_char in self.char2id:
                            cid = self.char2id[unicode_char]
                            # COCO format: [x, y, w, h]
                            bboxes.append([x, y, w, h])
                            class_ids.append(cid)
                    except (ValueError, IndexError):
                        continue

            parsed_data[img_id] = {"bboxes": bboxes, "class_ids": class_ids}

        # Save to cache
        try:
            np.save(cache_path, parsed_data)
            print(f"[{self.mode}] Saved annotations to {cache_path}")
        except Exception as e:
            print(f"[{self.mode}] Failed to save cache: {e}")

        return parsed_data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["image_id"]

        # Construct full image path
        img_path = os.path.join(self.config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback (should not happen based on EDA)
            image = np.zeros(
                (self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8
            )

        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- TRAIN / VAL MODE ---
        if self.mode in ["train", "val"]:
            ann = self.annotations[img_id]
            bboxes = ann["bboxes"]  # [x, y, w, h]
            class_ids = ann["class_ids"]

            # Apply Transforms
            if self.transform:
                try:
                    transformed = self.transform(
                        image=image, bboxes=bboxes, class_ids=class_ids
                    )
                    image_tensor = transformed["image"]
                    bboxes = transformed["bboxes"]
                    class_ids = transformed["class_ids"]
                except Exception as e:
                    # Fallback if geometric aug fails (e.g. box pushed out of bounds)
                    # Use a safe transform (resize only)
                    print(f"Augmentation error on {img_id}: {e}")
                    fallback = A.Compose(
                        [
                            A.LongestMaxSize(max_size=self.config.IMG_SIZE),
                            A.PadIfNeeded(
                                min_height=self.config.IMG_SIZE,
                                min_width=self.config.IMG_SIZE,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0,
                            ),
                            A.Normalize(
                                mean=self.config.NORM_MEAN, std=self.config.NORM_STD
                            ),
                            ToTensorV2(),
                        ],
                        bbox_params=A.BboxParams(
                            format="coco", label_fields=["class_ids"]
                        ),
                    )
                    transformed = fallback(
                        image=image, bboxes=ann["bboxes"], class_ids=ann["class_ids"]
                    )
                    image_tensor = transformed["image"]
                    bboxes = transformed["bboxes"]
                    class_ids = transformed["class_ids"]

            # --- Generate CenterNet Targets ---
            output_h = self.config.IMG_SIZE // self.down_ratio
            output_w = self.config.IMG_SIZE // self.down_ratio

            # Initialize targets
            # hm: (1, H, W) - Objectness heatmap
            hm = np.zeros((1, output_h, output_w), dtype=np.float32)
            # wh: (K, 2) - Width/Height
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            # reg: (K, 2) - Local offsets
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            # ind: (K) - Indices for gather
            ind = np.zeros((self.max_objs), dtype=np.int64)
            # cls_ids: (K) - Class IDs for specific centers
            cls_ids_target = np.zeros((self.max_objs), dtype=np.int64)
            # reg_mask: (K) - Mask valid objects
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

            num_objs = min(len(bboxes), self.max_objs)

            for k in range(num_objs):
                bbox = bboxes[k]
                cls_id = class_ids[k]

                # BBox is [x, y, w, h] in transformed image coordinates
                x, y, w, h = bbox

                # Calculate center and size in output feature map coordinates
                ct_x = (x + w / 2) / self.down_ratio
                ct_y = (y + h / 2) / self.down_ratio
                h_out = h / self.down_ratio
                w_out = w / self.down_ratio

                ct_int_x = int(ct_x)
                ct_int_y = int(ct_y)

                # Bounds check
                if (
                    ct_int_x < 0
                    or ct_int_x >= output_w
                    or ct_int_y < 0
                    or ct_int_y >= output_h
                ):
                    continue

                # Gaussian Radius
                radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
                radius = max(0, int(radius))

                # Draw Gaussian (channel 0)
                draw_gaussian(hm[0], (ct_int_x, ct_int_y), radius)

                # Fill regression targets
                ind[k] = ct_int_y * output_w + ct_int_x
                reg[k] = [ct_x - ct_int_x, ct_y - ct_int_y]
                wh[k] = [w_out, h_out]
                cls_ids_target[k] = cls_id
                reg_mask[k] = 1

            return {
                "image": image_tensor,
                "hm": torch.from_numpy(hm),
                "ind": torch.from_numpy(ind),
                "wh": torch.from_numpy(wh),
                "reg": torch.from_numpy(reg),
                "cls_ids": torch.from_numpy(cls_ids_target),
                "reg_mask": torch.from_numpy(reg_mask),
            }

        # --- TEST MODE ---
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image_tensor = transformed["image"]
            else:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

            return {
                "image": image_tensor,
                "image_id": img_id,
                "orig_h": orig_h,
                "orig_w": orig_w,
            }
