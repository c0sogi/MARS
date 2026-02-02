import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import gaussian_radius, draw_umich_gaussian

# ==========================================
# 1. Helper Functions & Caching Logic
# ==========================================


def get_class_map(load_cached=True):
    """
    Creates or loads a mapping between Unicode characters and integer indices.
    """
    cache_path = Config.CACHE_CLASS_MAP

    if load_cached and os.path.exists(cache_path):
        try:
            class_map = np.load(cache_path, allow_pickle=True).item()
            return class_map["char_to_idx"], class_map["idx_to_char"]
        except Exception:
            pass  # Fallback to creation

    # Create from unicode_translation.csv to ensure global consistency
    df = pd.read_csv(Config.UNICODE_MAP_PATH)
    chars = df["Unicode"].unique().tolist()
    chars.sort()

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char})

    return char_to_idx, idx_to_char


def process_detector_metadata(metadata_path, cache_path, load_cached=True):
    """
    Parses the string-based metadata into a structured list for the Detector.
    """
    if load_cached and os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True).tolist()
        except Exception:
            pass

    df = pd.read_csv(metadata_path, keep_default_na=False)
    data = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        # Construct full path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        labels_str = row["labels"]
        bboxes = []

        if labels_str:
            parts = labels_str.split()
            # Format: Unicode X Y W H
            num_chars = len(parts) // 5
            for i in range(num_chars):
                code = parts[i * 5]
                try:
                    x = int(parts[i * 5 + 1])
                    y = int(parts[i * 5 + 2])
                    w = int(parts[i * 5 + 3])
                    h = int(parts[i * 5 + 4])
                    bboxes.append({"code": code, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue

        data.append({"image_id": img_id, "image_path": full_path, "bboxes": bboxes})

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array(data, dtype=object))

    return data


def process_classifier_metadata(
    metadata_path, cache_path, char_to_idx, load_cached=True
):
    """
    Flattens page-level metadata into crop-level metadata for the Classifier.
    """
    if load_cached and os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True).tolist()
        except Exception:
            pass

    df = pd.read_csv(metadata_path, keep_default_na=False)
    data = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        labels_str = row["labels"]

        if not labels_str:
            continue

        parts = labels_str.split()
        num_chars = len(parts) // 5

        for i in range(num_chars):
            code = parts[i * 5]
            if code not in char_to_idx:
                continue

            try:
                x = int(parts[i * 5 + 1])
                y = int(parts[i * 5 + 2])
                w = int(parts[i * 5 + 3])
                h = int(parts[i * 5 + 4])

                data.append(
                    {
                        "image_path": full_path,
                        "bbox": [x, y, w, h],
                        "label_idx": char_to_idx[code],
                    }
                )
            except ValueError:
                continue

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array(data, dtype=object))

    return data


# ==========================================
# 2. Augmentations
# ==========================================


def get_transforms(stage="detector", split="train", img_size=1024):
    """
    Returns Albumentations transforms for the specific stage and split.
    """
    if stage == "detector":
        # Detector Transforms: Geometric focused
        if split == "train":
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=img_size),
                    A.PadIfNeeded(
                        min_height=img_size,
                        min_width=img_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    # Limited rotation to preserve text structure
                    # Cite solution_lesson_node_00006: Increased scale variance to drive generalization
                    # Cite solution_lesson_node_00007: Reduced rotation to avoid false difficulty
                    # Cite solution_lesson_node_00011: Prioritize scale augmentation over rotation
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.3,
                        rotate_limit=5,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="coco", min_visibility=0.3, label_fields=["labels"]
                ),
            )
        else:
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=img_size),
                    A.PadIfNeeded(
                        min_height=img_size,
                        min_width=img_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            )

    elif stage == "classifier":
        # Classifier Transforms: Appearance focused
        if split == "train":
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.0),  # Text shouldn't be flipped
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussNoise(p=0.1),
                    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.1),
                    A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                    ToTensorV2(),
                ]
            )

    return None


# ==========================================
# 3. Datasets
# ==========================================


class KuzushijiDetectorDataset(Dataset):
    """
    Dataset for Stage 1: Class-Agnostic Keypoint Detection.
    """

    def __init__(self, split="train", debug=False, load_cached=True):
        self.split = split
        self.input_size = Config.DETECTOR_IMG_SIZE
        self.stride = Config.DETECTOR_OUTPUT_STRIDE
        self.output_size = self.input_size // self.stride

        # Select metadata file
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.CACHE_DETECTOR_TRAIN
        else:
            meta_path = Config.VAL_METADATA_PATH
            cache_path = Config.CACHE_DETECTOR_VAL

        # Load Data
        self.data = process_detector_metadata(
            meta_path, cache_path, load_cached=load_cached
        )

        if debug:
            self.data = self.data[: Config.DEBUG_SAMPLE_SIZE]

        self.transforms = get_transforms("detector", split, self.input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]

        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare BBoxes for Albumentations
        # Format: [x, y, w, h]
        bboxes = []
        for b in item["bboxes"]:
            bboxes.append([b["x"], b["y"], b["w"], b["h"]])

        # Dummy labels (class agnostic)
        labels = [1] * len(bboxes)

        # Apply Augmentations
        try:
            transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img_tensor = transformed["image"]
            aug_bboxes = transformed["bboxes"]
        except Exception:
            # Fallback if augmentation fails (e.g. bbox out of bounds)
            return None

        # Initialize Targets
        heatmap = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        size_map = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        offset_map = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)

        # Generate Targets
        for box in aug_bboxes:
            x, y, w, h = box

            # Map to feature map scale
            # Note: The model predicts in feature map scale, so we regress towards feature map w/h
            feat_x = x / self.stride
            feat_y = y / self.stride
            feat_w = w / self.stride
            feat_h = h / self.stride

            # Center
            ct_x = feat_x + feat_w / 2
            ct_y = feat_y + feat_h / 2
            ct_int_x = int(ct_x)
            ct_int_y = int(ct_y)

            # Boundary check
            if not (
                0 <= ct_int_x < self.output_size and 0 <= ct_int_y < self.output_size
            ):
                continue

            # Radius
            radius = gaussian_radius((feat_h, feat_w))
            radius = max(0, int(radius))

            # Draw Gaussian
            draw_umich_gaussian(heatmap[0], (ct_int_x, ct_int_y), radius)

            # Size Target (w, h)
            size_map[0, ct_int_y, ct_int_x] = feat_w
            size_map[1, ct_int_y, ct_int_x] = feat_h

            # Offset Target
            offset_map[0, ct_int_y, ct_int_x] = ct_x - ct_int_x
            offset_map[1, ct_int_y, ct_int_x] = ct_y - ct_int_y

        return {
            "img": img_tensor,
            "heatmap": torch.from_numpy(heatmap),
            "size_map": torch.from_numpy(size_map),
            "offset_map": torch.from_numpy(offset_map),
            "meta": {"image_id": item["image_id"], "orig_size": img.shape[:2]},
        }


class KuzushijiCropDataset(Dataset):
    """
    Dataset for Stage 2: Character Classification on Crops.
    """

    def __init__(self, split="train", debug=False, load_cached=True):
        self.split = split
        self.img_size = Config.CLASSIFIER_IMG_SIZE

        # Load Class Map
        self.char_to_idx, _ = get_class_map(load_cached)

        # Select metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.CACHE_CLASSIFIER_TRAIN
        else:
            meta_path = Config.VAL_METADATA_PATH
            cache_path = Config.CACHE_CLASSIFIER_VAL

        # Load Data
        self.data = process_classifier_metadata(
            meta_path, cache_path, self.char_to_idx, load_cached
        )

        if debug:
            self.data = self.data[: Config.DEBUG_SAMPLE_SIZE]

        self.transforms = get_transforms("classifier", split, self.img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        x, y, w, h = item["bbox"]
        label_idx = item["label_idx"]

        # Load Parent Image
        # Note: In a production setting with workers, OS caching handles repeated reads efficiently.
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop
        # Handle boundaries
        img_h, img_w = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Apply Transforms
        transformed = self.transforms(image=crop)
        img_tensor = transformed["image"]

        return {"img": img_tensor, "label_idx": label_idx}


class KuzushijiTestDataset(Dataset):
    """
    Dataset for Inference (Stage 1).
    Loads test images for the detector.
    """

    def __init__(self, load_cached=True):
        self.input_size = Config.DETECTOR_IMG_SIZE
        self.df = pd.read_csv(Config.TEST_METADATA_PATH)
        self.transforms = get_transforms("detector", "test", self.input_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        img = cv2.imread(full_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Dummy bbox for transform compatibility
        dummy_bbox = [[0, 0, 10, 10]]
        dummy_labels = [1]

        transformed = self.transforms(image=img, bboxes=dummy_bbox, labels=dummy_labels)

        return {
            "img": transformed["image"],
            "image_id": img_id,
            "orig_shape": (orig_h, orig_w),
        }
