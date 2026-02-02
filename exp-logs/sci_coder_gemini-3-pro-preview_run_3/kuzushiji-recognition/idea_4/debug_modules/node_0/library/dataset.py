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
from library.utils import gaussian_radius, draw_gaussian

# ==========================================
# Helper Functions
# ==========================================


def get_class_map(df, save_path=Config.CLASS_MAP_PATH):
    """
    Generates or loads a mapping from Unicode character to integer ID.
    """
    if os.path.exists(save_path):
        class_map = np.load(save_path, allow_pickle=True).item()
        return class_map

    # Extract all unique labels
    unique_labels = set()
    for labels_str in df["labels"]:
        if not isinstance(labels_str, str) or not labels_str:
            continue
        parts = labels_str.split()
        # Format: Code X Y W H
        for i in range(0, len(parts), 5):
            unique_labels.add(parts[i])

    # Sort for determinism
    sorted_labels = sorted(list(unique_labels))
    class_map = {label: idx for idx, label in enumerate(sorted_labels)}

    # Save
    np.save(save_path, class_map)
    return class_map


def prepare_classifier_data(
    df, class_map, split_name, cache_dir=Config.CACHE_DIR, load_cached_data=True
):
    """
    Parses the dataframe to create a list of (image_path, class_id, bbox).
    Caches the result to disk.
    """
    cache_path = os.path.join(cache_dir, f"classifier_samples_{split_name}.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached classifier samples from {cache_path}")
        return np.load(cache_path, allow_pickle=True).tolist()

    print(f"Processing classifier data for {split_name} from scratch...")
    samples = []

    for _, row in df.iterrows():
        img_path = row["file_path"]
        labels_str = row["labels"]

        if not isinstance(labels_str, str) or not labels_str:
            continue

        parts = labels_str.split()
        # Format: Code X Y W H
        for i in range(0, len(parts), 5):
            code = parts[i]
            try:
                x = int(parts[i + 1])
                y = int(parts[i + 2])
                w = int(parts[i + 3])
                h = int(parts[i + 4])

                if code in class_map:
                    class_id = class_map[code]
                    samples.append(
                        {
                            "image_path": img_path,
                            "class_id": class_id,
                            "bbox": [x, y, w, h],
                        }
                    )
            except ValueError:
                continue

    # Save
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, samples)
    print(f"Saved {len(samples)} classifier samples to {cache_path}")

    return samples


def get_transforms(mode="train", img_size=1024):
    """
    Returns Albumentations transforms.
    """
    if mode == "train_detector":
        return A.Compose(
            [
                # Scale +/- 30%
                A.RandomScale(scale_limit=0.3, p=0.5),
                # Rotate +/- 5 degrees
                A.Rotate(limit=5, p=0.5),
                # Random Crop to patch size
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                # Color augmentations
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", min_visibility=0.3, label_fields=["class_labels"]
            ),
        )

    elif mode == "val_detector":
        # For validation, we use RandomCrop to ensure fixed tensor size
        return A.Compose(
            [
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", min_visibility=0.3, label_fields=["class_labels"]
            ),
        )

    elif mode == "train_classifier":
        return A.Compose(
            [
                A.Resize(
                    height=Config.CLASSIFIER_INPUT_SIZE,
                    width=Config.CLASSIFIER_INPUT_SIZE,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )

    elif mode == "val_classifier":
        return A.Compose(
            [
                A.Resize(
                    height=Config.CLASSIFIER_INPUT_SIZE,
                    width=Config.CLASSIFIER_INPUT_SIZE,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )

    return A.Compose([ToTensorV2()])


# ==========================================
# Datasets
# ==========================================


class PatchDetectorDataset(Dataset):
    def __init__(self, metadata_df, mode="train_detector", transform=None):
        self.df = metadata_df
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR
        self.stride = Config.DETECTOR_STRIDE
        self.output_size = Config.DETECTOR_INPUT_SIZE // self.stride

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback
            image = np.zeros(
                (Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE, 3),
                dtype=np.uint8,
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse Labels
        labels_str = row["labels"]
        bboxes = []
        class_labels = []

        if isinstance(labels_str, str) and labels_str:
            parts = labels_str.split()
            for i in range(0, len(parts), 5):
                try:
                    x = float(parts[i + 1])
                    y = float(parts[i + 2])
                    w = float(parts[i + 3])
                    h = float(parts[i + 4])
                    bboxes.append([x, y, w, h])
                    class_labels.append(1)  # Single class for detector
                except ValueError:
                    continue

        # Pad if smaller than patch size
        h, w, _ = image.shape
        pad_h = max(0, Config.DETECTOR_INPUT_SIZE - h)
        pad_w = max(0, Config.DETECTOR_INPUT_SIZE - w)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Apply Transforms
        if self.transform:
            try:
                transformed = self.transform(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
                image = transformed["image"]
                bboxes = transformed["bboxes"]
            except Exception as e:
                print(f"Transform failed at {img_path}: {e}")
                image = torch.zeros(
                    (3, Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE)
                )
                bboxes = []

        # Generate CenterNet Targets
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)

        for bbox in bboxes:
            x, y, w, h = bbox
            ct_x = x + w / 2
            ct_y = y + h / 2

            ct_x_idx = ct_x / self.stride
            ct_y_idx = ct_y / self.stride

            ct_int_x = int(ct_x_idx)
            ct_int_y = int(ct_y_idx)

            if (
                ct_int_x >= 0
                and ct_int_x < self.output_size
                and ct_int_y >= 0
                and ct_int_y < self.output_size
            ):
                radius = gaussian_radius(
                    (math.ceil(h / self.stride), math.ceil(w / self.stride))
                )
                radius = max(0, int(radius))
                draw_gaussian(hm[0], (ct_int_x, ct_int_y), radius)

                wh[0, ct_int_y, ct_int_x] = w
                wh[1, ct_int_y, ct_int_x] = h

                reg[0, ct_int_y, ct_int_x] = ct_x_idx - ct_int_x
                reg[1, ct_int_y, ct_int_x] = ct_y_idx - ct_int_y

                reg_mask[0, ct_int_y, ct_int_x] = 1

        target = {
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "reg_mask": torch.from_numpy(reg_mask),
        }

        return image, target


class CharacterCropDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        class_map,
        split_name="train",
        mode="train_classifier",
        transform=None,
        cache_images=True,
    ):
        self.samples = prepare_classifier_data(metadata_df, class_map, split_name)
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR
        self.cache_images = cache_images
        self.image_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_rel_path = sample["image_path"]
        class_id = sample["class_id"]
        bbox = sample["bbox"]  # x, y, w, h

        # Load Image
        image = None
        if self.cache_images:
            image = self.image_cache.get(img_rel_path)

        if image is None:
            img_full_path = os.path.join(self.input_dir, img_rel_path)
            image = cv2.imread(img_full_path)
            if image is None:
                image = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.cache_images:
                self.image_cache[img_rel_path] = image

        # Crop
        x, y, w, h = bbox
        img_h, img_w, _ = image.shape

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))

        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
        else:
            crop = np.zeros(
                (Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE, 3),
                dtype=np.uint8,
            )

        # Transform
        if self.transform:
            transformed = self.transform(image=crop)
            crop = transformed["image"]

        return crop, torch.tensor(class_id, dtype=torch.long)
