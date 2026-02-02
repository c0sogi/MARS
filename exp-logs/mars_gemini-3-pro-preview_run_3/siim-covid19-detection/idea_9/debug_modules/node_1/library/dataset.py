import os
import ast
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.transforms import get_transforms


class CovidDataset(Dataset):
    """
    PyTorch Dataset for COVID-19 Radiography detection and classification.
    """

    def __init__(self, split, load_cached_data=True, transform=None, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.transform = transform
        self.debug = debug

        # Select appropriate CSV path
        if split == "train":
            self.csv_path = Config.TRAIN_CSV
        elif split == "val":
            self.csv_path = Config.VAL_CSV
        elif split == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load and process metadata
        self.df = self._load_metadata(self.csv_path, split, load_cached_data)

        if self.debug:
            self.df = self.df.iloc[:50].reset_index(drop=True)

    def _load_metadata(self, csv_path, split, load_cached_data):
        """
        Loads metadata from CSV, parses boxes/labels, and caches to Parquet.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_df.parquet")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                # If load fails, fall back to processing
                pass

        # 2. Process from scratch
        df = pd.read_csv(csv_path)

        # Parse Boxes (convert string representation to list of [x1, y1, x2, y2])
        if "boxes" in df.columns:

            def parse_box_str(x):
                if pd.isna(x):
                    return []
                try:
                    # Input format: [{'x': 10, 'y': 10, 'width': 50, 'height': 50}]
                    dicts = ast.literal_eval(x)
                    boxes = []
                    for d in dicts:
                        # Convert to Pascal VOC: [xmin, ymin, xmax, ymax]
                        boxes.append(
                            [d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]]
                        )
                    return boxes
                except:
                    return []

            df["parsed_boxes"] = df["boxes"].apply(parse_box_str)
        else:
            df["parsed_boxes"] = [[] for _ in range(len(df))]

        # Parse Study Labels (One-hot to Index)
        label_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]

        # Check if label columns exist (Train/Val) or create dummy (Test)
        if all(col in df.columns for col in label_cols):
            # argmax returns index 0-3 corresponding to the columns
            df["study_label"] = df[label_cols].values.argmax(axis=1)
        else:
            df["study_label"] = -1

        # 3. Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            # If caching fails (e.g. list column serialization issues), proceed without caching
            pass

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Image Loading ---
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            dcm = pydicom.dcmread(img_path)
            image = dcm.pixel_array

            # Fix Photometric Interpretation (MONOCHROME1 means 0 is white)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                image = np.max(image) - image

            # Normalize to 0-255 and convert to uint8
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255.0
            image = image.astype(np.uint8)

            # Convert Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        except Exception:
            # Fallback for corrupt files (creates black image)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]

        # --- 2. Target Preparation ---
        # Retrieve boxes
        # When loading from parquet, lists might be numpy arrays or lists
        boxes = row["parsed_boxes"]
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        elif not isinstance(boxes, list):
            boxes = []

        # Create detection labels (1 for opacity)
        labels = [1] * len(boxes)

        study_label = int(row["study_label"])

        # --- 3. Augmentation ---
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            # Basic tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # --- 4. Tensor Formatting ---
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "study_label": torch.tensor(study_label, dtype=torch.int64),
            "image_id": row["image_id"],
            "study_id": row["study_id"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int32),
        }

        return image, target


def collate_fn(batch):
    """
    Collates a batch of images and targets.
    Images are stacked into a tensor.
    Targets are kept as a list of dictionaries (required for detection models).
    """
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)

    return images, targets


def get_dataloader(split, batch_size=None, shuffle=None, num_workers=None, debug=False):
    """
    Factory function to create a DataLoader for a specific split.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if shuffle is None:
        shuffle = split == "train"

    # Get transforms from library
    transform = get_transforms(split)

    dataset = CovidDataset(
        split=split, load_cached_data=True, transform=transform, debug=debug
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return dataloader
