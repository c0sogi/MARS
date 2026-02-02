import os
import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset
from library.config import Config
from library.utils import read_dicom
from library.transforms import get_transforms


class ChestXrayDataset(Dataset):
    """
    Dataset class for Chest X-Ray Object Detection and Classification.
    """

    def __init__(self, split="train", transform=None, load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Albumentations transform pipeline.
            load_cached_data (bool): Whether to load/save cached dataframe.
        """
        self.split = split
        self.transform = transform or get_transforms(split)

        # Define cache path
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"cached_{split}_df.parquet")

        # Load Data
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {split} data from {self.cache_path}")
            self.df = pd.read_parquet(self.cache_path)
        else:
            print(f"Processing {split} data from scratch...")
            self.df = self._load_and_process_metadata()
            # Save to cache
            if load_cached_data:
                print(f"Saving {split} data cache to {self.cache_path}")
                self.df.to_parquet(self.cache_path, index=False)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)

        print(f"Dataset {split} initialized. Size: {len(self.df)}")

    def _load_and_process_metadata(self):
        """
        Loads the raw CSV metadata and processes labels.
        """
        if self.split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif self.split == "val":
            path = Config.VAL_METADATA_PATH
        elif self.split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {self.split}")

        df = pd.read_csv(path)

        # Process Study Labels if available (train/val)
        if self.split in ["train", "val"]:
            # Create a single integer label for the study
            # Columns: 'Negative for Pneumonia', 'Typical Appearance', etc.
            # We use argmax to get the index (0-3)
            label_cols = Config.STUDY_LABELS

            # Ensure columns exist
            if all(col in df.columns for col in label_cols):
                df["study_label"] = (
                    df[label_cols].values.argmax(axis=1).astype(np.int64)
                )
            else:
                # Fallback or error if columns missing in train/val
                raise ValueError(
                    f"Study label columns missing in {self.split} metadata."
                )

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Load Image
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        image = read_dicom(file_path)  # Returns (H, W) uint8

        # 2. Prepare Boxes and Labels
        boxes_pascal = []
        class_labels = []

        if self.split in ["train", "val"]:
            # Parse boxes string -> list of dicts
            # Format in CSV: [{'x': 10, 'y': 10, 'width': 100, 'height': 100}, ...]
            if pd.notna(row.get("boxes")):
                try:
                    # boxes column is a string representation of a list
                    raw_boxes = ast.literal_eval(row["boxes"])
                    for box in raw_boxes:
                        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                        # Convert to Pascal VOC [xmin, ymin, xmax, ymax]
                        boxes_pascal.append([x, y, x + w, y + h])
                        class_labels.append(0)  # 0 is the class ID for 'opacity'
                except Exception as e:
                    print(f"Error parsing boxes for {image_id}: {e}")
                    boxes_pascal = []
                    class_labels = []

        # 3. Apply Transforms (Augmentation / Resizing)
        # Albumentations expects boxes in Pascal VOC format
        if self.transform:
            transformed = self.transform(
                image=image, bboxes=boxes_pascal, class_labels=class_labels
            )
            image = transformed["image"]  # Tensor (3, H, W)
            boxes_pascal = transformed["bboxes"]
            # class_labels might change if boxes are dropped (e.g. crop), though we use min_visibility
            class_labels = transformed["class_labels"]

        # 4. Format Targets for DETR
        # Convert Pascal VOC [xmin, ymin, xmax, ymax] -> CXCYWH normalized [0, 1]
        _, h_img, w_img = image.shape  # Tensor shape is (C, H, W)

        boxes_coco = []
        for box in boxes_pascal:
            xmin, ymin, xmax, ymax = box
            w_box = xmax - xmin
            h_box = ymax - ymin
            cx = xmin + w_box / 2
            cy = ymin + h_box / 2

            # Normalize
            boxes_coco.append([cx / w_img, cy / h_img, w_box / w_img, h_box / h_img])

        # Convert to Tensors
        target = {}
        target["boxes"] = torch.as_tensor(boxes_coco, dtype=torch.float32).reshape(
            -1, 4
        )
        target["labels"] = torch.as_tensor(class_labels, dtype=torch.int64)

        # Add Study Label
        if self.split in ["train", "val"]:
            target["study_label"] = torch.as_tensor(
                row["study_label"], dtype=torch.int64
            )
        else:
            # Dummy label for test set
            target["study_label"] = torch.as_tensor(0, dtype=torch.int64)

        # Additional metadata for evaluation
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor(
            [row["height"], row["width"]]
        )  # original H, W
        target["size"] = torch.as_tensor([h_img, w_img])  # resized H, W

        return image, target, image_id
