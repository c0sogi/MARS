import os
import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import read_dicom_image, get_transforms


class SIIMDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        """
        Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load processed dataframe from cache.
        """
        self.split = split
        self.input_dir = Config.INPUT_DIR

        # Determine metadata source
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Invalid split: {split}")

        # Caching Logic for Dataframe
        cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_df.parquet")

        if load_cached_data and os.path.exists(cache_path):
            try:
                self.df = pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from CSV.")
                self.df = self._load_and_process_csv()
                self.df.to_parquet(cache_path, index=False)
        else:
            self.df = self._load_and_process_csv()
            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.df.to_parquet(cache_path, index=False)

        # Load transforms
        self.transforms = get_transforms(split)

    def _load_and_process_csv(self):
        """
        Loads the CSV and processes labels.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Process Study Labels for Train/Val
        if self.split != "test":
            study_label_cols = [
                "Negative for Pneumonia",
                "Typical Appearance",
                "Indeterminate Appearance",
                "Atypical Appearance",
            ]
            # Verify columns exist
            if all(col in df.columns for col in study_label_cols):
                # Argmax to get class index 0-3
                df["study_label"] = df[study_label_cols].values.argmax(axis=1)
            else:
                # Fallback or error if columns missing in train/val
                raise ValueError("Study label columns missing in metadata.")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = os.path.join(self.input_dir, row["file_path"])

        # 1. Load Image
        # Returns (H, W, 3) uint8 RGB
        image = read_dicom_image(file_path)
        h_orig, w_orig, _ = image.shape

        # 2. Parse Boxes (if available)
        boxes = []
        labels = []  # 1 for opacity

        if self.split != "test":
            box_str = row.get("boxes", np.nan)
            if pd.notna(box_str) and box_str != "nan":
                try:
                    # Format: "[{'x': 10, 'y': 10, 'width': 100, 'height': 100}, ...]"
                    box_dicts = ast.literal_eval(box_str)
                    for b in box_dicts:
                        x_min = float(b["x"])
                        y_min = float(b["y"])
                        w = float(b["width"])
                        h = float(b["height"])
                        x_max = x_min + w
                        y_max = y_min + h

                        # Clip to image boundaries
                        x_min = max(0, min(x_min, w_orig))
                        y_min = max(0, min(y_min, h_orig))
                        x_max = max(0, min(x_max, w_orig))
                        y_max = max(0, min(y_max, h_orig))

                        # Validate box
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # Class 1: Opacity
                except Exception as e:
                    # In case of malformed string, treat as no finding
                    boxes = []
                    labels = []

        # 3. Apply Augmentations
        # Albumentations expects boxes as list of lists
        if len(boxes) > 0:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image_t = transformed["image"]
            boxes_t = transformed["bboxes"]
            labels_t = transformed["labels"]
        else:
            # Handle empty boxes
            transformed = self.transforms(image=image, bboxes=[], labels=[])
            image_t = transformed["image"]
            boxes_t = []
            labels_t = []

        # 4. Construct Target Dictionary
        target = {}

        # Boxes: FloatTensor [N, 4]
        if len(boxes_t) > 0:
            boxes_tensor = torch.as_tensor(boxes_t, dtype=torch.float32).view(-1, 4)
            labels_tensor = torch.as_tensor(labels_t, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
                boxes_tensor[:, 2] - boxes_tensor[:, 0]
            )
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])  # Unique integer ID for evaluator
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes_t),), dtype=torch.int64)

        # Study Label
        if self.split != "test":
            study_label = int(row["study_label"])
            target["study_label"] = torch.tensor(study_label, dtype=torch.int64)
        else:
            # Dummy label for test set
            target["study_label"] = torch.tensor(-1, dtype=torch.int64)

        # Return format: image, target, image_id_string
        return image_t, target, str(image_id)
