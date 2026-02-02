import os
import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom, get_train_transforms, get_valid_transforms


def load_dataset_metadata(split, load_cached_data=True):
    """
    Loads dataset metadata from CSV or Parquet cache.
    Parses bounding boxes and encodes study labels.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'study_label' and 'parsed_boxes'.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"cached_{split}_df.parquet")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Cache load failed for {split}: {e}. Reloading from CSV.")

    # 2. Load from CSV
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # 3. Process Data (if not test)
    if split != "test":
        # Map Study Labels
        # Negative: 0, Typical: 1, Indeterminate: 2, Atypical: 3
        label_map = {
            "Negative for Pneumonia": 0,
            "Typical Appearance": 1,
            "Indeterminate Appearance": 2,
            "Atypical Appearance": 3,
        }

        def get_label(row):
            for col, idx in label_map.items():
                if row.get(col, 0) == 1:
                    return idx
            return 0

        df["study_label"] = df.apply(get_label, axis=1)

        # Parse Bounding Boxes
        # Input format: "[{'x': 10, 'y': 20, 'width': 100, 'height': 50}, ...]"
        # Output format: [[x, y, w, h], ...]
        def parse_boxes(box_str):
            if pd.isna(box_str):
                return []
            try:
                # Handle cases where it might already be a list (if reloading partially processed df)
                if not isinstance(box_str, str):
                    return []

                dicts = ast.literal_eval(box_str)
                boxes = []
                for d in dicts:
                    boxes.append([d["x"], d["y"], d["width"], d["height"]])
                return boxes
            except:
                return []

        df["parsed_boxes"] = df["boxes"].apply(parse_boxes)
    else:
        # For test set, add placeholder columns for consistency
        df["study_label"] = 0
        df["parsed_boxes"] = [[] for _ in range(len(df))]

    # 4. Save to Cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return df


class CovidDataset(Dataset):
    def __init__(self, split, transform=None, load_cached_data=True, debug=False):
        """
        Dataset for COVID-19 Radiography.

        Args:
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform.
            load_cached_data (bool): Whether to use cached metadata.
            debug (bool): If True, use a small subset.
        """
        self.split = split
        self.df = load_dataset_metadata(split, load_cached_data)

        if debug:
            self.df = self.df.sample(
                n=min(len(self.df), 50), random_state=Config.SEED
            ).reset_index(drop=True)

        if transform is None:
            if split == "train":
                self.transform = get_train_transforms()
            else:
                self.transform = get_valid_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        # Load Image
        img = load_dicom(file_path)
        h, w = img.shape[:2]

        # Get Targets
        study_label = row["study_label"]
        raw_boxes = row["parsed_boxes"]

        # Ensure raw_boxes is a list of lists (handle parquet loading artifacts)
        if isinstance(raw_boxes, np.ndarray):
            raw_boxes = raw_boxes.tolist()
        if not isinstance(raw_boxes, list):
            raw_boxes = []

        # Convert [x, y, w, h] to [x_min, y_min, x_max, y_max] for Albumentations
        boxes_pascal = []
        labels = []

        for b in raw_boxes:
            bx, by, bw, bh = b
            x_min = bx
            y_min = by
            x_max = bx + bw
            y_max = by + bh

            # Clip
            x_min = max(0.0, min(x_min, float(w)))
            y_min = max(0.0, min(y_min, float(h)))
            x_max = max(0.0, min(x_max, float(w)))
            y_max = max(0.0, min(y_max, float(h)))

            if x_max > x_min and y_max > y_min:
                boxes_pascal.append([x_min, y_min, x_max, y_max])
                labels.append(0)  # Class 0 for opacity

        # Apply Transforms
        if self.transform:
            # Albumentations requires bboxes arg even if empty
            transformed = self.transform(
                image=img, bboxes=boxes_pascal, class_labels=labels
            )
            img_tensor = transformed["image"]
            boxes_aug = transformed["bboxes"]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            boxes_aug = boxes_pascal

        # Format Targets for Model (DINO/DETR)
        # Boxes: (cx, cy, w, h) normalized [0, 1]
        final_boxes = []
        final_labels = []

        img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]

        for box in boxes_aug:
            x_min, y_min, x_max, y_max = box

            cx = (x_min + x_max) / 2.0 / img_w
            cy = (y_min + y_max) / 2.0 / img_h
            bw = (x_max - x_min) / img_w
            bh = (y_max - y_min) / img_h

            # Clamp to [0, 1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            final_boxes.append([cx, cy, bw, bh])
            final_labels.append(0)

        target = {
            "boxes": torch.as_tensor(final_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(final_labels, dtype=torch.int64),
            "study_labels": torch.as_tensor([study_label], dtype=torch.int64),
            "image_id": image_id,
            "orig_size": torch.as_tensor([h, w], dtype=torch.int64),
            "size": torch.as_tensor([img_h, img_w], dtype=torch.int64),
        }

        return img_tensor, target, image_id
