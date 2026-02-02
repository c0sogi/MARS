import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_transforms


class VinDrDataset(Dataset):
    def __init__(self, mode="train", dataset_fraction=1.0, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            dataset_fraction (float): Fraction of data to use (0.0 to 1.0).
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.mode = mode
        self.dataset_fraction = dataset_fraction

        # Determine metadata source
        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA
        elif self.mode == "test":
            self.metadata_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load and process metadata (with caching)
        self.df = self._prepare_data(load_cached_data)

        # Initialize transforms
        self.transforms = get_transforms(train=(self.mode == "train"))

    def _prepare_data(self, load_cached_data):
        """
        Loads metadata, groups it by image_id, and manages caching.
        """
        # Construct a unique cache filename based on parameters
        cache_filename = f"cached_{self.mode}_{self.dataset_fraction}.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df_grouped = pd.read_parquet(cache_path)
                return df_grouped
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # 2. Compute from scratch
        df = pd.read_csv(self.metadata_path)

        # Filter dataset by fraction (stratified by image_id)
        if self.dataset_fraction < 1.0:
            unique_ids = df["image_id"].unique()
            n_keep = int(len(unique_ids) * self.dataset_fraction)
            # Use fixed seed for reproducibility of the split
            rng = np.random.RandomState(Config.SEED)
            keep_ids = rng.choice(unique_ids, n_keep, replace=False)
            df = df[df["image_id"].isin(keep_ids)].copy()

        # Group data by image_id
        if self.mode in ["train", "val"]:
            grouped_rows = []
            # Group by image_id to aggregate boxes and labels
            for img_id, group in df.groupby("image_id"):
                file_path = group.iloc[0]["file_path"]

                boxes = []
                labels = []

                for _, row in group.iterrows():
                    cid = int(row["class_id"])
                    # Class 14 is "No finding". It implies background (no boxes).
                    # We only add boxes for classes 0-13.
                    if cid != 14:
                        # PASCAL VOC format: [xmin, ymin, xmax, ymax]
                        b = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
                        boxes.append(b)
                        labels.append(cid)

                grouped_rows.append(
                    {
                        "image_id": img_id,
                        "file_path": file_path,
                        "boxes": boxes,
                        "labels": labels,
                    }
                )

            df_grouped = pd.DataFrame(grouped_rows)
        else:
            # Test mode: No ground truth boxes needed
            # Just ensure one row per image
            df_grouped = df[["image_id", "file_path"]].drop_duplicates()
            # Add empty columns for consistency
            df_grouped["boxes"] = [[] for _ in range(len(df_grouped))]
            df_grouped["labels"] = [[] for _ in range(len(df_grouped))]

        # 3. Save to cache
        try:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            df_grouped.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return df_grouped

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata for this index
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        rel_path = row["file_path"]

        # Load Image
        dicom_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            dicom = pydicom.dcmread(dicom_path)
            image = dicom.pixel_array

            # Fix Photometric Interpretation (MONOCHROME1 -> MONOCHROME2)
            if (
                hasattr(dicom, "PhotometricInterpretation")
                and dicom.PhotometricInterpretation == "MONOCHROME1"
            ):
                image = np.amax(image) - image

            # Normalize to [0, 255] uint8
            image = image.astype(np.float32)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min) * 255.0
            else:
                image = np.zeros_like(image)
            image = image.astype(np.uint8)

            # Convert to RGB (3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        except Exception as e:
            # Fallback for corrupt files
            print(f"Error loading {dicom_path}: {e}")
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)

        # Store original dimensions for scaling predictions later (if needed)
        original_h, original_w = image.shape[:2]

        # Prepare Targets for Transform
        boxes = (
            np.array(row["boxes"], dtype=np.float32)
            if len(row["boxes"]) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = (
            np.array(row["labels"], dtype=np.int64)
            if len(row["labels"]) > 0
            else np.zeros((0,), dtype=np.int64)
        )

        # Apply Albumentations
        # Note: Albumentations handles box resizing automatically
        if len(boxes) > 0:
            # Clip boxes to image boundaries
            boxes[:, 0] = np.clip(boxes[:, 0], 0, original_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, original_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, original_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, original_h)

            # Remove degenerate boxes
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        image_tensor = transformed["image"]

        # Prepare Final Target Dict
        target = {}

        # We need to return specific metadata for inference/submission
        target["image_id"] = image_id
        target["original_size"] = (original_h, original_w)

        if self.mode in ["train", "val"]:
            boxes_t = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels_t = torch.as_tensor(transformed["labels"], dtype=torch.int64)

            # Map Dataset Class ID (0-13) to Model Class ID (1-14)
            # 0 is reserved for Background in Faster R-CNN
            labels_t = labels_t + 1

            # Handle empty targets (No finding)
            if boxes_t.shape[0] == 0:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

            target["boxes"] = boxes_t
            target["labels"] = labels_t
            target["area"] = area
            target["iscrowd"] = torch.zeros((boxes_t.shape[0],), dtype=torch.uint8)
            # Note: 'image_id' in target for R-CNN is usually an int index,
            # but we stored the string ID above.
            # For training, we add a numeric ID for internal use if needed.
            target["image_id_int"] = torch.tensor([idx])

        return image_tensor, target
