import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import library.config as config
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    IMG_SIZE,
    STUDY_LABEL_TO_ID,
    DEBUG,
)
from library.utils import get_train_transforms, get_valid_transforms


class CovidDataset(Dataset):
    def __init__(
        self,
        mode="train",
        transforms=None,
        load_cached_data=True,
        debug=DEBUG,
    ):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transforms (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to load processed metadata from cache.
            debug (bool): If True, subsamples the dataset for debugging.
        """
        self.mode = mode
        self.transforms = transforms
        self.debug = debug

        # Determine paths
        if self.mode == "train":
            self.csv_path = TRAIN_META_PATH
        elif self.mode == "val":
            self.csv_path = VAL_META_PATH
        elif self.mode == "test":
            self.csv_path = TEST_META_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load and process metadata
        self.df = self._load_metadata(load_cached_data)

        # Debug subsampling
        if self.debug:
            self.df = self.df.iloc[: config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
            print(f"[{self.mode.upper()}] Debug mode: Sampled {len(self.df)} images.")

    def _load_metadata(self, load_cached_data):
        """
        Loads metadata, processing boxes and labels, with caching.
        """
        cache_path = os.path.join(WORKING_DIR, f"cached_{self.mode}_metadata.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # Ensure boxes are lists (parquet might save as array or list)
                # If reading from parquet, object columns might need validation,
                # but usually parquet preserves types well.
                return df
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}. Re-processing.")

        # 2. Process from scratch
        df = pd.read_csv(self.csv_path)

        # Process Study Labels (Train/Val only)
        if self.mode in ["train", "val"]:
            # Map one-hot columns to single integer ID
            # Columns: 'Negative for Pneumonia', 'Typical Appearance',
            # 'Indeterminate Appearance', 'Atypical Appearance'

            def get_study_label(row):
                for label_name, label_id in STUDY_LABEL_TO_ID.items():
                    if row.get(label_name, 0) == 1:
                        return label_id
                return 0  # Default to Negative/Background if not found (should not happen in clean data)

            df["study_label_id"] = df.apply(get_study_label, axis=1)

            # Process Bounding Boxes
            # Format in CSV: [{'x': 789.28836, 'y': 582.43035, 'width': 1026.65662, 'height': 1917.30292}]
            # or NaN

            def process_boxes(box_str):
                if pd.isna(box_str):
                    return []
                try:
                    boxes = ast.literal_eval(box_str)
                    clean_boxes = []
                    for b in boxes:
                        # Convert xywh to xyxy (Pascal VOC)
                        x_min = float(b["x"])
                        y_min = float(b["y"])
                        w = float(b["width"])
                        h = float(b["height"])
                        x_max = x_min + w
                        y_max = y_min + h
                        clean_boxes.append([x_min, y_min, x_max, y_max])
                    return clean_boxes
                except:
                    return []

            df["parsed_boxes"] = df["boxes"].apply(process_boxes)
        else:
            # Test mode: placeholders
            df["study_label_id"] = 0
            df["parsed_boxes"] = df.apply(lambda x: [], axis=1)

        # 3. Save to cache
        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return df

    def _read_dicom(self, path):
        """
        Reads a DICOM file and converts it to a standard RGB numpy array (0-255).
        """
        full_path = os.path.join("./input", path)
        if not os.path.exists(full_path):
            # Fallback: create a black image if file missing (should not happen with verified metadata)
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        image = None

        # Try pydicom
        try:
            dcm = pydicom.dcmread(full_path)
            image = dcm.pixel_array

            # Handle Photometric Interpretation (Invert MONOCHROME1 to make bones white/air black is standard,
            # but usually for opacity detection we want dense=white.
            # MONOCHROME1: 0=White. MONOCHROME2: 0=Black.
            # X-rays: Air is black (low density), Bone/Opacity is white (high density).
            # If MONOCHROME1 (0=White), then Air (low val) is White. We need to invert.
            if (
                "PhotometricInterpretation" in dcm
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                image = np.max(image) - image
        except:
            pass

        # Try OpenCV fallback
        if image is None:
            try:
                image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            except:
                pass

        if image is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Normalize to 0-255
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image = image - np.min(image)
            max_val = np.max(image)
            if max_val > 0:
                image = image / max_val * 255.0
            image = image.astype(np.uint8)

        # Convert to RGB (Albumentations expects 3 channels usually)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)

        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load Image
        image = self._read_dicom(row["file_path"])

        # Get Targets
        boxes = []
        labels = []

        if self.mode in ["train", "val"]:
            raw_boxes = row["parsed_boxes"]
            # Parquet might load lists as numpy arrays, ensure list
            if isinstance(raw_boxes, np.ndarray):
                raw_boxes = raw_boxes.tolist()

            study_label = int(row["study_label_id"])

            if len(raw_boxes) > 0:
                boxes = np.array(raw_boxes, dtype=np.float32)
                # Assign the study label to all boxes (Granular Supervision)
                # If study is Negative (0), there should be no boxes.
                # If study is Typical (1), boxes are 1.
                labels = np.full(len(boxes), study_label, dtype=np.int64)
            else:
                # Negative case or no boxes found
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty((0,), dtype=np.int64)
        else:
            # Test mode
            study_label = 0  # Dummy
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        # Apply Transforms
        if self.transforms:
            # Albumentations requires labels for bbox_params
            # If no boxes, we still pass the empty lists
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert to Torch Tensors
        # Image is already tensor from ToTensorV2

        # Boxes
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["study_label"] = torch.tensor(study_label, dtype=torch.int64)

        return image, target, image_id
