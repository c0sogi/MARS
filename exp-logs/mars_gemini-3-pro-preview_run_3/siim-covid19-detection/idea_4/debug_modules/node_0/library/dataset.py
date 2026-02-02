import os
import cv2
import pydicom
import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import get_transforms


class CovidDataset(Dataset):
    """
    PyTorch Dataset for COVID-19 Radiograph Detection and Classification.
    """

    def __init__(self, phase: str, transform=None, load_cached_data=True):
        """
        Args:
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
            load_cached_data (bool): Whether to load parsed metadata from cache.
        """
        self.phase = phase
        self.transform = transform or get_transforms(phase)
        self.input_dir = Config.INPUT_DIR

        # Determine which metadata file to use
        if self.phase == "train":
            self.meta_path = Config.TRAIN_META_PATH
        elif self.phase == "val":
            self.meta_path = Config.VAL_META_PATH
        elif self.phase == "test":
            self.meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        self.data = self._load_metadata(load_cached_data)

    def _load_metadata(self, load_cached_data):
        """
        Loads metadata, parses bounding boxes and labels.
        Implements caching using .npy files.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        cache_file = os.path.join(Config.CACHE_DIR, f"{self.phase}_data.npy")

        if load_cached_data and os.path.exists(cache_file):
            try:
                # Using allow_pickle=True to load the list of dictionaries
                data = np.load(cache_file, allow_pickle=True)
                return data.tolist()
            except Exception as e:
                print(f"Failed to load cache for {self.phase}: {e}. Recomputing...")

        # Load from CSV
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        df = pd.read_csv(self.meta_path)
        data_list = []

        # Iterate over rows to parse data
        for _, row in df.iterrows():
            entry = {
                "image_id": str(row["image_id"]),
                "study_id": str(row["study_id"]),
                "file_path": str(row["file_path"]),
            }

            if self.phase in ["train", "val"]:
                # 1. Parse Study Label
                # Default to 0 (Negative) if not found, though dataset should be complete
                study_label_idx = 0
                for idx, col in enumerate(Config.STUDY_CLASSES):
                    if col in row and row[col] == 1:
                        study_label_idx = idx
                        break
                entry["study_label"] = study_label_idx

                # 2. Parse Bounding Boxes
                boxes = []
                labels = []

                # 'boxes' column contains string representation of list of dicts
                if "boxes" in row and pd.notna(row["boxes"]):
                    try:
                        # ast.literal_eval is safer than eval
                        box_list = ast.literal_eval(row["boxes"])
                        for box in box_list:
                            # Original format: x, y, width, height
                            x = float(box["x"])
                            y = float(box["y"])
                            w = float(box["width"])
                            h = float(box["height"])

                            # Convert to Pascal VOC: x_min, y_min, x_max, y_max
                            x_min = x
                            y_min = y
                            x_max = x + w
                            y_max = y + h

                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # Class 1 is 'opacity'
                    except (ValueError, SyntaxError):
                        # Handle malformed strings if any
                        pass

                entry["boxes"] = np.array(boxes, dtype=np.float32)
                entry["labels"] = np.array(labels, dtype=np.int64)
            else:
                # Test phase: No targets
                entry["study_label"] = -1
                entry["boxes"] = np.zeros((0, 4), dtype=np.float32)
                entry["labels"] = np.zeros((0,), dtype=np.int64)

            data_list.append(entry)

        # Save to cache
        try:
            np.save(cache_file, np.array(data_list))
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_file}: {e}")

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # 1. Load DICOM Image
        dicom_path = os.path.join(self.input_dir, entry["file_path"])

        try:
            dcm = pydicom.dcmread(dicom_path)
            image = dcm.pixel_array.astype(np.float32)

            # Fix Photometric Interpretation (Monochrome1 means 0 is white)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                image = np.max(image) - image

            # Normalize to 0-255 range
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min) * 255.0
            else:
                image = np.zeros_like(image)

            image = image.astype(np.uint8)

            # Convert to 3 channels (RGB) for backbone compatibility
            image = np.stack([image] * 3, axis=-1)

        except Exception as e:
            # Fallback for corrupt images
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # 2. Prepare Data for Transforms
        boxes = entry["boxes"]
        labels = entry["labels"]

        # Albumentations expects lists
        if len(boxes) > 0:
            boxes_list = boxes.tolist()
            labels_list = labels.tolist()
        else:
            boxes_list = []
            labels_list = []

        # 3. Apply Transforms
        # The transform pipeline handles resizing (Letterbox) and augmentation
        transformed = self.transform(image=image, bboxes=boxes_list, labels=labels_list)

        image_tensor = transformed["image"]

        # Retrieve transformed boxes
        boxes_transformed = transformed["bboxes"]
        labels_transformed = transformed["labels"]

        if len(boxes_transformed) > 0:
            boxes_tensor = torch.tensor(boxes_transformed, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_transformed, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # 4. Construct Target Dictionary
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "study_label": torch.tensor(entry["study_label"], dtype=torch.int64),
            "image_id": entry["image_id"],
            "study_id": entry["study_id"],
        }

        return image_tensor, target, entry["image_id"]
