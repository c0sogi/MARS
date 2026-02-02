import os
import ast
import cv2
import torch
import pydicom
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import get_transforms


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Handles DICOM loading, metadata parsing, and caching.
    """

    def __init__(self, split, load_cached_data=True, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform

        # Determine metadata source based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load data (with caching mechanism)
        self.df = self._load_data(load_cached_data)

        # Apply Debug Slicing
        if Config.DEBUG:
            original_len = len(self.df)
            self.df = self.df.iloc[: Config.DEBUG_DATA_SIZE]
            print(
                f"[{self.split.upper()}] DEBUG Mode: Reduced dataset from {original_len} to {len(self.df)} samples."
            )

    def _load_data(self, load_cached_data):
        """
        Loads metadata dataframe. Implements strict caching logic for deterministic processing.
        """
        # Ensure working directory exists (Config does this, but being safe)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_file = os.path.join(Config.WORKING_DIR, f"cached_{self.split}_df.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                return df
            except Exception as e:
                print(
                    f"[{self.split.upper()}] Cache load failed: {e}. Re-processing data."
                )

        # 2. Process from scratch
        df = pd.read_csv(self.metadata_path)

        # Preprocess Study Labels for Train/Val splits
        # Test split does not have these columns
        if self.split in ["train", "val"]:
            label_cols = Config.STUDY_LABELS
            # Check if columns exist
            if all(col in df.columns for col in label_cols):
                # Convert one-hot encoded columns to a single integer index
                df["study_label_idx"] = df[label_cols].values.argmax(axis=1)
            else:
                # Fallback or error if labels are missing in labeled splits
                print(
                    f"[{self.split.upper()}] Warning: Study label columns missing. Assigning default 0."
                )
                df["study_label_idx"] = 0

        # 3. Save to cache
        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(
                f"[{self.split.upper()}] Warning: Failed to save cache to {cache_file}: {e}"
            )

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            # --- 1. Load Image ---
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            dcm = pydicom.dcmread(file_path)

            # Extract pixel array
            try:
                pixel_array = dcm.pixel_array.astype(np.float32)
            except Exception:
                # Fallback to rasterio for JPEG Lossless DICOMs where pydicom plugins are missing
                # Cite debug_lesson_6: Verify Codec Compatibility
                # Cite debug_lesson_5: Avoid Using TensorFlow Inside PyTorch DataLoader Workers
                with rasterio.open(file_path) as src:
                    pixel_array = src.read().astype(np.float32)
                    # rasterio reads as (C, H, W), pydicom as (H, W) for single channel
                    if pixel_array.shape[0] == 1:
                        pixel_array = pixel_array[0]
                    else:
                        pixel_array = np.transpose(pixel_array, (1, 2, 0))

            # Handle Photometric Interpretation
            # MONOCHROME1: 0 is white, high is black. We want standard (0=black).
            photometric_interpretation = getattr(
                dcm, "PhotometricInterpretation", "MONOCHROME2"
            )
            if photometric_interpretation == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

            # Normalize to 0-255
            p_min = np.min(pixel_array)
            p_max = np.max(pixel_array)
            if p_max > p_min:
                img = (pixel_array - p_min) / (p_max - p_min) * 255.0
            else:
                img = np.zeros_like(pixel_array)

            img = img.astype(np.uint8)

            # Convert to RGB (3 channels) for ResNet backbone
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]

            # --- 2. Prepare Target ---
            # Default empty targets
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            study_label = 0

            if self.split in ["train", "val"]:
                # Get Study Label
                study_label = row["study_label_idx"]

                # Get Boxes
                # 'boxes' column is a string representation of a list of dicts or NaN
                boxes_str = row.get("boxes", np.nan)

                if pd.notna(boxes_str) and boxes_str != "nan":
                    try:
                        box_list = ast.literal_eval(boxes_str)
                        if len(box_list) > 0:
                            parsed_boxes = []
                            for b in box_list:
                                x_min = float(b["x"])
                                y_min = float(b["y"])
                                w_box = float(b["width"])
                                h_box = float(b["height"])
                                x_max = x_min + w_box
                                y_max = y_min + h_box
                                parsed_boxes.append([x_min, y_min, x_max, y_max])

                            boxes = np.array(parsed_boxes, dtype=np.float32)
                            # Class label is always 0 for 'opacity'
                            labels = np.zeros((len(boxes),), dtype=np.int64)
                    except Exception:
                        # Fallback to empty if parsing fails
                        pass

            target = {
                "boxes": boxes,
                "labels": labels,
                "study_label": study_label,  # Passed as int/numpy int, converted by ToTensor
                "image_id": row["image_id"],
                "study_id": row["study_id"],
                "orig_size": np.array([h, w]),
            }

            # --- 3. Apply Transforms ---
            if self.transform:
                img, target = self.transform(img, target)

            return img, target

        except Exception as e:
            # Return None to be filtered by collate_fn
            print(f"Error loading sample {idx}: {e}")
            return None
