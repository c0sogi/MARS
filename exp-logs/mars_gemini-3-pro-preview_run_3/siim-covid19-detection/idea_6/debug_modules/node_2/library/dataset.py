import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from library.config import Config


def process_dicom(file_path):
    """
    Reads a DICOM file, handles photometric interpretation, and normalizes to uint8.

    Args:
        file_path (str): Relative path to the DICOM file.

    Returns:
        np.ndarray: 2D numpy array (uint8) of the image.
    """
    try:
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        dcm = pydicom.dcmread(full_path)

        pixel_array = dcm.pixel_array

        # Handle Photometric Interpretation (Invert if MONOCHROME1)
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = np.amax(pixel_array) - pixel_array

        # Normalize to [0, 255]
        pixel_array = pixel_array.astype(np.float32)
        p_min = pixel_array.min()
        p_max = pixel_array.max()

        if p_max > p_min:
            pixel_array = (pixel_array - p_min) / (p_max - p_min)
            pixel_array = (pixel_array * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        return pixel_array
    except Exception as e:
        # Return a blank image in case of error
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)


def save_image_npy(args):
    """
    Worker function to process and save a single image as .npy.
    """
    image_id, file_path, save_dir = args
    save_path = os.path.join(save_dir, f"{image_id}.npy")

    # Process and save
    img = process_dicom(file_path)
    np.save(save_path, img)


def load_image_npy(args):
    """
    Worker function to load a single .npy image.
    """
    image_id, load_dir = args
    load_path = os.path.join(load_dir, f"{image_id}.npy")
    try:
        # Allow pickle=False for security and strict compliance
        return image_id, np.load(load_path, allow_pickle=False)
    except:
        return image_id, np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)


class CovidDataset(Dataset):
    def __init__(self, df, transforms=None, split="train", load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transformations pipeline.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to use cached .npy files.
                                     If False, regenerates cache.
        """
        self.df = df
        self.transforms = transforms
        self.split = split
        self.study_classes = Config.STUDY_CLASSES

        # Directory for caching individual numpy images
        self.cache_dir = os.path.join(Config.CACHE_DIR, "npy_images")
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- Caching Logic ---
        expected_ids = self.df["image_id"].unique()
        missing_ids = []

        if load_cached_data:
            # Check which files are missing
            for img_id in expected_ids:
                if not os.path.exists(os.path.join(self.cache_dir, f"{img_id}.npy")):
                    missing_ids.append(img_id)
        else:
            # Force regeneration of all files
            missing_ids = list(expected_ids)

        if len(missing_ids) > 0:
            print(f"[{split}] Generating cache for {len(missing_ids)} images...")

            # Prepare tasks
            # We need file paths for the missing IDs
            missing_df = self.df[self.df["image_id"].isin(missing_ids)].drop_duplicates(
                "image_id"
            )
            tasks = []
            for _, row in missing_df.iterrows():
                tasks.append((row["image_id"], row["file_path"], self.cache_dir))

            # Execute in parallel
            with ThreadPoolExecutor(
                max_workers=Config.NUM_WORKERS if Config.NUM_WORKERS > 0 else 4
            ) as executor:
                list(executor.map(save_image_npy, tasks))

        # --- Loading Logic ---
        # Load all images into RAM for fast access during training
        print(f"[{split}] Loading {len(expected_ids)} images into memory...")
        self.images = {}
        load_tasks = [(img_id, self.cache_dir) for img_id in expected_ids]

        with ThreadPoolExecutor(
            max_workers=Config.NUM_WORKERS if Config.NUM_WORKERS > 0 else 4
        ) as executor:
            results = list(executor.map(load_image_npy, load_tasks))

        for img_id, img_arr in results:
            self.images[img_id] = img_arr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Get Image
        image = self.images.get(image_id)
        if image is None:
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

        # Store original dimensions (before transform)
        h_orig, w_orig = image.shape[:2]

        # 2. Parse Targets (if not test)
        boxes = []
        labels = []
        study_label = 0

        if self.split != "test":
            # Study Label
            for i, col in enumerate(self.study_classes):
                if row[col] == 1:
                    study_label = i
                    break

            # Bounding Boxes
            if pd.notna(row["boxes"]):
                try:
                    # Parse string representation of list of dicts
                    box_data = ast.literal_eval(row["boxes"])
                    for b in box_data:
                        xmin = float(b["x"])
                        ymin = float(b["y"])
                        w = float(b["width"])
                        h = float(b["height"])
                        xmax = xmin + w
                        ymax = ymin + h
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(1)  # Class 1: Opacity
                except:
                    pass

        # Convert to numpy for Albumentations
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # 3. Apply Transforms
        if self.transforms:
            # Expand to 3 channels (H, W) -> (H, W, 3)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
                image = np.repeat(image, 3, axis=-1)

            if len(boxes) > 0:
                transformed = self.transforms(
                    image=image, bboxes=boxes, class_labels=labels
                )
                image = transformed["image"]
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)
            else:
                transformed = self.transforms(image=image, bboxes=[], class_labels=[])
                image = transformed["image"]
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Fallback (should not happen in main pipeline)
            image = torch.from_numpy(image).float()
            if image.ndim == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            else:
                image = image.permute(2, 0, 1)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # 4. Construct Target Dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["study_label"] = torch.tensor(study_label, dtype=torch.int64)
        target["image_id"] = image_id
        target["orig_size"] = torch.tensor([h_orig, w_orig])

        return image, target, image_id
