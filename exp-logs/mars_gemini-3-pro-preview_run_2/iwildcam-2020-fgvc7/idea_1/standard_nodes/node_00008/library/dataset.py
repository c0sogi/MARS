import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from library.config import Config
from library.utils import load_megadetector_data, get_crop_coordinates


class CameraTrapDataset(Dataset):
    """
    PyTorch Dataset for iWildCam 2020 species classification.
    Handles loading images, cropping based on MegaDetector results, and applying transformations.
    """

    def __init__(self, split, transform=None, load_cached_data=True, sample_size=None):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to load cached MegaDetector results.
            sample_size (int, optional): Limit the dataset size for debugging.
        """
        self.split = split
        self.transform = transform

        # 1. Load Metadata
        if split == "train":
            self.metadata_path = Config.TRAIN_CSV
        elif split == "val":
            self.metadata_path = Config.VAL_CSV
        elif split == "test":
            self.metadata_path = Config.TEST_CSV
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Optional: Subsample for debugging
        if sample_size is not None and sample_size > 0:
            if sample_size < len(self.df):
                # Use fixed random state for reproducibility
                self.df = self.df.sample(
                    n=sample_size, random_state=Config.SEED
                ).reset_index(drop=True)

        # 2. Load MegaDetector Results
        # This dictionary maps image_id -> {'bbox': [x, y, w, h], 'conf': float}
        self.detections = load_megadetector_data(
            json_path=Config.MEGADETECTOR_JSON, load_cached_data=load_cached_data
        )

        # 3. Define Default Transforms (if none provided)
        if self.transform is None:
            if self.split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(Config.IMG_SIZE),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(Config.IMG_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, target) where target is category_id (int) for train/val,
                   or image_id (str) for test.
        """
        row = self.df.iloc[idx]
        img_id = row["id"]

        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Load Image
        # cv2.imread returns BGR or None
        img = cv2.imread(img_path)

        if img is None:
            # Fallback for missing/corrupt images: return black image
            # This prevents the dataloader from crashing
            h, w = Config.IMG_SIZE
            img = np.zeros((h, w, 3), dtype=np.uint8)

        # Get original dimensions
        h_orig, w_orig = img.shape[:2]

        # 2. Crop based on MegaDetector
        detection_info = self.detections.get(img_id)

        # get_crop_coordinates returns absolute (x_min, y_min, x_max, y_max)
        # It handles cases where detection is None or low confidence (returns full image)
        x_min, y_min, x_max, y_max = get_crop_coordinates(
            w_orig, h_orig, detection_info, conf_threshold=0.0
        )

        # Apply crop
        # Numpy slicing: [y:y+h, x:x+w]
        crop = img[y_min:y_max, x_min:x_max]

        # Safety check: if crop is empty (should be handled by utils, but double check)
        if crop.size == 0:
            crop = img

        # 3. Preprocessing
        # Convert BGR (OpenCV) to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Apply Transforms (PIL -> Resize -> Tensor -> Normalize)
        if self.transform:
            image_tensor = self.transform(crop_rgb)
        else:
            image_tensor = transforms.ToTensor()(crop_rgb)

        # 4. Return Data
        if self.split in ["train", "val"]:
            # Return label
            label = int(row["category_id"])
            return image_tensor, label
        else:
            # Return ID for submission generation
            return image_tensor, img_id
