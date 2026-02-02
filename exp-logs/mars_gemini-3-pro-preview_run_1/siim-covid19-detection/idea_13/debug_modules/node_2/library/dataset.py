import os
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(split="train"):
    """
    Returns the Albumentations transformations for the specific split.
    Includes CoarseDropout for training with mask_fill_value=0 to prevent label noise.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Handles DICOM loading, mask generation, and caching.
    """

    def __init__(self, split, load_cached_data=True, transform=None):
        self.split = split
        self.transform = transform
        self.label_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]

        # Determine metadata path
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = pd.read_csv(self.metadata_path)

        # Load or create data
        self.images, self.masks, self.labels, self.ids = self._process_and_cache(
            load_cached_data
        )

    def _process_and_cache(self, load_cached_data):
        """
        Implements the strict caching logic:
        1. Try load from .npy
        2. Else process from scratch and save.
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        img_path = os.path.join(cache_dir, f"{self.split}_images.npy")
        mask_path = os.path.join(cache_dir, f"{self.split}_masks.npy")
        label_path = os.path.join(cache_dir, f"{self.split}_labels.npy")
        id_path = os.path.join(cache_dir, f"{self.split}_ids.npy")

        # 1. Try to load
        if load_cached_data and os.path.exists(img_path) and os.path.exists(id_path):
            # For test set, masks and labels might not exist or be relevant, but we handle consistency
            try:
                images = np.load(img_path)
                ids = np.load(id_path)

                if self.split != "test":
                    masks = np.load(mask_path)
                    labels = np.load(label_path)
                else:
                    masks = None
                    labels = None

                return images, masks, labels, ids
            except Exception:
                # If load fails, fall through to processing
                pass

        # 2. Process from scratch
        images_list = []
        masks_list = []
        labels_list = []
        ids_list = []

        for idx, row in self.df.iterrows():
            # --- Image Processing ---
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                dcm = pydicom.dcmread(file_path)
                img = dcm.pixel_array

                # Handle Photometric Interpretation
                if (
                    hasattr(dcm, "PhotometricInterpretation")
                    and dcm.PhotometricInterpretation == "MONOCHROME1"
                ):
                    img = np.max(img) - img

                # Normalize to 0-255
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img = (img * 255).astype(np.uint8)

                # Resize image
                img_resized = cv2.resize(
                    img,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                # Convert to 3 channels
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

                orig_h, orig_w = dcm.Rows, dcm.Columns

            except Exception as e:
                # Fallback for corrupt images (should be rare given EDA)
                img_resized = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )
                orig_h, orig_w = 1024, 1024

            images_list.append(img_resized)

            # Use study_id for study-level predictions, image_id for image-level
            # The submission format requires specific IDs.
            # For training, we track the study ID to map to labels.
            ids_list.append(row["study_id"])

            # --- Mask & Label Processing (Train/Val only) ---
            if self.split != "test":
                # Labels: Argmax of the one-hot columns
                label_vec = row[self.label_cols].values.astype(np.float32)
                label_idx = np.argmax(label_vec)
                labels_list.append(label_idx)

                # Masks
                mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                boxes_str = row.get("boxes", np.nan)

                if pd.notna(boxes_str):
                    try:
                        boxes = ast.literal_eval(boxes_str)
                        for box in boxes:
                            x = int(box["x"])
                            y = int(box["y"])
                            w = int(box["width"])
                            h = int(box["height"])
                            mask[y : y + h, x : x + w] = 1.0
                    except:
                        pass

                # Resize mask
                mask_resized = cv2.resize(
                    mask,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                masks_list.append(mask_resized)

        # Convert to numpy arrays
        images_np = np.array(images_list, dtype=np.uint8)
        ids_np = np.array(ids_list)

        if self.split != "test":
            masks_np = np.array(masks_list, dtype=np.float32)
            labels_np = np.array(labels_list, dtype=np.int64)
        else:
            masks_np = np.empty(0)  # Placeholder
            labels_np = np.empty(0)

        # Save to cache
        np.save(img_path, images_np)
        np.save(id_path, ids_np)
        if self.split != "test":
            np.save(mask_path, masks_np)
            np.save(label_path, labels_np)

        return images_np, masks_np, labels_np, ids_np

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.split != "test":
            mask = self.masks[idx]
            label = self.labels[idx]

            if self.transform:
                # Albumentations expects mask to be passed if present
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask has channel dimension (1, H, W)
            # Albumentations ToTensorV2 doesn't add channel dim to mask automatically if it's 2D
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return {
                "image": image,
                "mask": mask.float(),
                "label": torch.tensor(label, dtype=torch.long),
                "study_id": self.ids[idx],
            }
        else:
            # Test set
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

            return {
                "image": image,
                "study_id": self.ids[idx],
                # Pass original image ID for submission mapping if needed,
                # but study_id is primary for dataset alignment.
                # We can retrieve image_id from df if strictly necessary,
                # but 'ids' cache stores study_id.
                "image_id": self.df.iloc[idx]["image_id"],
            }
