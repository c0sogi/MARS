import os
import cv2
import pydicom
import torch
import numpy as np
import pandas as pd
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class ChestXrayDataset(Dataset):
    """
    Dataset class for Chest X-Ray Analysis.
    Handles DICOM loading, preprocessing, caching, and augmentation.
    """

    def __init__(
        self,
        metadata_df,
        mode="train",
        transform=None,
        cache_dir=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata (paths, labels, boxes).
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            cache_dir (str): Directory to store/load cached .npy files.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir if cache_dir else Config.WORKING_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.images_cache_path = os.path.join(self.cache_dir, f"{mode}_images.npy")
        self.masks_cache_path = os.path.join(self.cache_dir, f"{mode}_masks.npy")
        self.meta_cache_path = os.path.join(self.cache_dir, f"{mode}_dims.parquet")

        # Load or Process Data
        self.images = None
        self.masks = None
        self.dims_df = None

        if load_cached_data and self._check_cache_exists():
            print(f"[{self.mode.upper()}] Loading cached data from {self.cache_dir}...")
            self._load_cache()
        else:
            print(
                f"[{self.mode.upper()}] Processing data from scratch (Cache not found or disabled)..."
            )
            self._process_and_cache()

        # Parse labels for study level
        if self.mode != "test":
            self.labels = self.metadata_df[Config.STUDY_LABELS].values.astype(
                np.float32
            )
        else:
            self.labels = np.zeros(
                (len(self.metadata_df), Config.NUM_CLASSES_STUDY), dtype=np.float32
            )

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        # Test set doesn't need masks cache if we don't generate them,
        # but for consistency in structure we might generate empty ones or handle separately.
        # For this implementation, we generate placeholders for test.
        files = [self.images_cache_path, self.meta_cache_path]
        if self.mode != "test":
            files.append(self.masks_cache_path)
        return all(os.path.exists(f) for f in files)

    def _load_cache(self):
        """Loads data from .npy and .parquet files."""
        self.images = np.load(
            self.images_cache_path, mmap_mode="r"
        )  # Use mmap to save RAM if needed
        self.dims_df = pd.read_parquet(self.meta_cache_path)

        if self.mode != "test":
            self.masks = np.load(self.masks_cache_path, mmap_mode="r")
        else:
            # Create dummy masks for test set to simplify __getitem__
            self.masks = np.zeros_like(self.images)

    def _process_and_cache(self):
        """Reads DICOMs, resizes, generates masks, and saves to disk."""
        img_size = Config.IMAGE_SIZE

        processed_images = []
        processed_masks = []
        original_dims = []  # Store (h, w, scale_h, scale_w)

        total = len(self.metadata_df)

        for idx, row in self.metadata_df.iterrows():
            # 1. Read Image
            dicom_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                dcm = pydicom.dcmread(dicom_path)
                img = dcm.pixel_array

                # Handle Photometric Interpretation
                if (
                    "PhotometricInterpretation" in dcm
                    and dcm.PhotometricInterpretation == "MONOCHROME1"
                ):
                    img = np.amax(img) - img

                # Normalize to 0-255
                img = img.astype(np.float32)
                img = img - np.min(img)
                img = img / (np.max(img) + 1e-6)
                img = (img * 255).astype(np.uint8)

            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")
                # Fallback: create black image
                img = np.zeros(img_size, dtype=np.uint8)

            # 2. Record Dimensions and Resize
            h_orig, w_orig = img.shape
            img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            processed_images.append(img_resized)

            # Calculate scale factors
            scale_h = img_size[1] / h_orig
            scale_w = img_size[0] / w_orig
            original_dims.append(
                {
                    "orig_h": h_orig,
                    "orig_w": w_orig,
                    "scale_h": scale_h,
                    "scale_w": scale_w,
                }
            )

            # 3. Generate Mask (Train/Val only)
            if self.mode != "test":
                mask = np.zeros(img_size, dtype=np.uint8)
                box_str = row.get("boxes", np.nan)

                if pd.notna(box_str) and box_str != str(np.nan):
                    try:
                        boxes = ast.literal_eval(box_str)
                        for box in boxes:
                            # Box format: x, y, width, height
                            x, y, w, h = box["x"], box["y"], box["width"], box["height"]

                            # Scale to new size
                            x = int(x * scale_w)
                            y = int(y * scale_h)
                            w = int(w * scale_w)
                            h = int(h * scale_h)

                            # Draw on mask (1 for opacity)
                            cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
                    except:
                        pass  # No valid boxes

                processed_masks.append(mask)

        # Convert to arrays
        self.images = np.array(processed_images, dtype=np.uint8)

        if self.mode != "test":
            self.masks = np.array(processed_masks, dtype=np.uint8)
        else:
            self.masks = np.zeros_like(self.images)  # Dummy

        self.dims_df = pd.DataFrame(original_dims)

        # Save to cache
        print(f"Saving cache to {self.cache_dir}...")
        np.save(self.images_cache_path, self.images)
        if self.mode != "test":
            np.save(self.masks_cache_path, self.masks)
        self.dims_df.to_parquet(self.meta_cache_path)
        print("Cache saved.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        # If using mmap, accessing index reads from disk
        image = self.images[idx]  # (H, W)
        mask = self.masks[idx]  # (H, W)

        # Convert to 3 channels for backbone compatibility if needed,
        # or keep 1 channel. ResNet usually expects 3.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 2. Apply Augmentations
        if self.transform:
            # Albumentations expects mask to be passed if we want it transformed
            # CoarseDropout with mask_fill_value=0 will handle the consistency constraint
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Basic ToTensor if no transform provided
            converter = ToTensorV2()
            augmented = converter(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 3. Prepare Labels and Metadata
        study_label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Mask needs to be float for BCE loss, add channel dim: (1, H, W)
        mask = mask.float().unsqueeze(0)

        row = self.metadata_df.iloc[idx]
        study_id = row["study_id"]
        image_id = row["image_id"]

        # Retrieve original boxes for validation metric calculation
        # We need to return them scaled to the current image size (512x512)
        # because the model outputs predictions in 512x512 space.

        boxes = []
        if self.mode != "test":
            box_str = row.get("boxes", np.nan)
            scale_info = self.dims_df.iloc[idx]
            scale_w = scale_info["scale_w"]
            scale_h = scale_info["scale_h"]

            if pd.notna(box_str) and box_str != str(np.nan):
                try:
                    raw_boxes = ast.literal_eval(box_str)
                    for b in raw_boxes:
                        # x, y, w, h -> xmin, ymin, xmax, ymax
                        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                        x_min = x * scale_w
                        y_min = y * scale_h
                        x_max = (x + w) * scale_w
                        y_max = (y + h) * scale_h
                        boxes.append([x_min, y_min, x_max, y_max])
                except:
                    pass

        # Convert boxes to tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Create labels for boxes (all are class 0: opacity)
            box_labels = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            # No boxes
            boxes = torch.empty((0, 4), dtype=torch.float32)
            box_labels = torch.empty((0,), dtype=torch.int64)

        return {
            "image": image,
            "mask": mask,
            "study_label": study_label,
            "boxes": boxes,
            "box_labels": box_labels,
            "study_id": study_id,
            "image_id": image_id,
        }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # CoarseDropout: Crucial for learning robust features.
                # mask_fill_value=0 ensures label consistency (removes occluded opacities from mask).
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMAGE_SIZE[0] // 10,
                    max_width=Config.IMAGE_SIZE[1] // 10,
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
        # Val/Test
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
