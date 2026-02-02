import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ast
from library.config import cfg


def get_transforms(data_split):
    """
    Returns the Albumentations transformations for the specific data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=cfg.image_size, width=cfg.image_size),
                A.HorizontalFlip(p=0.5),
                # CoarseDropout with mask_fill_value=0 ensures that if an opacity is occluded
                # in the image, it is also removed from the ground truth mask.
                A.CoarseDropout(
                    max_holes=cfg.aug_dropout_holes,
                    max_height=cfg.aug_dropout_size,
                    max_width=cfg.aug_dropout_size,
                    min_holes=1,
                    min_height=int(cfg.aug_dropout_size * 0.5),
                    min_width=int(cfg.aug_dropout_size * 0.5),
                    fill_value=0,
                    mask_fill_value=0,
                    p=cfg.aug_dropout_prob,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(height=cfg.image_size, width=cfg.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_dicom(path):
    """
    Reads a DICOM file and converts it to a standard image format.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # Handle Photometric Interpretation if necessary (Monochrome1 vs Monochrome2)
        # Usually standardizing to 0-255 is sufficient for this dataset
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.amax(img) - img

        img = img.astype(np.float32)

        # Normalize to 0-255
        if np.max(img) != 0:
            img = (img / np.max(img)) * 255.0
        img = img.astype(np.uint8)

        # Convert to 3 channels for ResNet compatibility
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        return img
    except Exception as e:
        print(f"Error loading DICOM {path}: {e}")
        # Return a black image in case of failure to prevent crash
        return np.zeros((cfg.image_size, cfg.image_size, 3), dtype=np.uint8)


def create_mask(boxes_str, height, width):
    """
    Creates a binary mask from bounding box string.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if pd.isna(boxes_str) or boxes_str == "":
        return mask

    try:
        boxes = ast.literal_eval(boxes_str)
        for box in boxes:
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["width"])
            h = int(box["height"])

            # Clip coordinates to image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            mask[y1:y2, x1:x2] = 1
    except:
        pass

    return mask


def process_and_cache_data(split, load_cached_data=True):
    """
    Loads metadata and processes images/masks. Uses caching to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (dataframe, images_array, masks_array, dims_array)
               masks_array is None for 'test' split.
    """
    # Define paths
    cache_dir = cfg.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    images_path = os.path.join(cache_dir, f"{split}_images.npy")
    masks_path = os.path.join(cache_dir, f"{split}_masks.npy")
    dims_path = os.path.join(cache_dir, f"{split}_dims.npy")

    # Load Metadata
    if split == "train":
        csv_path = cfg.train_metadata
    elif split == "val":
        csv_path = cfg.val_metadata
    else:
        csv_path = cfg.test_metadata

    df = pd.read_csv(csv_path)

    # Attempt to load from cache
    if load_cached_data:
        has_images = os.path.exists(images_path)
        has_dims = os.path.exists(dims_path)
        has_masks = os.path.exists(masks_path) if split != "test" else True

        if has_images and has_dims and has_masks:
            print(f"Loading cached {split} data from {cache_dir}...")
            images = np.load(images_path)
            dims = np.load(dims_path)
            masks = np.load(masks_path) if split != "test" else None
            return df, images, masks, dims

    # Process from scratch
    print(f"Processing {split} data (Cache miss or force reload)...")

    img_list = []
    mask_list = []
    dim_list = []

    for idx, row in df.iterrows():
        # Load Image
        file_path = os.path.join(cfg.input_dir, row["file_path"])
        img = load_dicom(file_path)

        # Store original dimensions (Height, Width)
        orig_h, orig_w = img.shape[:2]
        dim_list.append([orig_h, orig_w])

        # Resize Image for storage
        img_resized = cv2.resize(
            img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR
        )
        img_list.append(img_resized)

        # Process Mask (Train/Val only)
        if split != "test":
            if "boxes" in row and pd.notna(row["boxes"]):
                mask = create_mask(row["boxes"], orig_h, orig_w)
            else:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            # Resize mask
            mask_resized = cv2.resize(
                mask, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST
            )
            mask_list.append(mask_resized)

    # Convert to arrays
    images = np.array(img_list, dtype=np.uint8)
    dims = np.array(dim_list, dtype=np.int32)

    # Save to cache
    np.save(images_path, images)
    np.save(dims_path, dims)

    if split != "test":
        masks = np.array(mask_list, dtype=np.uint8)
        np.save(masks_path, masks)
    else:
        masks = None

    print(f"Processed and cached {len(images)} images for {split}.")

    return df, images, masks, dims


class SIIMDataset(Dataset):
    def __init__(self, df, images, masks, dims, split="train"):
        self.df = df
        self.images = images
        self.masks = masks
        self.dims = dims
        self.split = split
        self.transforms = get_transforms(split)

        # Pre-compute study labels for train/val
        if self.split != "test":
            self.study_labels = []
            for _, row in self.df.iterrows():
                # Convert one-hot to class index
                # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
                label_idx = 0
                for i, col in enumerate(cfg.study_label_cols):
                    if row[col] == 1:
                        label_idx = i
                        break
                self.study_labels.append(label_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 3)
        orig_h, orig_w = self.dims[idx]

        if self.split != "test":
            mask = self.masks[idx]  # (H, W)
            study_label = self.study_labels[idx]

            # Apply Augmentations
            # Albumentations expects mask to be passed as 'mask'
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = (
                augmented["mask"].float().unsqueeze(0)
            )  # Add channel dim: (1, H, W)

            # Prepare Bounding Boxes for Metric Calculation (Validation only)
            # We need to rescale boxes from original dimensions to current image size (512x512)
            # Note: The mask is already resized, but for mAP calculation we often need boxes.
            # However, the training loop uses mask for segmentation loss.
            # The validation loop calculates mAP. We provide boxes relative to the 512x512 image.

            boxes = []
            row = self.df.iloc[idx]
            if pd.notna(row["boxes"]):
                try:
                    raw_boxes = ast.literal_eval(row["boxes"])
                    scale_x = cfg.image_size / orig_w
                    scale_y = cfg.image_size / orig_h

                    for box in raw_boxes:
                        x = float(box["x"]) * scale_x
                        y = float(box["y"]) * scale_y
                        w = float(box["width"]) * scale_x
                        h = float(box["height"]) * scale_y

                        # Format: xmin, ymin, xmax, ymax
                        boxes.append([x, y, x + w, y + h])
                except:
                    pass

            boxes_tensor = (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            labels_tensor = torch.ones(
                (len(boxes),), dtype=torch.int64
            )  # Class 1 for Opacity

            target = {
                "mask": mask_tensor,
                "study_label": torch.tensor(study_label, dtype=torch.long),
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": row["image_id"],
                "study_id": row["study_id"],
            }

            return image_tensor, target

        else:
            # Test Split
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]

            # Metadata for submission
            row = self.df.iloc[idx]
            meta = {
                "study_id": row["study_id"],
                "image_id": row["image_id"],
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            }

            return image_tensor, meta
