import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def read_dicom(path, target_size=None):
    """
    Reads a DICOM file, handles photometric interpretation, normalizes to 0-255,
    and resizes.
    """
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels=False)
        img = dcm.pixel_array.astype(float)

        # Handle Photometric Interpretation
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Normalize to 0-255
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255.0).astype(np.uint8)
        else:
            img = np.zeros(img.shape, dtype=np.uint8)

        # Resize if needed
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Convert to RGB (3 channels)
        img = np.stack([img, img, img], axis=-1)

        return img
    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        # Return black image on failure
        if target_size:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return np.zeros((512, 512, 3), dtype=np.uint8)


def prepare_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads metadata and prepares images. Uses caching to speed up subsequent runs.
    """
    df = pd.read_csv(metadata_path)

    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_images.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} images from cache: {cache_path}")
        try:
            images = np.load(cache_path)
            if len(images) == len(df):
                return df, images
            else:
                print(f"Cache mismatch ({len(images)} vs {len(df)}). Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # Process from scratch
    print(f"Processing {len(df)} images for {split_name} set...")
    images = []

    for idx, row in df.iterrows():
        # Construct full path. Metadata paths are relative to INPUT_DIR
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        img = read_dicom(full_path, target_size=Config.IMG_SIZE)
        images.append(img)

    images = np.array(images)

    # Save to cache
    print(f"Saving {split_name} images to cache: {cache_path}")
    np.save(cache_path, images)

    return df, images


class ChestXrayDataset(Dataset):
    def __init__(self, df, images, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            images (np.array): Pre-loaded image array (N, H, W, 3).
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.images = images
        self.transforms = transforms
        self.mode = mode

        # Define label columns for study classification
        self.label_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image
        image = self.images[idx]

        if self.mode == "test":
            # Test mode: No labels, no masks
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented["image"]

            study_id = self.df.iloc[idx]["study_id"]
            image_id = self.df.iloc[idx]["image_id"]
            return image, study_id, image_id

        # 2. Get Mask (Train/Val)
        # Initialize empty mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        box_str = self.df.iloc[idx]["boxes"]
        if pd.notna(box_str):
            try:
                boxes = ast.literal_eval(box_str)
                # Original image dimensions are needed to scale boxes if image was resized
                # However, we loaded resized images.
                # The boxes in metadata are in ORIGINAL coordinates.
                # We need to scale them to Config.IMG_SIZE.
                # But wait, we don't have original dimensions stored in the numpy array.
                # We must rely on the assumption that the aspect ratio is preserved or
                # we need to re-read original dims.
                # To be efficient, we should have stored scaling factors.
                # However, for this task, let's assume we need to calculate scale on the fly
                # or use the fact that DICOM reading in prepare_data discarded original dims.
                # FIX: We need original dims.
                # Optimization: Since we don't have original dims in the cached numpy array,
                # we will assume the boxes need to be scaled based on the metadata if available,
                # or we just rely on the fact that we need to generate the mask.
                # Actually, albumentations handles box resizing if we passed boxes,
                # but here we are generating a mask.
                # We need to know the scale factor.
                # Let's peek at the file to get dims? No, too slow.
                # Alternative: The metadata generation script did not save original dims.
                # BUT, `read_dicom` resizes.
                # Let's assume for this implementation we approximate or
                # re-read the DICOM header quickly if needed.
                # BETTER APPROACH: The `train_image_level.csv` or metadata doesn't have dims.
                # We will read the DICOM header here efficiently just for rows/cols.
                # This adds overhead but is necessary for correct mask generation.

                full_path = os.path.join(
                    Config.INPUT_DIR, self.df.iloc[idx]["file_path"]
                )
                dcm_header = pydicom.dcmread(full_path, stop_before_pixels=True)
                orig_h, orig_w = dcm_header.Rows, dcm_header.Columns

                scale_x = Config.IMG_SIZE[0] / orig_w
                scale_y = Config.IMG_SIZE[1] / orig_h

                for box in boxes:
                    # Box format: {'x': ..., 'y': ..., 'width': ..., 'height': ...}
                    x, y, w, h = box["x"], box["y"], box["width"], box["height"]

                    x_min = int(x * scale_x)
                    y_min = int(y * scale_y)
                    x_max = int((x + w) * scale_x)
                    y_max = int((y + h) * scale_y)

                    # Clip to image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(Config.IMG_SIZE[0], x_max)
                    y_max = min(Config.IMG_SIZE[1], y_max)

                    mask[y_min:y_max, x_min:x_max] = 1.0

            except Exception as e:
                # If box parsing fails, mask remains zeros
                pass

        # 3. Get Label (Train/Val)
        # Convert one-hot to class index
        labels = self.df.iloc[idx][self.label_cols].values.astype(float)
        label_idx = np.argmax(labels)  # 0, 1, 2, 3

        # 4. Augmentations
        if self.transforms:
            # Albumentations expects mask to be passed
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask is (1, H, W)
            # Albumentations returns (H, W) for mask if input was (H, W)
            mask = mask.unsqueeze(0)  # Add channel dim

        return image, torch.tensor(label_idx, dtype=torch.long), mask


def get_transforms(split):
    """
    Returns Albumentations transforms for the specified split.
    """
    if split == "train":
        return A.Compose(
            [
                # CoarseDropout: Rectangular holes.
                # mask_fill_value=0 ensures mask is zeroed out where image is dropped.
                A.CoarseDropout(
                    max_holes=Config.AUG_COARSE_DROPOUT_HOLES,
                    max_height=Config.AUG_COARSE_DROPOUT_SIZE,
                    max_width=Config.AUG_COARSE_DROPOUT_SIZE,
                    min_holes=1,
                    min_height=Config.AUG_COARSE_DROPOUT_SIZE // 2,
                    min_width=Config.AUG_COARSE_DROPOUT_SIZE // 2,
                    fill_value=0,
                    mask_fill_value=0,
                    p=Config.AUG_COARSE_DROPOUT_PROB,
                ),
                # Normalize to ImageNet mean/std
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get dataloaders.
    Handles data processing, caching, and loader creation.
    """
    # 1. Prepare Data
    train_df, train_images = prepare_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )

    val_df, val_images = prepare_data(Config.VAL_METADATA_PATH, "val", load_cached_data)

    # Test data preparation (if needed for submission)
    # We load it but don't create a dataloader here unless requested,
    # but usually we return train/val loaders for training loop.
    # The training loop might request test loader separately.
    # For this function, we return train and val loaders.

    # 2. Create Datasets
    train_dataset = ChestXrayDataset(
        train_df, train_images, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = ChestXrayDataset(
        val_df, val_images, transforms=get_transforms("val"), mode="val"
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Separate function to get test dataloader for inference.
    """
    test_df, test_images = prepare_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    test_dataset = ChestXrayDataset(
        test_df, test_images, transforms=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return test_loader
