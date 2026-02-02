import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from library.config import Config


# =========================================================================
# Dataset Class
# =========================================================================
class ChestXrayDataset(Dataset):
    def __init__(self, images, masks=None, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            masks (np.ndarray, optional): Array of masks (N, H, W).
            labels (np.ndarray, optional): Array of one-hot labels (N, NumClasses).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is uint8 (H, W, 3)
        image = self.images[idx]

        data = {"image": image}

        # Add mask to data dictionary if available
        if self.masks is not None:
            # Mask is uint8 (H, W) or (H, W, 1)
            mask = self.masks[idx]
            data["mask"] = mask

        # Apply transforms
        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if self.masks is not None:
                mask = augmented["mask"]

        # Post-processing
        # Image is already Tensor (C, H, W) due to ToTensorV2 in transform

        # Handle Mask
        if self.masks is not None:
            # Albumentations might return mask as (H, W) or (H, W, 1)
            # We need (1, H, W) float tensor
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)
            else:
                # Fallback if ToTensorV2 didn't apply to mask (depends on config)
                mask = torch.from_numpy(mask).float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)

            # Binarize just in case interpolation messed up edges (though usually nearest)
            mask = (mask > 0.5).float()
        else:
            # Dummy mask for test set
            mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        # Handle Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Dummy label for test set
            label = torch.zeros(Config.NUM_STUDY_CLASSES, dtype=torch.float32)

        return image, label, mask


# =========================================================================
# Data Processing & Caching
# =========================================================================
def process_data(df, input_dir, img_size, is_test=False):
    """
    Reads DICOMs, resizes them, and generates masks/labels.
    Returns numpy arrays.
    """
    images = []
    masks = []
    labels = []

    # Pre-calculate scaling factors if we were doing box scaling,
    # but here we read image -> resize -> draw mask on resized dimensions or
    # draw on original -> resize.
    # Strategy: Read Dicom -> Resize Image. Scale Box Coords -> Draw on Resized Mask.

    print(f"Processing {len(df)} images...")

    for idx, row in df.iterrows():
        # 1. Read Image
        file_path = os.path.join(input_dir, row["file_path"])
        try:
            dcm = pydicom.dcmread(file_path)
            pixel_array = dcm.pixel_array

            # Handle Photometric Interpretation (Invert if MONOCHROME1)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                pixel_array = np.amax(pixel_array) - pixel_array

            # Normalize to 0-255 uint8
            pixel_array = pixel_array.astype(float)
            pixel_array = (pixel_array - pixel_array.min()) / (
                pixel_array.max() - pixel_array.min() + 1e-6
            )
            pixel_array = (pixel_array * 255).astype(np.uint8)

            # Convert to RGB (EfficientNet expects 3 channels)
            if pixel_array.ndim == 2:
                img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

            # Original Dimensions
            orig_h, orig_w = pixel_array.shape[:2]

            # Resize Image
            img_resized = cv2.resize(
                img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
            images.append(img_resized)

            if not is_test:
                # 2. Generate Mask
                mask = np.zeros((img_size, img_size), dtype=np.float32)

                if pd.notna(row["boxes"]):
                    try:
                        boxes = literal_eval(row["boxes"])
                        for box in boxes:
                            # Box format: x, y, width, height (in original coords)
                            x, y, w, h = box["x"], box["y"], box["width"], box["height"]

                            # Scale to new size
                            x_scale = img_size / orig_w
                            y_scale = img_size / orig_h

                            x_min = int(x * x_scale)
                            y_min = int(y * y_scale)
                            x_max = int((x + w) * x_scale)
                            y_max = int((y + h) * y_scale)

                            # Draw rectangle (fill=1)
                            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1.0, -1)
                    except:
                        pass  # Empty mask if parse fails

                masks.append(mask)

                # 3. Labels
                # Columns: Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance
                # Mapped via Config.STUDY_LABELS
                label_vec = np.zeros(Config.NUM_STUDY_CLASSES, dtype=np.float32)
                for i, label_name in enumerate(Config.STUDY_LABELS):
                    if row[label_name] == 1:
                        label_vec[i] = 1.0
                labels.append(label_vec)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Append dummy data to maintain alignment or skip?
            # Safer to skip, but indices must align.
            # Given the clean metadata, we expect files to exist.
            # If fail, we append zeros to avoid crashing, but this shouldn't happen with verified metadata.
            images.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            if not is_test:
                masks.append(np.zeros((img_size, img_size), dtype=np.float32))
                labels.append(np.zeros(Config.NUM_STUDY_CLASSES, dtype=np.float32))

    images = np.array(images, dtype=np.uint8)

    if is_test:
        return images, None, None

    masks = np.array(masks, dtype=np.uint8)  # Save as uint8 to save space (0/1)
    labels = np.array(labels, dtype=np.float32)

    return images, masks, labels


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to prepare datasets and dataloaders.
    Handles caching logic.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ====================================================
    # 1. Load Train Data
    # ====================================================
    if (
        load_cached_data
        and os.path.exists(Config.TRAIN_CACHE_IMAGES)
        and os.path.exists(Config.TRAIN_CACHE_MASKS)
        and os.path.exists(Config.TRAIN_CACHE_LABELS)
    ):

        print("Loading cached training data...")
        train_images = np.load(Config.TRAIN_CACHE_IMAGES)
        train_masks = np.load(Config.TRAIN_CACHE_MASKS)
        train_labels = np.load(Config.TRAIN_CACHE_LABELS)
    else:
        print("Processing training data from scratch...")
        df_train = pd.read_csv(train_metadata_path)

        # Debug Mode
        if Config.DEBUG:
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)

        train_images, train_masks, train_labels = process_data(
            df_train, Config.INPUT_DIR, Config.IMG_SIZE, is_test=False
        )

        # Save to cache
        np.save(Config.TRAIN_CACHE_IMAGES, train_images)
        np.save(Config.TRAIN_CACHE_MASKS, train_masks)
        np.save(Config.TRAIN_CACHE_LABELS, train_labels)

    # ====================================================
    # 2. Load Val Data
    # ====================================================
    if (
        load_cached_data
        and os.path.exists(Config.VAL_CACHE_IMAGES)
        and os.path.exists(Config.VAL_CACHE_MASKS)
        and os.path.exists(Config.VAL_CACHE_LABELS)
    ):

        print("Loading cached validation data...")
        val_images = np.load(Config.VAL_CACHE_IMAGES)
        val_masks = np.load(Config.VAL_CACHE_MASKS)
        val_labels = np.load(Config.VAL_CACHE_LABELS)
    else:
        print("Processing validation data from scratch...")
        df_val = pd.read_csv(val_metadata_path)

        if Config.DEBUG:
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)

        val_images, val_masks, val_labels = process_data(
            df_val, Config.INPUT_DIR, Config.IMG_SIZE, is_test=False
        )

        np.save(Config.VAL_CACHE_IMAGES, val_images)
        np.save(Config.VAL_CACHE_MASKS, val_masks)
        np.save(Config.VAL_CACHE_LABELS, val_labels)

    # ====================================================
    # 3. Define Transforms
    # ====================================================
    # Note: mask_fill_value=0 in CoarseDropout ensures labels are removed in occluded areas
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=int(Config.IMG_SIZE * 0.1),
                max_width=int(Config.IMG_SIZE * 0.1),
                min_holes=1,
                min_height=int(Config.IMG_SIZE * 0.05),
                min_width=int(Config.IMG_SIZE * 0.05),
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # ====================================================
    # 4. Create Datasets & Loaders
    # ====================================================
    train_dataset = ChestXrayDataset(
        train_images, train_masks, train_labels, transform=train_transform
    )
    val_dataset = ChestXrayDataset(
        val_images, val_masks, val_labels, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_metadata_path=Config.TEST_METADATA_PATH,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Prepares test dataloader.
    Also caches test images and saves original dimensions for rescaling predictions later.
    """
    # Check cache
    if (
        load_cached_data
        and os.path.exists(Config.TEST_CACHE_IMAGES)
        and os.path.exists(Config.TEST_CACHE_DIMS)
    ):

        print("Loading cached test data...")
        test_images = np.load(Config.TEST_CACHE_IMAGES)
        # We don't necessarily need to return dims here, usually needed during inference loop
        # But we ensure they exist.
    else:
        print("Processing test data from scratch...")
        df_test = pd.read_csv(test_metadata_path)

        # We need to capture original dimensions to rescale boxes later
        # process_data doesn't return dims by default, so we do a custom loop here or modify process_data
        # Modifying process_data is cleaner but let's keep it simple and just do it here since we need dims.

        images = []
        dims = []  # List of (width, height)

        print(f"Processing {len(df_test)} test images...")
        for idx, row in df_test.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                dcm = pydicom.dcmread(file_path)
                pixel_array = dcm.pixel_array

                if (
                    hasattr(dcm, "PhotometricInterpretation")
                    and dcm.PhotometricInterpretation == "MONOCHROME1"
                ):
                    pixel_array = np.amax(pixel_array) - pixel_array

                orig_h, orig_w = pixel_array.shape[:2]
                dims.append(
                    {
                        "id": row["image_id"],
                        "width": orig_w,
                        "height": orig_h,
                        "study_id": row["study_id"],
                    }
                )

                # Normalize
                pixel_array = pixel_array.astype(float)
                pixel_array = (pixel_array - pixel_array.min()) / (
                    pixel_array.max() - pixel_array.min() + 1e-6
                )
                pixel_array = (pixel_array * 255).astype(np.uint8)

                if pixel_array.ndim == 2:
                    img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

                img_resized = cv2.resize(
                    img_rgb,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                images.append(img_resized)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                images.append(
                    np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
                )
                dims.append(
                    {
                        "id": row["image_id"],
                        "width": 1000,
                        "height": 1000,
                        "study_id": row["study_id"],
                    }
                )

        test_images = np.array(images, dtype=np.uint8)

        # Save cache
        np.save(Config.TEST_CACHE_IMAGES, test_images)
        pd.DataFrame(dims).to_parquet(Config.TEST_CACHE_DIMS)

    # Transform
    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_dataset = ChestXrayDataset(
        test_images, masks=None, labels=None, transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
