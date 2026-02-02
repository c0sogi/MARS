import os
import ast
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_split="train"):
    """
    Returns albumentations transforms for the specified data split.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # CoarseDropout with mask_fill_value=0 ensures that if we drop a region
                # in the image, the corresponding region in the mask is also zeroed out.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMAGE_SIZE // 10,
                    max_width=Config.IMAGE_SIZE // 10,
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
        # Val and Test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def preprocess_and_cache(df, prefix, load_cached_data=True):
    """
    Loads DICOMs, resizes them, generates masks, and caches the result as .npy files.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        prefix (str): Prefix for cache filenames (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, labels, image_ids)
               masks and labels will be None for test set if not available.
    """
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)

    img_path = os.path.join(save_dir, f"{prefix}_images.npy")
    mask_path = os.path.join(save_dir, f"{prefix}_masks.npy")
    label_path = os.path.join(save_dir, f"{prefix}_labels.npy")
    ids_path = os.path.join(save_dir, f"{prefix}_ids.npy")

    has_labels = "Negative for Pneumonia" in df.columns

    # 1. Try Loading Cache
    if load_cached_data:
        if os.path.exists(img_path) and os.path.exists(ids_path):
            # Check if masks/labels exist if expected
            if has_labels and (
                not os.path.exists(mask_path) or not os.path.exists(label_path)
            ):
                pass  # Cache incomplete, rebuild
            else:
                print(f"Loading cached data for {prefix}...")
                images = np.load(img_path)
                image_ids = np.load(ids_path, allow_pickle=True)

                masks = np.load(mask_path) if has_labels else None
                labels = np.load(label_path) if has_labels else None

                return images, masks, labels, image_ids

    # 2. Process from Scratch
    print(f"Processing data for {prefix}...")

    img_list = []
    mask_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        # --- Load Image ---
        dcm_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array

            # Handle Photometric Interpretation
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.max(img) - img

            # Normalize to 0-255
            img = img.astype(float)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

            # Keep original dims for box scaling
            orig_h, orig_w = img.shape[:2]

            # Resize to target size
            img_resized = cv2.resize(
                img,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            # Convert to RGB (3 channels) for model compatibility
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

            img_list.append(img_resized)
            id_list.append(row["image_id"])

            # --- Load Labels & Masks (Train/Val only) ---
            if has_labels:
                # 1. Study Label
                # 'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance'
                # These are mutually exclusive in this problem formulation usually, or we treat them as one-hot.
                # The config defines CLASS_LABELS list.
                l = [
                    row["Negative for Pneumonia"],
                    row["Typical Appearance"],
                    row["Indeterminate Appearance"],
                    row["Atypical Appearance"],
                ]
                # Convert to class index (0-3) if single label, or keep as one-hot.
                # Config says "Soft Target CrossEntropy", implying we might use probabilities or one-hot.
                # Let's store as one-hot vector (4,)
                label_list.append(np.array(l, dtype=np.float32))

                # 2. Segmentation Mask
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                )

                if pd.notna(row["boxes"]):
                    try:
                        boxes = ast.literal_eval(row["boxes"])
                        for box in boxes:
                            # Box format: x, y, width, height
                            x, y, w, h = box["x"], box["y"], box["width"], box["height"]

                            # Scale to new size
                            x = x * (Config.IMAGE_SIZE / orig_w)
                            w = w * (Config.IMAGE_SIZE / orig_w)
                            y = y * (Config.IMAGE_SIZE / orig_h)
                            h = h * (Config.IMAGE_SIZE / orig_h)

                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)

                            cv2.rectangle(
                                mask, (x1, y1), (x2, y2), 1.0, -1
                            )  # Fill with 1
                    except:
                        pass  # No valid boxes or parsing error

                mask_list.append(mask)

        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")
            continue

    # Convert to numpy arrays
    images = np.array(img_list)
    image_ids = np.array(id_list)

    if has_labels:
        masks = np.array(mask_list)
        # Expand masks dim to (N, H, W, 1) or keep (N, H, W) depending on albumentations needs
        # Albumentations expects (H, W) or (H, W, C).
        labels = np.array(label_list)
    else:
        masks = None
        labels = None

    # Save to cache
    print(f"Saving cached data to {save_dir}...")
    np.save(img_path, images)
    np.save(ids_path, image_ids)
    if has_labels:
        np.save(mask_path, masks)
        np.save(label_path, labels)

    return images, masks, labels, image_ids


class SIIMDataset(Dataset):
    def __init__(self, images, masks=None, labels=None, image_ids=None, transform=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.masks is not None:
            # Train/Val mode
            mask = self.masks[idx]
            label = self.labels[idx]

            if self.transform:
                # Albumentations expects mask to be passed as named argument
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask has channel dimension if needed by model (1, H, W)
            # Albumentations ToTensorV2 converts image to (C, H, W) but mask usually stays (H, W) if input was (H, W)
            # We explicitly add channel dim for mask: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return image, label, mask

        else:
            # Test mode
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

            image_id = self.image_ids[idx]
            return image, image_id


def get_train_val_loaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Process Data
    train_imgs, train_masks, train_labels, train_ids = preprocess_and_cache(
        train_df, "train", load_cached_data=load_cached_data
    )
    val_imgs, val_masks, val_labels, val_ids = preprocess_and_cache(
        val_df, "val", load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = SIIMDataset(
        train_imgs,
        train_masks,
        train_labels,
        train_ids,
        transform=get_transforms("train"),
    )
    val_dataset = SIIMDataset(
        val_imgs, val_masks, val_labels, val_ids, transform=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for MixUp to have consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA)

    if Config.DEBUG:
        test_df = test_df.head(20)

    test_imgs, _, _, test_ids = preprocess_and_cache(
        test_df, "test", load_cached_data=load_cached_data
    )

    test_dataset = SIIMDataset(
        test_imgs,
        masks=None,
        labels=None,
        image_ids=test_ids,
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
