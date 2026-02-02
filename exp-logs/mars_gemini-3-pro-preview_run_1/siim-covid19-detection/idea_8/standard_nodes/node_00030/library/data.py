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


def get_transforms(data="train"):
    """
    Defines the albumentations transformation pipeline.

    Args:
        data (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # Images are already resized to Config.IMG_SIZE in preprocessing
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Label-Consistent CoarseDropout: mask_fill_value=0 ensures mask is also occluded
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
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def preprocess_data(df, split_name, load_cached_data=True):
    """
    Loads DICOMs, resizes images, generates masks, and caches the result as .npy files.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        split_name (str): Name of the split (train/val/test) for file naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, masks_array, labels_array)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    img_path = os.path.join(Config.WORKING_DIR, f"{split_name}_images.npy")
    mask_path = os.path.join(Config.WORKING_DIR, f"{split_name}_masks.npy")
    label_path = os.path.join(Config.WORKING_DIR, f"{split_name}_labels.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(img_path)
        and os.path.exists(mask_path)
        and os.path.exists(label_path)
    ):
        print(f"Loading cached {split_name} data from {Config.WORKING_DIR}...")
        images = np.load(img_path)
        masks = np.load(mask_path)
        labels = np.load(label_path)
        return images, masks, labels

    # 2. Process from scratch
    print(f"Processing {split_name} data (Cache miss or forced reload)...")

    num_samples = len(df)
    img_size = Config.IMG_SIZE

    # Pre-allocate arrays to save memory fragmentation
    # Images: (N, H, W, 3) uint8 - We replicate channels for ImageNet pretraining compatibility
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.uint8)
    # Masks: (N, H, W) uint8 - Binary mask
    masks = np.zeros((num_samples, img_size, img_size), dtype=np.uint8)
    # Labels: (N,) int64 - Class index
    labels = np.zeros((num_samples,), dtype=np.int64)

    for idx, row in df.iterrows():
        # --- Load Image ---
        dcm_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            dcm = pydicom.dcmread(dcm_path)
            pixel_array = dcm.pixel_array

            # Handle Photometric Interpretation if necessary (simple fix for common inversion)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                pixel_array = np.amax(pixel_array) - pixel_array

            # Normalize to 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            if np.max(pixel_array) > 0:
                pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)

            # Resize
            orig_h, orig_w = pixel_array.shape[:2]
            resized_img = cv2.resize(
                pixel_array, (img_size, img_size), interpolation=cv2.INTER_AREA
            )

            # Stack to 3 channels
            images[idx] = np.stack([resized_img] * 3, axis=-1)

        except Exception as e:
            print(f"Error reading {dcm_path}: {e}")
            # Leave as zeros
            orig_h, orig_w = 1, 1  # dummy

        # --- Generate Mask ---
        # Only for train/val splits that have 'boxes'
        if "boxes" in row and pd.notna(row["boxes"]):
            try:
                boxes = ast.literal_eval(row["boxes"])
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                for box in boxes:
                    x = int(box["x"])
                    y = int(box["y"])
                    w = int(box["width"])
                    h = int(box["height"])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)

                # Resize mask
                resized_mask = cv2.resize(
                    mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST
                )
                masks[idx] = resized_mask
            except:
                pass  # Empty mask

        # --- Extract Label ---
        # Only for train/val splits
        label_idx = 0  # Default to Negative
        found_label = False
        for i, label_name in enumerate(Config.STUDY_LABELS):
            if label_name in row and row[label_name] == 1:
                label_idx = i
                found_label = True
                break

        labels[idx] = label_idx

    # 3. Save to cache
    print(f"Saving processed {split_name} data to {Config.WORKING_DIR}...")
    np.save(img_path, images)
    np.save(mask_path, masks)
    np.save(label_path, labels)

    return images, masks, labels


class SIIMDataset(Dataset):
    def __init__(self, images, masks, labels, study_ids, image_ids, transforms=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.study_ids = study_ids
        self.image_ids = image_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        study_id = self.study_ids[idx]
        image_id = self.image_ids[idx]

        if self.transforms:
            # Albumentations expects image as HWC, mask as H,W
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W) for PyTorch
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask.float(),  # BCE expects float
            "label": torch.tensor(label, dtype=torch.long),  # CrossEntropy expects long
            "study_id": study_id,
            "image_id": image_id,
        }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_limit=None,
):
    """
    Prepares DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug_limit (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val = df_val.head(debug_limit)
        df_test = df_test.head(debug_limit)

    # 2. Preprocess Data (Deterministic & Cached)
    train_imgs, train_masks, train_lbls = preprocess_data(
        df_train, "train", load_cached_data
    )
    val_imgs, val_masks, val_lbls = preprocess_data(df_val, "val", load_cached_data)
    test_imgs, test_masks, test_lbls = preprocess_data(
        df_test, "test", load_cached_data
    )

    # 3. Create Datasets
    train_dataset = SIIMDataset(
        train_imgs,
        train_masks,
        train_lbls,
        df_train["study_id"].values,
        df_train["image_id"].values,
        transforms=get_transforms("train"),
    )

    val_dataset = SIIMDataset(
        val_imgs,
        val_masks,
        val_lbls,
        df_val["study_id"].values,
        df_val["image_id"].values,
        transforms=get_transforms("val"),
    )

    test_dataset = SIIMDataset(
        test_imgs,
        test_masks,
        test_lbls,
        df_test["study_id"].values,
        df_test["image_id"].values,
        transforms=get_transforms("test"),
    )

    # 4. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
