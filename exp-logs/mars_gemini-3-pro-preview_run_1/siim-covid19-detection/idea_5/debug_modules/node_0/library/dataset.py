import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_META,
    VAL_META,
    TEST_META,
    WORKING_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    CLASS_LABELS,
)


def get_transforms(split="train"):
    """
    Returns the albumentations transformation pipeline.

    Key Strategy: Label-Consistent CoarseDropout.
    We set mask_fill_value=0 so that when regions are dropped in the image,
    the corresponding ground truth mask area is also zeroed out.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # Label-Consistent CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(IMG_SIZE * 0.1),
                    max_width=int(IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=0,  # Crucial: remove label where image is occluded
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file and converts it to a standard numpy array (0-255).
    """
    try:
        dicom = pydicom.dcmread(path)

        # VOI LUT (if available)
        if "VOILUTSequence" in dicom:
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Photometric Interpretation
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        return data
    except Exception as e:
        # Fallback for corrupt or problematic DICOMs
        print(f"Error reading DICOM {path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)


def create_mask(boxes_str, orig_h, orig_w, target_h, target_w):
    """
    Parses bounding box string and generates a binary mask resized to target dimensions.
    """
    mask = np.zeros((target_h, target_w), dtype=np.float32)

    if pd.isna(boxes_str) or boxes_str == "":
        return mask

    try:
        boxes = ast.literal_eval(boxes_str)
    except:
        return mask

    # Calculate scale factors
    scale_y = target_h / orig_h
    scale_x = target_w / orig_w

    for box in boxes:
        # box format in csv is usually {'x': ..., 'y': ..., 'width': ..., 'height': ...}
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

        # Scale to target size
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        # Draw filled rectangle (1.0)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)

    return mask


def cache_data(df, split_name, load_cached_data=True):
    """
    Processes DICOM images and annotations, caching them as .npy files.
    """
    # Define paths
    img_path = os.path.join(WORKING_DIR, f"{split_name}_images.npy")
    mask_path = os.path.join(WORKING_DIR, f"{split_name}_masks.npy")
    lbl_path = os.path.join(WORKING_DIR, f"{split_name}_labels.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(img_path)
        and os.path.exists(mask_path)
        and os.path.exists(lbl_path)
    ):
        print(f"Loading cached {split_name} data from {WORKING_DIR}...")
        images = np.load(img_path)
        masks = np.load(mask_path)
        labels = np.load(lbl_path)
        return images, masks, labels

    print(f"Processing {split_name} data (Cache miss or forced reload)...")

    img_list = []
    mask_list = []
    lbl_list = []

    # Study level columns
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    for idx, row in df.iterrows():
        # 1. Image Path
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # 2. Read and Resize Image
        img = read_dicom(full_path)
        orig_h, orig_w = img.shape
        img_resized = cv2.resize(
            img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
        )

        # Convert to RGB (3 channels) for ResNet
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_list.append(img_rgb)

        # 3. Create Mask
        if "boxes" in row:
            mask = create_mask(row["boxes"], orig_h, orig_w, IMG_SIZE, IMG_SIZE)
        else:
            # Test set has no boxes
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask_list.append(mask)

        # 4. Labels
        if all(c in row for c in label_cols):
            # Argmax to get class index 0-3
            # We assume one-hot or similar, but dataset might have multiple?
            # Task description says "predict at least one".
            # For training with CrossEntropy, we usually need a single target.
            # We take the argmax. If multi-label exists, this simplifies it.
            vals = row[label_cols].values.astype(float)
            lbl_idx = np.argmax(vals)
            lbl_list.append(lbl_idx)
        else:
            # Test set dummy label
            lbl_list.append(0)

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    masks = np.array(
        mask_list, dtype=np.float32
    )  # Keep float for potential soft masks or resizing artifacts
    labels = np.array(lbl_list, dtype=np.int64)

    # Save to cache
    np.save(img_path, images)
    np.save(mask_path, masks)
    np.save(lbl_path, labels)

    print(f"Saved {split_name} data to {WORKING_DIR}")
    return images, masks, labels


class CovidDataset(Dataset):
    def __init__(self, images, masks, labels, ids, transform=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.ids = ids  # List of image IDs for tracking
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        image_id = self.ids[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Mask needs to be (1, H, W) for BCEWithLogitsLoss usually, or (H, W) depends on usage.
        # Albumentations returns (H, W) for mask. We add channel dim.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
        }


def get_dataloaders(load_cached_data=True, debug=False, debug_sample_size=100):
    """
    Main function to prepare DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_META)
    val_df = pd.read_csv(VAL_META)
    test_df = pd.read_csv(TEST_META)

    if debug:
        train_df = train_df.head(debug_sample_size)
        val_df = val_df.head(debug_sample_size)
        # Test usually kept as is or small sample
        test_df = test_df.head(debug_sample_size)

    # 2. Process/Load Data (Numpy Arrays)
    train_imgs, train_masks, train_lbls = cache_data(
        train_df, "train", load_cached_data
    )
    val_imgs, val_masks, val_lbls = cache_data(val_df, "val", load_cached_data)
    test_imgs, test_masks, test_lbls = cache_data(test_df, "test", load_cached_data)

    # 3. Create Datasets
    # Pass image_ids to track predictions
    train_dataset = CovidDataset(
        train_imgs,
        train_masks,
        train_lbls,
        train_df["image_id"].values,
        transform=get_transforms("train"),
    )

    val_dataset = CovidDataset(
        val_imgs,
        val_masks,
        val_lbls,
        val_df["image_id"].values,
        transform=get_transforms("val"),
    )

    test_dataset = CovidDataset(
        test_imgs,
        test_masks,
        test_lbls,
        test_df["image_id"].values,
        transform=get_transforms("test"),
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
