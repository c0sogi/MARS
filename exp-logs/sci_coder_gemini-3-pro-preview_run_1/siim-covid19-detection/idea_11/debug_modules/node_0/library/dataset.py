import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from library.utils import seed_everything

# Constants
IMG_SIZE = 512
CACHE_DIR = "./working/idea_11"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


class SIIMDataset(Dataset):
    def __init__(self, images, masks=None, labels=None, transforms=None, ids=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W) or (N, H, W, C).
            masks (np.ndarray, optional): Array of masks (N, H, W).
            labels (np.ndarray, optional): Array of one-hot labels (N, 4).
            transforms (albumentations.Compose, optional): Transforms to apply.
            ids (list, optional): List of image/study IDs for tracking.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms
        self.ids = ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Convert grayscale to RGB (H, W) -> (H, W, 3)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Prepare data for transforms
        data = {"image": img}

        if self.masks is not None:
            mask = self.masks[idx]
            data["mask"] = mask

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(**data)
            img = augmented["image"]
            if "mask" in augmented:
                mask = augmented["mask"]

        # Prepare return tuple
        # If labels are present, return (img, mask, label)
        # If test set (no labels/masks provided usually), return (img, id) or similar
        # Based on task, we need standard training loop format

        if self.labels is not None:
            label = self.labels[idx]
            # Ensure mask is (1, H, W) float tensor
            # Albumentations ToTensorV2 converts image to Tensor (C, H, W) but mask stays numpy or tensor depending on setup
            # Usually mask comes out as Tensor if passed to ToTensorV2, but let's be explicit
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            # Add channel dim to mask: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return img, mask.float(), torch.tensor(label, dtype=torch.float32)

        # Inference mode
        return img, self.ids[idx]


def get_transforms(split="train"):
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                # CoarseDropout with mask_fill_value=0 for consistency
                # Using Albumentations 2.0+ compatible arguments
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(int(IMG_SIZE * 0.05), int(IMG_SIZE * 0.1)),
                    hole_width_range=(int(IMG_SIZE * 0.05), int(IMG_SIZE * 0.1)),
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
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
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def process_dicom(file_path):
    """
    Reads DICOM, handles photometric interpretation, resizes to IMG_SIZE.
    Returns: resized_img (H, W), original_dims (h, w)
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    dcm = pydicom.dcmread(full_path)
    pixel_array = dcm.pixel_array

    # Handle Photometric Interpretation
    if (
        hasattr(dcm, "PhotometricInterpretation")
        and dcm.PhotometricInterpretation == "MONOCHROME1"
    ):
        pixel_array = np.amax(pixel_array) - pixel_array

    # Normalize to 0-255 uint8
    pixel_array = pixel_array.astype(np.float32)
    pixel_array = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min() + 1e-6)
        * 255.0
    )
    pixel_array = pixel_array.astype(np.uint8)

    orig_h, orig_w = pixel_array.shape[:2]

    # Resize
    resized_img = cv2.resize(
        pixel_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )

    return resized_img, (orig_h, orig_w)


def create_mask(boxes_str, orig_dims):
    """
    Parses boxes string and draws mask on 512x512 canvas.
    """
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    orig_h, orig_w = orig_dims

    if pd.isna(boxes_str) or boxes_str == "":
        return mask

    try:
        boxes = literal_eval(boxes_str)
    except:
        return mask

    # Scale factors
    scale_x = IMG_SIZE / orig_w
    scale_y = IMG_SIZE / orig_h

    for box in boxes:
        # Box format in metadata: {'x': ..., 'y': ..., 'width': ..., 'height': ...}
        x = float(box["x"]) * scale_x
        y = float(box["y"]) * scale_y
        w = float(box["width"]) * scale_x
        h = float(box["height"]) * scale_y

        x1 = int(np.round(x))
        y1 = int(np.round(y))
        x2 = int(np.round(x + w))
        y2 = int(np.round(y + h))

        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

    return mask


def get_data(split, load_cached_data=True, debug_limit=None):
    """
    Loads data for a specific split (train, val, test).
    Uses caching to speed up subsequent runs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    paths = {
        "images": os.path.join(CACHE_DIR, f"{split}_images.npy"),
        "masks": os.path.join(CACHE_DIR, f"{split}_masks.npy"),
        "labels": os.path.join(CACHE_DIR, f"{split}_labels.npy"),
        "ids": os.path.join(CACHE_DIR, f"{split}_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(
        os.path.exists(p)
        for p in paths.values()
        if split != "test" or "labels" not in p
    )

    if load_cached_data and cache_exists:
        print(f"Loading {split} data from cache...")
        images = np.load(paths["images"])
        ids = np.load(paths["ids"])

        if split != "test":
            masks = np.load(paths["masks"])
            labels = np.load(paths["labels"])
            return images, masks, labels, ids
        else:
            return images, None, None, ids

    # Process from scratch
    print(f"Processing {split} data from scratch...")

    # Load metadata
    df = pd.read_csv(os.path.join(METADATA_DIR, f"{split}.csv"))

    if debug_limit:
        df = df.head(debug_limit)

    img_list = []
    mask_list = []
    label_list = []
    id_list = []

    # Study level columns for one-hot encoding
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    for idx, row in df.iterrows():
        # Process Image
        img, orig_dims = process_dicom(row["file_path"])
        img_list.append(img)
        id_list.append(row["image_id"])

        if split != "test":
            # Process Mask
            mask = create_mask(row.get("boxes", ""), orig_dims)
            mask_list.append(mask)

            # Process Labels
            labels = row[label_cols].values.astype(np.float32)
            label_list.append(labels)

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)

    # Save to cache
    np.save(paths["images"], images)
    np.save(paths["ids"], ids)

    if split != "test":
        masks = np.array(mask_list, dtype=np.uint8)
        labels = np.array(label_list, dtype=np.float32)

        np.save(paths["masks"], masks)
        np.save(paths["labels"], labels)

        return images, masks, labels, ids
    else:
        return images, None, None, ids


def get_dataset(split, load_cached_data=True, debug=False):
    """
    Factory function to get a SIIMDataset instance.
    """
    seed_everything(42)

    debug_limit = 100 if debug else None

    images, masks, labels, ids = get_data(split, load_cached_data, debug_limit)

    transforms = get_transforms(split)

    # For validation, we treat it like test in terms of transforms (no augmentation),
    # but we provide masks/labels for metric calculation if needed.
    # However, standard practice is to use "val" split with deterministic transforms.

    if split == "train":
        ds = SIIMDataset(images, masks, labels, transforms=transforms, ids=ids)
    elif split == "val":
        ds = SIIMDataset(
            images, masks, labels, transforms=get_transforms("val"), ids=ids
        )
    else:  # test
        ds = SIIMDataset(
            images, masks=None, labels=None, transforms=get_transforms("test"), ids=ids
        )

    return ds
