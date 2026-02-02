import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train", input_size=256):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        input_size (int): Target input size for the model.
                          Cached images are 256x256. If input_size differs (e.g., 224 for Swin),
                          a Resize operation is prepended.
    """
    transforms_list = []

    # 1. Resize if model input requirement differs from cached crop size (256)
    if input_size != Config.CROP_SIZE:
        transforms_list.append(A.Resize(height=input_size, width=input_size))

    # 2. Augmentations (Train only)
    if mode == "train":
        transforms_list.append(A.HorizontalFlip(p=0.5))

    # 3. Normalization and Tensor Conversion (All modes)
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Serves images from pre-loaded numpy arrays and applies transforms on-the-fly.
    """

    def __init__(self, images, labels=None, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as (H, W, C) uint8 RGB
        image = self.images[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            # For test set, return only image (or dummy label if needed, but here just image)
            return image, torch.tensor(-1)


def preprocess_image(img_path, resize_dim, crop_size):
    """
    Reads an image, converts to RGB, and applies the deterministic geometric pipeline:
    Resize (maintaining aspect ratio) -> Center Crop.
    """
    img = cv2.imread(img_path)
    if img is None:
        # Fallback or error; raising error is safer to detect data issues
        raise FileNotFoundError(f"Image not found at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # Resize logic: Scale image so the smaller dimension equals resize_dim
    if h < w:
        new_h = resize_dim
        new_w = int(w * (resize_dim / h))
    else:
        new_w = resize_dim
        new_h = int(h * (resize_dim / w))

    # Use Cubic interpolation for high quality downscaling
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Center Crop
    h, w, _ = img.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # Clamp to 0 to be safe
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    img = img[start_y : start_y + crop_size, start_x : start_x + crop_size]

    return img


def load_data(
    metadata_path,
    input_dir,
    cache_name,
    load_cached_data=True,
    label_map=None,
    is_test=False,
):
    """
    Loads data from metadata, processes images, and caches them as .npy files.

    Args:
        metadata_path (str): Path to the CSV file.
        input_dir (str): Root directory for images.
        cache_name (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        label_map (dict): Mapping from breed string to int. Required if not is_test and not generating new map.
        is_test (bool): If True, does not look for labels.

    Returns:
        images (np.array), labels (np.array or None), label_map (dict)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_images.npy")
    labels_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")
    map_cache_path = os.path.join(Config.WORKING_DIR, "label_map.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(images_cache_path):
        # For training data, we also need labels
        if is_test or os.path.exists(labels_cache_path):
            print(f"Loading cached {cache_name} data from {Config.WORKING_DIR}...")
            images = np.load(images_cache_path)

            labels = None
            if not is_test:
                labels = np.load(labels_cache_path)
                # Load label map if not provided
                if label_map is None and os.path.exists(map_cache_path):
                    label_map = np.load(map_cache_path, allow_pickle=True).item()

            return images, labels, label_map

    # 2. Process from Scratch
    print(f"Processing {cache_name} data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Generate Label Map if needed (only for training set usually)
    if not is_test and label_map is None:
        if os.path.exists(map_cache_path) and load_cached_data:
            label_map = np.load(map_cache_path, allow_pickle=True).item()
        else:
            unique_breeds = sorted(df["breed"].unique())
            label_map = {breed: i for i, breed in enumerate(unique_breeds)}
            np.save(map_cache_path, label_map)
            print(f"Generated and saved label map with {len(label_map)} classes.")

    images = []
    labels = []

    for _, row in df.iterrows():
        # Metadata file_path is relative to input_dir (e.g., "train/id.jpg")
        full_path = os.path.join(input_dir, row["file_path"])

        try:
            img = preprocess_image(full_path, Config.RESIZE_DIM, Config.CROP_SIZE)

            if not is_test:
                label = label_map[row["breed"]]
                images.append(img)
                labels.append(label)
            else:
                images.append(img)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    np.save(images_cache_path, images)

    if not is_test:
        labels = np.array(labels, dtype=np.int64)
        np.save(labels_cache_path, labels)
    else:
        labels = None

    print(f"Processed and cached {len(images)} images for {cache_name}.")
    return images, labels, label_map


def get_datasets(load_cached_data=True):
    """
    Main entry point to get Train, Val, and Test datasets.
    Handles loading, caching, and dataset creation.

    Returns:
        train_data (tuple): (images, labels)
        val_data (tuple): (images, labels)
        test_data (tuple): (images, None)
        label_map (dict): Class mapping
    """
    # Load Train
    train_images, train_labels, label_map = load_data(
        Config.TRAIN_METADATA,
        Config.INPUT_DIR,
        "train",
        load_cached_data=load_cached_data,
    )

    # Load Val (Use label_map from train)
    val_images, val_labels, _ = load_data(
        Config.VAL_METADATA,
        Config.INPUT_DIR,
        "val",
        load_cached_data=load_cached_data,
        label_map=label_map,
    )

    # Load Test
    test_images, _, _ = load_data(
        Config.TEST_METADATA,
        Config.INPUT_DIR,
        "test",
        load_cached_data=load_cached_data,
        is_test=True,
    )

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, None),
        label_map,
    )
