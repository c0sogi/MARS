import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Supports hard labels for standard training and soft targets for knowledge distillation.
    """

    def __init__(
        self,
        images: np.ndarray,
        ids: np.ndarray,
        labels: np.ndarray = None,
        soft_targets: np.ndarray = None,
        transform: A.Compose = None,
    ):
        """
        Args:
            images (np.ndarray): Pre-processed image data (N, H, W, C).
            ids (np.ndarray): Image IDs corresponding to the images.
            labels (np.ndarray, optional): Hard class labels (indices).
            soft_targets (np.ndarray, optional): Soft targets (logits) for distillation.
            transform (A.Compose, optional): Albumentations transforms to apply.
        """
        self.images = images
        self.ids = ids
        self.labels = labels
        self.soft_targets = soft_targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve pre-loaded image
        image = self.images[idx]
        img_id = self.ids[idx]

        # Apply augmentations (Flip, Normalize)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = ToTensorV2()(image=image)["image"]

        sample = {"image": image, "id": img_id}

        # Add hard labels if available
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Add soft targets for distillation if available
        if self.soft_targets is not None:
            sample["soft_target"] = torch.tensor(
                self.soft_targets[idx], dtype=torch.float
            )

        return sample


def get_transforms(config: Config, mode: str = "train") -> A.Compose:
    """
    Returns the Albumentations transforms for the specified mode.

    Note: Geometric transforms (Resize, CenterCrop) are applied during the
    data loading/caching phase to ensure consistency and speed.
    This function handles stochastic augmentations and normalization.

    Args:
        config (Config): Configuration object.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                # Only Random Horizontal Flip as per strategy
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test (Deterministic)
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def _process_images(df: pd.DataFrame, input_dir: str, config: Config):
    """
    Reads images from disk, applies deterministic geometric preprocessing,
    and returns numpy arrays.

    Pipeline: Resize(274) -> CenterCrop(256)
    """
    img_list = []
    id_list = []

    # Define the deterministic geometric transform
    # SmallestMaxSize(274) ensures the aspect ratio is preserved while resizing
    # the smallest edge to 274. CenterCrop(256) extracts the central features.
    pre_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=config.resize_size),
            A.CenterCrop(height=config.image_size, width=config.image_size),
        ]
    )

    # Use OpenCV single-threaded to avoid contention with DataLoader workers
    cv2.setNumThreads(0)

    print(f"Processing {len(df)} images...")
    for idx, row in df.iterrows():
        # Debugging limit
        if config.debug and idx >= 100:
            break

        img_id = row["id"]
        # Metadata contains relative path, e.g., "train/xxx.jpg"
        file_path = row["file_path"]
        full_path = os.path.join(input_dir, file_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {full_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply geometric transform
        res = pre_transform(image=img)
        img_processed = res["image"]

        img_list.append(img_processed)
        id_list.append(img_id)

    return np.array(img_list, dtype=np.uint8), np.array(id_list)


def load_data(config: Config, load_cached_data: bool = True):
    """
    Loads training and test data with caching mechanism.
    Merges metadata/train.csv and metadata/val.csv to allow for custom Cross-Validation.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays for images, ids, and labels.
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(os.path.join(config.metadata_dir, "train.csv"))
    val_meta = pd.read_csv(os.path.join(config.metadata_dir, "val.csv"))
    test_meta = pd.read_csv(os.path.join(config.metadata_dir, "test.csv"))

    # Merge train and val for full 5-Fold CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Create Label Map (Breed -> Index)
    unique_breeds = sorted(full_train_meta["breed"].unique())
    label_map = {breed: i for i, breed in enumerate(unique_breeds)}

    # 3. Define Cache Paths
    # We cache the full training set and the test set separately
    cache_train_imgs = os.path.join(config.cache_dir, "train_val_imgs.npy")
    cache_train_ids = os.path.join(config.cache_dir, "train_val_ids.npy")
    cache_test_imgs = os.path.join(config.cache_dir, "test_imgs.npy")
    cache_test_ids = os.path.join(config.cache_dir, "test_ids.npy")

    # Ensure cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    # 4. Load or Process Train Data
    if (
        load_cached_data
        and os.path.exists(cache_train_imgs)
        and os.path.exists(cache_train_ids)
    ):
        print(f"Loading cached training data from {config.cache_dir}...")
        train_images = np.load(cache_train_imgs)
        train_ids = np.load(cache_train_ids)
    else:
        print("Processing and caching training data...")
        train_images, train_ids = _process_images(
            full_train_meta, config.input_dir, config
        )
        np.save(cache_train_imgs, train_images)
        np.save(cache_train_ids, train_ids)

    # 5. Load or Process Test Data
    if (
        load_cached_data
        and os.path.exists(cache_test_imgs)
        and os.path.exists(cache_test_ids)
    ):
        print(f"Loading cached test data from {config.cache_dir}...")
        test_images = np.load(cache_test_imgs)
        test_ids = np.load(cache_test_ids)
    else:
        print("Processing and caching test data...")
        test_images, test_ids = _process_images(test_meta, config.input_dir, config)
        np.save(cache_test_imgs, test_images)
        np.save(cache_test_ids, test_ids)

    # 6. Align Labels for Training Data
    # Map the loaded IDs to their breed labels
    id_to_breed = dict(zip(full_train_meta["id"], full_train_meta["breed"]))

    train_labels_list = []
    for img_id in train_ids:
        # Handle potential numpy string/bytes issues
        img_id_str = str(img_id) if not isinstance(img_id, str) else img_id

        breed = id_to_breed.get(img_id_str)
        if breed:
            train_labels_list.append(label_map[breed])
        else:
            # Fallback for debug mode or missing keys
            train_labels_list.append(-1)

    train_labels = np.array(train_labels_list, dtype=np.int64)

    return {
        "train_images": train_images,
        "train_ids": train_ids,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_ids": test_ids,
        "label_map": label_map,
    }
