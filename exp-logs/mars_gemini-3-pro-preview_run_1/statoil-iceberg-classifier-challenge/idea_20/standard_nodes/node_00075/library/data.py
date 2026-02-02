import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def process_and_cache_data(load_cached_data=True):
    """
    Parses raw JSON data and metadata, converts to numpy arrays, and caches them.
    Or loads from cache if available.

    Returns:
        dict: Contains 'train', 'val', 'test' dictionaries with keys 'images', 'angles', 'ids', and 'labels' (for train/val).
    """
    cache_dir = Config.WORK_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "train": [
            "train_images.npy",
            "train_angles.npy",
            "train_labels.npy",
            "train_ids.npy",
        ],
        "val": ["val_images.npy", "val_angles.npy", "val_labels.npy", "val_ids.npy"],
        "test": ["test_images.npy", "test_angles.npy", "test_ids.npy"],
    }

    # Check if all files exist
    all_exist = True
    for split, flist in files.items():
        for fname in flist:
            if not os.path.exists(os.path.join(cache_dir, fname)):
                all_exist = False
                break

    if load_cached_data and all_exist:
        print("Loading data from cache...")
        data = {}
        for split in ["train", "val", "test"]:
            data[split] = {}
            data[split]["images"] = np.load(
                os.path.join(cache_dir, f"{split}_images.npy")
            )
            data[split]["angles"] = np.load(
                os.path.join(cache_dir, f"{split}_angles.npy")
            )
            data[split]["ids"] = np.load(
                os.path.join(cache_dir, f"{split}_ids.npy"), allow_pickle=True
            )
            if split != "test":
                data[split]["labels"] = np.load(
                    os.path.join(cache_dir, f"{split}_labels.npy")
                )
        return data

    print("Processing data from scratch...")

    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSON
    print(f"Loading {Config.TRAIN_JSON}...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train = json.load(f)

    print(f"Loading {Config.TEST_JSON}...")
    with open(Config.TEST_JSON, "r") as f:
        raw_test = json.load(f)

    # Helper to extract data based on metadata indices
    def extract_data(df_meta, raw_data, is_test=False):
        indices = df_meta["sample_index"].values
        ids = df_meta["id"].values

        images = []
        angles = []
        labels = []

        for i, idx in enumerate(indices):
            item = raw_data[idx]

            # Extract Bands (flattened 75x75 = 5625)
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            # Stack to (75, 75, 2)
            img = np.dstack((b1, b2))
            images.append(img)

            # Extract Angle
            # Handle 'na' by converting to None, will impute later
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            # Extract Label
            if not is_test:
                labels.append(item["is_iceberg"])

        return (
            np.array(images),
            np.array(angles),
            np.array(ids),
            np.array(labels) if not is_test else None,
        )

    # Process Splits
    train_imgs, train_angs, train_ids, train_lbls = extract_data(
        df_train_meta, raw_train
    )
    val_imgs, val_angs, val_ids, val_lbls = extract_data(df_val_meta, raw_train)
    test_imgs, test_angs, test_ids, _ = extract_data(
        df_test_meta, raw_test, is_test=True
    )

    # Impute missing angles with mean of training set (excluding NaNs)
    # Note: 'na' angles are only in train/val (derived from train.json) according to description,
    # but we should be robust.
    train_valid_mask = ~np.isnan(train_angs)
    angle_mean = np.mean(train_angs[train_valid_mask])

    train_angs[np.isnan(train_angs)] = angle_mean
    val_angs[np.isnan(val_angs)] = angle_mean
    test_angs[np.isnan(test_angs)] = angle_mean  # Just in case test has NaNs

    # Save to cache
    np.save(os.path.join(cache_dir, "train_images.npy"), train_imgs)
    np.save(os.path.join(cache_dir, "train_angles.npy"), train_angs)
    np.save(os.path.join(cache_dir, "train_labels.npy"), train_lbls)
    np.save(os.path.join(cache_dir, "train_ids.npy"), train_ids)

    np.save(os.path.join(cache_dir, "val_images.npy"), val_imgs)
    np.save(os.path.join(cache_dir, "val_angles.npy"), val_angs)
    np.save(os.path.join(cache_dir, "val_labels.npy"), val_lbls)
    np.save(os.path.join(cache_dir, "val_ids.npy"), val_ids)

    np.save(os.path.join(cache_dir, "test_images.npy"), test_imgs)
    np.save(os.path.join(cache_dir, "test_angles.npy"), test_angs)
    np.save(os.path.join(cache_dir, "test_ids.npy"), test_ids)

    data = {
        "train": {
            "images": train_imgs,
            "angles": train_angs,
            "labels": train_lbls,
            "ids": train_ids,
        },
        "val": {
            "images": val_imgs,
            "angles": val_angs,
            "labels": val_lbls,
            "ids": val_ids,
        },
        "test": {"images": test_imgs, "angles": test_angs, "ids": test_ids},
    }
    return data


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        # image is (75, 75, 2)
        image = self.images[idx]
        angle = self.angles[idx]

        # Independent Band Normalization
        # Band 1 (HH)
        b1 = image[:, :, 0]
        b1 = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)

        # Band 2 (HV)
        b2 = image[:, :, 1]
        b2 = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

        # Composite Band 3 (Average)
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        img_composite = np.dstack((b1, b2, b3))

        # Bicubic Upsampling to 224x224
        img_resized = cv2.resize(
            img_composite,
            (Config.IMG_SIZE, Config.IMG_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        # Ensure float32
        img_resized = img_resized.astype(np.float32)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform (shouldn't happen with get_transforms)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1))

        # Prepare Angle
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Prepare Label
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # (1,)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for train or val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                # Continuous Rotation +/- 20 degrees
                A.Rotate(limit=20, p=0.5),
                # Random Shift/Scale
                A.ShiftScaleRotate(
                    rotate_limit=0, shift_limit=0.1, scale_limit=0.1, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.
    """
    data = process_and_cache_data(load_cached_data=load_cached_data)

    train_dataset = IcebergDataset(
        images=data["train"]["images"],
        angles=data["train"]["angles"],
        labels=data["train"]["labels"],
        transform=get_transforms("train"),
    )

    val_dataset = IcebergDataset(
        images=data["val"]["images"],
        angles=data["val"]["angles"],
        labels=data["val"]["labels"],
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for testing.
    """
    data = process_and_cache_data(load_cached_data=load_cached_data)

    test_dataset = IcebergDataset(
        images=data["test"]["images"],
        angles=data["test"]["angles"],
        labels=None,  # No labels for test
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, data["test"]["ids"]
