import os
import struct
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    CATEGORY_CLASSES,
    IMG_SIZE,
    SEED,
    TRAIN_META,
    VAL_META,
    TEST_META,
    TRAIN_BSON,
    TEST_BSON,
    CACHE_DIR,
)

# Ensure reproducibility
import random

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CategoryEncoder:
    """
    Encodes raw category_id integers to contiguous indices [0, num_classes-1].
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, y):
        """Fit the encoder on a list/array of category_ids."""
        self.classes_ = np.unique(y)
        self.classes_.sort()
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """Transform category_ids to indices."""
        if self.class_to_idx_ is None:
            raise ValueError("Encoder not fitted.")
        return np.array([self.class_to_idx_.get(x, -1) for x in y], dtype=np.int64)

    def inverse_transform(self, y):
        """Transform indices back to category_ids."""
        if self.classes_ is None:
            raise ValueError("Encoder not fitted.")
        return self.classes_[y]

    def save(self, path):
        """Save classes to a .npy file."""
        if self.classes_ is None:
            raise ValueError("Encoder not fitted.")
        np.save(path, self.classes_)

    def load(self, path):
        """Load classes from a .npy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.classes_ = np.load(path)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def __len__(self):
        return len(self.classes_) if self.classes_ is not None else 0


def get_category_encoder(load_cached_data=True):
    """
    Factory function to get a fitted CategoryEncoder.
    Follows strict caching logic: Load if available/requested, else compute and save.
    """
    encoder = CategoryEncoder()

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(CATEGORY_CLASSES):
            try:
                encoder.load(CATEGORY_CLASSES)
                return encoder
            except Exception as e:
                print(f"Failed to load cached encoder: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Fitting CategoryEncoder from training metadata...")
    if not os.path.exists(TRAIN_META):
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_META}")

    train_df = pd.read_csv(TRAIN_META)
    encoder.fit(train_df["category_id"].values)

    # 3. Save to cache
    os.makedirs(os.path.dirname(CATEGORY_CLASSES), exist_ok=True)
    encoder.save(CATEGORY_CLASSES)
    print(f"CategoryEncoder saved to {CATEGORY_CLASSES}")

    return encoder


def get_val_size(type_byte, data, ptr):
    """Helper to determine BSON value size."""
    if type_byte == 0x01:
        return 8
    elif type_byte == 0x02:
        return 4 + struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x03:
        return struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x04:
        return struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x05:
        return 4 + 1 + struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x07:
        return 12
    elif type_byte == 0x08:
        return 1
    elif type_byte == 0x09:
        return 8
    elif type_byte == 0x0A:
        return 0
    elif type_byte == 0x10:
        return 4
    elif type_byte == 0x12:
        return 8
    return 0


def extract_images_from_bson(data):
    """
    Parses a raw BSON document bytes to find 'imgs' array and extract 'picture' binaries.
    Returns a list of byte strings (JPEGs).
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        if name_end == -1:
            break
        name = data[ptr:name_end].decode("utf-8", errors="ignore")
        ptr = name_end + 1

        if name == "imgs" and type_byte == 0x04:
            # Found 'imgs' array
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len
            ap = ptr + 4  # Enter Array

            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1
                ename_end = data.find(b"\x00", ap)
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len
                    dp = ap + 4  # Enter Document

                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1
                        dname_end = data.find(b"\x00", dp)
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype at dp+4, data at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            dp += get_val_size(dtype, data, dp)
                    ap += doc_len
                else:
                    ap += get_val_size(etype, data, ap)
            ptr += arr_len
        else:
            ptr += get_val_size(type_byte, data, ptr)

    return images


class RawImageDataset(Dataset):
    """
    Dataset that reads images directly from BSON files using metadata offsets.
    Used for feature extraction.
    Returns: (images_tensor, category_idx, product_id)
    images_tensor shape: (Num_Images, 3, H, W)
    """

    def __init__(self, metadata_df, bson_path, encoder=None, transform=None):
        self.metadata = metadata_df
        self.bson_path = bson_path
        self.encoder = encoder
        self.file_handle = None

        if transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        _id = row["_id"]

        # Lazy file opening for multiprocessing safety
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        self.file_handle.seek(offset)
        data = self.file_handle.read(length)

        # Extract images
        img_binaries = extract_images_from_bson(data)

        processed_images = []
        for img_bytes in img_binaries:
            # Decode
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Augment/Transform
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]

            processed_images.append(img)

        if not processed_images:
            # Fallback for corrupt/empty images: Black image
            # Shape: (3, H, W)
            fallback = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
            processed_images.append(fallback)

        # Stack images: (N, 3, H, W)
        images_tensor = torch.stack(processed_images)

        # Label
        label = -1
        if "category_id" in row and self.encoder:
            label = self.encoder.transform([row["category_id"]])[0]

        return images_tensor, label, _id


class FeatureDataset(Dataset):
    """
    Dataset for training the classifier on pre-computed features.
    Features are expected to be loaded in RAM (numpy arrays).
    """

    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        else:
            return x
