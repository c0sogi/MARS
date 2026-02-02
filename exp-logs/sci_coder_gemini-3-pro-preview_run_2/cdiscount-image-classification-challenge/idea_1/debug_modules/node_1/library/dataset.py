import os
import struct
import io
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import (
    TRAIN_BSON,
    TEST_BSON,
    TRAIN_META,
    VAL_META,
    TEST_META,
    CATEGORY_NAMES,
    INPUT_SIZE,
    DEBUG_SAMPLE_SIZE,
)

# ==========================================
# BSON Parsing Utilities
# ==========================================
TYPE_DOUBLE = 1
TYPE_STRING = 2
TYPE_DOC = 3
TYPE_ARRAY = 4
TYPE_BINARY = 5
TYPE_BOOL = 8
TYPE_INT32 = 16
TYPE_INT64 = 18
TYPE_OBJECT_ID = 7
TYPE_DATETIME = 9
TYPE_NULL = 10


def read_cstring(buffer, offset):
    """Reads a null-terminated string from the buffer."""
    end = offset
    while end < len(buffer) and buffer[end] != 0:
        end += 1
    return buffer[offset:end].decode("utf-8", errors="ignore"), end + 1


def skip_value(buffer, offset, dtype):
    """Calculates the new offset after skipping a value of a given BSON type."""
    if dtype == TYPE_DOUBLE:
        return offset + 8
    elif dtype == TYPE_STRING:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + l
    elif dtype == TYPE_DOC:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_ARRAY:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_BINARY:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + 1 + l
    elif dtype == TYPE_BOOL:
        return offset + 1
    elif dtype == TYPE_INT32:
        return offset + 4
    elif dtype == TYPE_INT64:
        return offset + 8
    elif dtype == TYPE_OBJECT_ID:
        return offset + 12
    elif dtype == TYPE_DATETIME:
        return offset + 8
    elif dtype == TYPE_NULL:
        return offset
    else:
        return offset


def extract_images_from_bytes(data):
    """
    Parses a single raw BSON record bytes and extracts image binary data.
    Returns a list of bytes objects (JPEG data).
    """
    images = []
    offset = 4  # Skip total size header
    length = len(data)

    while offset < length - 1:
        dtype = data[offset]
        offset += 1
        key, offset = read_cstring(data, offset)

        if key == "imgs" and dtype == TYPE_ARRAY:
            arr_size = struct.unpack_from("<i", data, offset)[0]
            arr_end = offset + arr_size
            offset += 4

            while offset < arr_end - 1:
                e_type = data[offset]
                offset += 1
                e_key, offset = read_cstring(data, offset)

                if e_type == TYPE_DOC:
                    doc_size = struct.unpack_from("<i", data, offset)[0]
                    doc_end = offset + doc_size
                    sub_offset = offset + 4
                    while sub_offset < doc_end - 1:
                        s_type = data[sub_offset]
                        sub_offset += 1
                        s_key, sub_offset = read_cstring(data, sub_offset)

                        if s_key == "picture" and s_type == TYPE_BINARY:
                            b_len = struct.unpack_from("<i", data, sub_offset)[0]
                            sub_offset += 4
                            # subtype = data[sub_offset] # skip subtype read
                            sub_offset += 1
                            img_data = data[sub_offset : sub_offset + b_len]
                            images.append(img_data)
                            sub_offset += b_len
                        else:
                            sub_offset = skip_value(data, sub_offset, s_type)
                    offset = doc_end
                else:
                    offset = skip_value(data, offset, e_type)
        else:
            offset = skip_value(data, offset, dtype)

    return images


# ==========================================
# Dataset Class
# ==========================================
class CdiscountDataset(Dataset):
    def __init__(self, mode="train", transform=None, debug_size=DEBUG_SAMPLE_SIZE):
        self.mode = mode
        self.transform = transform
        self.file_handle = None

        # 1. Load Category Mapping
        # We need a consistent mapping from category_id to class_index (0..N-1)
        df_cats = pd.read_csv(CATEGORY_NAMES)
        self.sorted_categories = sorted(df_cats["category_id"].unique())
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.sorted_categories)}
        self.idx_to_cat = {i: cat for i, cat in enumerate(self.sorted_categories)}

        # 2. Determine Metadata and Source File
        if mode == "train":
            meta_path = TRAIN_META
            self.bson_path = TRAIN_BSON
        elif mode == "val":
            meta_path = VAL_META
            self.bson_path = TRAIN_BSON
        elif mode == "test":
            meta_path = TEST_META
            self.bson_path = TEST_BSON
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3. Load Metadata
        self.meta = pd.read_csv(meta_path)

        # 4. Handle Debugging
        if debug_size is not None:
            if len(self.meta) > debug_size:
                # Use a fixed seed for reproducible subsampling
                self.meta = self.meta.sample(n=debug_size, random_state=42).reset_index(
                    drop=True
                )

    def _get_handle(self):
        # Lazy loading of file handle to support multiprocessing
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")
        return self.file_handle

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        product_id = row["product_id"]

        # Get label (if available)
        category_id = row["category_id"]
        if pd.notna(category_id):
            target = self.cat_to_idx.get(int(category_id), -1)
        else:
            target = -1  # Dummy target for test set

        # Read BSON record
        f = self._get_handle()
        f.seek(offset)
        data = f.read(length)

        # Extract images
        img_bytes_list = extract_images_from_bytes(data)

        images = []
        for img_bytes in img_bytes_list:
            # Decode image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            # Convert BGR (OpenCV) to RGB
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transforms
            if self.transform:
                # Convert to PIL for torchvision transforms if needed,
                # but ToTensor handles numpy arrays.
                # However, Resize expects PIL or Tensor.
                # Let's convert to PIL to be safe with all torchvision transforms.
                img_pil = transforms.ToPILImage()(img)
                img_t = self.transform(img_pil)
                images.append(img_t)
            else:
                # Fallback to tensor
                images.append(transforms.ToTensor()(img))

        # Handle case with no valid images (should be rare/impossible in this dataset)
        if len(images) == 0:
            # Return a black image
            images.append(torch.zeros(3, INPUT_SIZE, INPUT_SIZE))

        return images, target, product_id


# ==========================================
# Transforms
# ==========================================
def get_transforms(mode="train"):
    """
    Returns the torchvision transforms for the dataset.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


# ==========================================
# Collate Functions
# ==========================================
def train_collate_fn(batch):
    """
    Collates a batch of products into a flattened batch of images.

    Input: List of tuples (images_list, target, product_id)
    Output: (flattened_images, replicated_targets)
    """
    all_images = []
    all_targets = []

    for images, target, _ in batch:
        for img in images:
            all_images.append(img)
            all_targets.append(target)

    return torch.stack(all_images), torch.tensor(all_targets, dtype=torch.long)


def eval_collate_fn(batch):
    """
    Collates a batch of products for evaluation, preserving structure.

    Input: List of tuples (images_list, target, product_id)
    Output:
        - flattened_images: Tensor [Total_Images, C, H, W]
        - targets: Tensor [Batch_Size]
        - product_ids: Tensor [Batch_Size]
        - num_imgs: Tensor [Batch_Size] (Number of images per product)
    """
    all_images = []
    targets = []
    product_ids = []
    num_imgs = []

    for images, target, pid in batch:
        all_images.extend(images)
        targets.append(target)
        product_ids.append(pid)
        num_imgs.append(len(images))

    return (
        torch.stack(all_images),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(product_ids, dtype=torch.long),
        torch.tensor(num_imgs, dtype=torch.long),
    )
