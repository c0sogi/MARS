import os
import struct
import io
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# ==========================================
# BSON Constants & Parsing Utilities
# ==========================================
TYPE_DOC = 3
TYPE_ARRAY = 4
TYPE_BINARY = 5


def read_cstring(buffer, offset):
    """Reads a null-terminated string from the buffer."""
    end = offset
    while end < len(buffer) and buffer[end] != 0:
        end += 1
    return buffer[offset:end].decode("utf-8", errors="ignore"), end + 1


def skip_value(buffer, offset, dtype):
    """Calculates the new offset after skipping a value of a given BSON type."""
    if dtype == 1:  # DOUBLE
        return offset + 8
    elif dtype == 2:  # STRING
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + l
    elif dtype == 3:  # DOC
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == 4:  # ARRAY
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == 5:  # BINARY
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + 1 + l
    elif dtype == 8:  # BOOL
        return offset + 1
    elif dtype == 16:  # INT32
        return offset + 4
    elif dtype == 18:  # INT64
        return offset + 8
    elif dtype == 7:  # OBJECT_ID
        return offset + 12
    elif dtype == 9:  # DATETIME
        return offset + 8
    elif dtype == 10:  # NULL
        return offset
    else:
        return offset


def extract_images_from_record(data):
    """
    Parses a single raw BSON record and extracts image binary data.
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
                            # subtype = data[sub_offset] # Skip subtype read
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
# Data Loading & Caching
# ==========================================


def load_metadata(csv_path, cache_dir, load_cached_data=True):
    """
    Loads metadata CSV with caching mechanism using Parquet.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(".csv", ".parquet")
    cache_path = os.path.join(cache_dir, filename)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to loading from CSV if cache is corrupt

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # If saving fails, just return the df

    return df


# ==========================================
# Dataset Class
# ==========================================


class BSONDataset(Dataset):
    def __init__(self, metadata_path, mode="train", transform=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, uses a small subset of data.
        """
        self.mode = mode
        self.transform = transform
        self.metadata = load_metadata(
            metadata_path, Config.WORKING_DIR, load_cached_data=True
        )

        if debug:
            self.metadata = self.metadata.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(
                drop=True
            )

        # Pre-compute paths to avoid joining strings in the loop
        # The metadata contains 'file_path' relative to input dir
        self.metadata["full_path"] = self.metadata["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        self.file_handles = {}  # Cache file handles per process

    def __len__(self):
        return len(self.metadata)

    def _get_file_handle(self, path):
        # Lazy initialization of file handle
        if path not in self.file_handles:
            self.file_handles[path] = open(path, "rb")
        return self.file_handles[path]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        path = row["full_path"]

        # Handle labels
        category_id = row["category_id"]
        if pd.isna(category_id):
            label = -1
        else:
            label = int(category_id)

        # Read BSON record
        try:
            f = self._get_file_handle(path)
            f.seek(offset)
            data = f.read(length)

            # Extract image bytes
            img_bytes_list = extract_images_from_record(data)

            # Decode images
            images = []
            for b in img_bytes_list:
                # Decode from buffer
                nparr = np.frombuffer(b, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)

            if len(images) == 0:
                # Fallback for empty/corrupt image records (should be rare)
                # Create a black image
                images.append(
                    np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
                )

        except Exception as e:
            # Fallback for read errors
            images = [np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)]

        # Apply Transforms and Return
        if self.mode == "train":
            # Select ONE image randomly for training
            img = images[np.random.randint(len(images))]

            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]

            return img, label

        else:
            # Return ALL images for val/test
            processed_images = []
            for img in images:
                if self.transform:
                    augmented = self.transform(image=img)
                    processed_images.append(augmented["image"])
                else:
                    # Fallback to simple tensor conversion if no transform provided
                    processed_images.append(
                        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    )

            # Stack images: (N_imgs, C, H, W)
            imgs_tensor = torch.stack(processed_images)
            product_id = int(row["product_id"])

            return imgs_tensor, label, product_id


# ==========================================
# Transforms
# ==========================================


def get_transforms(mode="train", img_size=180):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225],  # ImageNet std
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


# ==========================================
# Collate Functions
# ==========================================


def train_collate_fn(batch):
    """
    Standard collate for training where each sample is (image, label).
    """
    return torch.utils.data.dataloader.default_collate(batch)


def eval_collate_fn(batch):
    """
    Custom collate for validation/test where each sample is (images_tensor, label, product_id).
    images_tensor shape: (K, C, H, W) where K varies per product.

    Returns:
        flat_images: (Total_K, C, H, W)
        labels: (Batch_Size,)
        product_ids: (Batch_Size,)
        counts: (Batch_Size,) - Number of images per product
    """
    images_list = []
    labels_list = []
    ids_list = []
    counts_list = []

    for imgs, label, pid in batch:
        images_list.append(imgs)
        labels_list.append(label)
        ids_list.append(pid)
        counts_list.append(imgs.shape[0])

    # Concatenate all images into a single batch dimension
    flat_images = torch.cat(images_list, dim=0)

    labels = torch.tensor(labels_list, dtype=torch.long)
    product_ids = torch.tensor(ids_list, dtype=torch.long)
    counts = torch.tensor(counts_list, dtype=torch.long)

    return flat_images, labels, product_ids, counts
