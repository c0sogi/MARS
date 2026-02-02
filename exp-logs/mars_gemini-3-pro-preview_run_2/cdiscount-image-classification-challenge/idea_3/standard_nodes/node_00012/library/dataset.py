import os
import struct
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# ==== BSON Constants & Helpers ====
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
    if offset >= len(buffer):
        return len(buffer)

    if dtype == TYPE_DOUBLE:
        return offset + 8
    elif dtype == TYPE_STRING:
        if offset + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + l
    elif dtype == TYPE_DOC:
        if offset + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_ARRAY:
        if offset + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_BINARY:
        if offset + 4 > len(buffer):
            return len(buffer)
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


def extract_images_from_record(data):
    """Parses a single raw BSON record and extracts image binary data."""
    images = []
    offset = 4
    length = len(data)
    while offset < length - 1:
        dtype = data[offset]
        offset += 1
        key, offset = read_cstring(data, offset)
        if key == "imgs" and dtype == TYPE_ARRAY:
            if offset + 4 > len(data):
                break
            arr_size = struct.unpack_from("<i", data, offset)[0]
            arr_end = offset + arr_size
            offset += 4
            while offset < arr_end - 1:
                if offset >= len(data):
                    break
                e_type = data[offset]
                offset += 1
                e_key, offset = read_cstring(data, offset)
                if e_type == TYPE_DOC:
                    if offset + 4 > len(data):
                        break
                    doc_size = struct.unpack_from("<i", data, offset)[0]
                    doc_end = offset + doc_size
                    sub_offset = offset + 4
                    while sub_offset < doc_end - 1:
                        if sub_offset >= len(data):
                            break
                        s_type = data[sub_offset]
                        sub_offset += 1
                        s_key, sub_offset = read_cstring(data, sub_offset)
                        if s_key == "picture" and s_type == TYPE_BINARY:
                            if sub_offset + 4 > len(data):
                                break
                            b_len = struct.unpack_from("<i", data, sub_offset)[0]
                            sub_offset += 4
                            # subtype = data[sub_offset] # unused
                            sub_offset += 1
                            if sub_offset + b_len > len(data):
                                break
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


def get_category_mapping(load_cached_data=True):
    """
    Creates or loads a mapping from category_id to integer index.
    """
    cache_path = os.path.join(Config.IDEA_DIR, "category_map.npy")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True).item()
        except:
            pass

    # Compute mapping from category_names.csv
    df = pd.read_csv(Config.CATEGORY_NAMES)
    cats = sorted(df["category_id"].unique())
    mapping = {c: i for i, c in enumerate(cats)}

    # Save to cache
    np.save(cache_path, mapping)

    return mapping


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for the given split.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class BSONDataset(Dataset):
    def __init__(
        self, metadata_csv, bson_file, split="train", transform=None, debug_size=None
    ):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file.
            bson_file (str): Path to the BSON file.
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            debug_size (int): If set, only use this many samples.
        """
        self.meta = pd.read_csv(metadata_csv)
        if debug_size:
            self.meta = self.meta.iloc[:debug_size]

        self.bson_file = bson_file
        self.split = split
        self.transform = transform or get_transforms(split)
        self.file_handle = None

        # Load Label mapping
        self.cat_map = get_category_mapping(load_cached_data=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Lazy file opening for multiprocessing safety
        if self.file_handle is None:
            self.file_handle = open(self.bson_file, "rb")

        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        pid = row["product_id"]

        # Read BSON record
        self.file_handle.seek(offset)
        data = self.file_handle.read(length)

        # Extract images
        img_bytes_list = extract_images_from_record(data)

        images = []
        for img_bytes in img_bytes_list:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(image=img)["image"]
            images.append(img)

        # Handle case with no valid images (rare but possible)
        if len(images) == 0:
            # Create a blank image
            blank = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
            if self.transform:
                blank = self.transform(image=blank)["image"]
            images.append(blank)

        # Get Label
        cat_id = row["category_id"]
        if pd.isna(cat_id):
            label = -1
        else:
            label = self.cat_map.get(int(cat_id), -1)

        return images, label, pid


def collate_flatten(batch):
    """
    Flattens the list of images per product into a single batch of images.
    Used for training to treat every image as an independent sample.

    Returns:
        images_tensor: (N_total_images, C, H, W)
        labels_tensor: (N_total_images,)
    """
    all_images = []
    all_labels = []

    for images, label, pid in batch:
        for img in images:
            all_images.append(img)
            all_labels.append(label)

    return torch.stack(all_images), torch.tensor(all_labels, dtype=torch.long)


def collate_product(batch):
    """
    Maintains product grouping but stacks images for batch processing.
    Used for validation/inference to aggregate predictions by product.

    Returns:
        images_tensor: (N_total_images, C, H, W)
        product_ids_tensor: (N_total_images,)
        labels_tensor: (N_total_images,)
        sizes_tensor: (Batch_Size,) - Number of images per product
    """
    all_images = []
    all_pids = []
    all_labels = []
    sizes = []

    for images, label, pid in batch:
        sizes.append(len(images))
        for img in images:
            all_images.append(img)
            all_pids.append(pid)
            all_labels.append(label)

    return (
        torch.stack(all_images),
        torch.tensor(all_pids, dtype=torch.int64),
        torch.tensor(all_labels, dtype=torch.long),
        torch.tensor(sizes, dtype=torch.int32),
    )
