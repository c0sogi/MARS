import os
import struct
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from library.config import Config


def get_val_size(type_byte, data, ptr):
    """Returns the size of a BSON value based on its type byte."""
    if type_byte == 0x01:  # double
        return 8
    elif type_byte == 0x02:  # string
        s_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + s_len
    elif type_byte == 0x03:  # document
        d_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return d_len
    elif type_byte == 0x04:  # array
        a_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return a_len
    elif type_byte == 0x05:  # binary
        b_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + 1 + b_len
    elif type_byte == 0x07:  # objectid
        return 12
    elif type_byte == 0x08:  # boolean
        return 1
    elif type_byte == 0x09:  # utc datetime
        return 8
    elif type_byte == 0x0A:  # null
        return 0
    elif type_byte == 0x10:  # int32
        return 4
    elif type_byte == 0x12:  # int64
        return 8
    else:
        return 0


def extract_images_from_bytes(data):
    """
    Parses a raw BSON byte string to find the 'imgs' array and extract 'picture' binaries.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        # name = data[ptr:name_end].decode("utf-8", errors="ignore") # Optimization: Don't decode unless necessary
        # We only care if name is "imgs"
        is_imgs = data[ptr:name_end] == b"imgs"
        ptr = name_end + 1

        if is_imgs and type_byte == 0x04:
            # Found 'imgs' array
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        is_picture = data[dp:dname_end] == b"picture"
                        dp = dname_end + 1

                        if is_picture and dtype == 0x05:
                            # Found picture binary
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            # Skip other fields in image doc
                            v_len = get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


class RawBSONDataset(Dataset):
    """
    Dataset that reads raw BSON files using random access based on metadata offsets.
    Returns all images associated with a product.
    """

    def __init__(self, metadata_path, bson_path, transform=None, mode="train"):
        self.bson_path = bson_path
        self.transform = transform
        self.mode = mode

        # Load metadata
        self.meta = pd.read_csv(metadata_path)

        # Debug Mode: Slice dataset
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Reducing {mode} dataset to {Config.DEBUG_SAMPLES} samples."
            )
            self.meta = self.meta.iloc[: Config.DEBUG_SAMPLES].copy()

        # Pre-convert columns to numpy for faster access
        self.offsets = self.meta["bson_offset"].values.astype(np.int64)
        self.lengths = self.meta["bson_length"].values.astype(np.int32)
        self.ids = self.meta["_id"].values.astype(np.int64)

        if "category_id" in self.meta.columns:
            self.labels = self.meta["category_id"].values.astype(np.int64)
        else:
            self.labels = np.full(len(self.meta), -1, dtype=np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        length = self.lengths[idx]
        _id = self.ids[idx]
        label = self.labels[idx]

        # Read BSON chunk directly from file
        # Opening inside __getitem__ is safe for multiprocessing workers
        with open(self.bson_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        # Extract raw image bytes
        img_binaries = extract_images_from_bytes(data)

        images = []
        for img_bytes in img_binaries:
            # Decode using OpenCV (faster than PIL)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to PIL for standard torchvision transforms
            img_pil = Image.fromarray(img)

            if self.transform:
                img_t = self.transform(img_pil)
            else:
                # Fallback: Convert to tensor [C, H, W] and normalize to [0, 1]
                img_t = (
                    torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float()
                    / 255.0
                )

            images.append(img_t)

        # Handle case with no valid images (should be rare/impossible in this dataset)
        if len(images) == 0:
            dummy_np = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_np)
            if self.transform:
                images.append(self.transform(dummy_pil))
            else:
                images.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))

        # Stack images: (Num_Images, C, H, W)
        images_tensor = torch.stack(images)

        return images_tensor, _id, label


def product_collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.

    Args:
        batch: List of tuples (images_tensor, _id, label)
               images_tensor shape: (N, C, H, W) where N varies (1-4)

    Returns:
        flat_images: (Total_Images, C, H, W) - Flattened batch for GPU processing
        flat_ids: (Total_Images,) - Product ID repeated for each image
        flat_labels: (Total_Images,) - Category ID repeated for each image
        sizes: List[int] - Number of images per product (used for reconstruction/pooling)
    """
    batch_images = []
    batch_ids = []
    batch_labels = []
    sizes = []

    for images, _id, label in batch:
        n = images.shape[0]
        batch_images.append(images)
        batch_ids.extend([_id] * n)
        batch_labels.extend([label] * n)
        sizes.append(n)

    flat_images = torch.cat(batch_images, dim=0)
    flat_ids = torch.tensor(batch_ids, dtype=torch.int64)
    flat_labels = torch.tensor(batch_labels, dtype=torch.int64)

    return flat_images, flat_ids, flat_labels, sizes


def get_bson_loader(
    metadata_path,
    bson_path,
    batch_size,
    transform,
    mode="train",
    num_workers=4,
    shuffle=False,
):
    """
    Creates a DataLoader for the BSON dataset.
    """
    dataset = RawBSONDataset(metadata_path, bson_path, transform=transform, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=product_collate_fn,
        pin_memory=True,
    )

    return loader
