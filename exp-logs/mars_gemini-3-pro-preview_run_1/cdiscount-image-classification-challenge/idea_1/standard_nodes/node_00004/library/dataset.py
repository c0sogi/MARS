import os
import io
import struct
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from library.config import Config

# ==========================================
# BSON CONSTANTS & PARSING HELPERS
# ==========================================
BSON_TYPE_DOUBLE = 1
BSON_TYPE_STRING = 2
BSON_TYPE_OBJECT = 3
BSON_TYPE_ARRAY = 4
BSON_TYPE_BINARY = 5
BSON_TYPE_OBJECTID = 7
BSON_TYPE_BOOL = 8
BSON_TYPE_DATE = 9
BSON_TYPE_NULL = 10
BSON_TYPE_INT32 = 16
BSON_TYPE_INT64 = 18


def read_c_string(buffer, start):
    """Reads a C-style null-terminated string from buffer."""
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def skip_bson_value(buffer, idx, type_byte):
    """Helper to skip a BSON value based on its type."""
    buf_len = len(buffer)
    if idx >= buf_len:
        return buf_len

    if type_byte == BSON_TYPE_DOUBLE:
        return idx + 8
    elif type_byte == BSON_TYPE_STRING:
        if idx + 4 > buf_len:
            return buf_len
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + l
    elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
        if idx + 4 > buf_len:
            return buf_len
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + l
    elif type_byte == BSON_TYPE_BINARY:
        if idx + 4 > buf_len:
            return buf_len
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + 1 + l
    elif type_byte == BSON_TYPE_OBJECTID:
        return idx + 12
    elif type_byte == BSON_TYPE_BOOL:
        return idx + 1
    elif type_byte == BSON_TYPE_DATE:
        return idx + 8
    elif type_byte == BSON_TYPE_NULL:
        return idx
    elif type_byte == BSON_TYPE_INT32:
        return idx + 4
    elif type_byte == BSON_TYPE_INT64:
        return idx + 8
    else:
        return buf_len


def extract_images_from_bson(buffer):
    """
    Parses a BSON document buffer to extract image binary data from the 'imgs' array.
    Returns a list of binary image data.
    """
    images = []
    idx = 0
    buf_len = len(buffer)

    # Skip initial size if present (standard BSON starts with int32 size)
    # We assume the buffer passed is the full document.
    if buf_len >= 4:
        size = struct.unpack("<i", buffer[0:4])[0]
        if size == buf_len:
            idx = 4

    while idx < buf_len - 1:
        type_byte = buffer[idx]
        idx += 1

        name, idx = read_c_string(buffer, idx)
        if idx == -1:
            break

        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Enter array
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = buffer[a_idx]
                a_idx += 1
                e_name, a_idx = read_c_string(buffer, a_idx)

                if e_type == BSON_TYPE_OBJECT:
                    o_len = struct.unpack("<i", buffer[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len
                    o_curr = a_idx + 4

                    while o_curr < o_end - 1:
                        p_type = buffer[o_curr]
                        o_curr += 1
                        p_name, o_curr = read_c_string(buffer, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", buffer[o_curr : o_curr + 4])[0]
                            # subtype is at o_curr + 4, data starts at o_curr + 5
                            img_data = buffer[o_curr + 5 : o_curr + 5 + b_len]
                            images.append(img_data)
                            o_curr += 5 + b_len
                        else:
                            o_curr = skip_bson_value(buffer, o_curr, p_type)
                    a_idx = o_end
                else:
                    a_idx = skip_bson_value(buffer, a_idx, e_type)
            return images
        else:
            idx = skip_bson_value(buffer, idx, type_byte)

    return images


def get_category_mapping():
    """
    Creates a deterministic mapping between category_id and class index (0 to N-1).
    Reads from category_names.csv.
    """
    df = pd.read_csv(Config.CATEGORY_NAMES)
    # Sort to ensure deterministic order
    unique_cats = sorted(df["category_id"].unique())

    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    idx_to_cat = {i: cat for i, cat in enumerate(unique_cats)}

    return cat_to_idx, idx_to_cat


# ==========================================
# DATASET CLASS
# ==========================================
class CdiscountDataset(Dataset):
    def __init__(self, metadata_path, bson_path, transform=None, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            bson_path (str): Path to the BSON file.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.bson_path = bson_path
        self.transform = transform
        self.mode = mode
        self.file_handle = None

        # Load category mapping if not in test mode (or if needed for reference)
        self.cat_to_idx, self.idx_to_cat = get_category_mapping()

        # Pre-check if category_id exists in metadata for train/val
        self.has_labels = "category_id" in self.metadata.columns

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Lazy file opening (per worker)
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        # Retrieve metadata for the sample
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        sample_id = row["sample_id"]

        # Read BSON document
        self.file_handle.seek(offset)
        doc_bytes = self.file_handle.read(length)

        # Extract Images
        img_binaries = extract_images_from_bson(doc_bytes)

        # Decode and Transform Images
        images_tensors = []
        for b_data in img_binaries:
            try:
                img = Image.open(io.BytesIO(b_data))
                img = img.convert("RGB")  # Ensure 3 channels
                if self.transform:
                    img_t = self.transform(img)
                    images_tensors.append(img_t)
            except Exception:
                # In case of corrupt image, skip it
                continue

        # Handle case with no valid images (rare, but possible)
        if not images_tensors:
            # Create a black image as placeholder
            placeholder = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))
            if self.transform:
                images_tensors.append(self.transform(placeholder))
            else:
                # Fallback if no transform
                images_tensors.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))

        # Stack images: (N, C, H, W)
        images = torch.stack(images_tensors)

        # Get Target
        target = -1
        if self.has_labels:
            cat_id = row["category_id"]
            target = self.cat_to_idx.get(cat_id, -1)

        return images, target, sample_id


# ==========================================
# COLLATE FUNCTION
# ==========================================
def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.

    Args:
        batch: List of tuples (images, target, sample_id)
               images shape: (N, C, H, W)

    Returns:
        batch_images: (Total_Images, C, H, W)
        batch_indices: (Total_Images,) - maps each image to its sample index in the batch
        targets: (Batch_Size,)
        sample_ids: (Batch_Size,)
    """
    images_list = []
    indices_list = []
    targets_list = []
    ids_list = []

    for i, (imgs, tgt, sid) in enumerate(batch):
        # imgs is (N, C, H, W)
        images_list.append(imgs)

        # Create index mapping for these N images to sample i
        n_imgs = imgs.shape[0]
        indices_list.append(torch.full((n_imgs,), i, dtype=torch.long))

        targets_list.append(tgt)
        ids_list.append(sid)

    # Concatenate all images into a single large batch
    batch_images = torch.cat(images_list, dim=0)
    batch_indices = torch.cat(indices_list, dim=0)

    targets = torch.tensor(targets_list, dtype=torch.long)
    sample_ids = torch.tensor(ids_list, dtype=torch.long)

    return batch_images, batch_indices, targets, sample_ids
