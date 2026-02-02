import struct
import cv2
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
import library.config as config


def get_val_size(type_byte, data, ptr):
    """
    Returns the size of a BSON value based on its type byte.
    Used internally by extract_images_from_bson.
    """
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


def extract_images_from_bson(data):
    """
    Parses a raw BSON document to find the 'imgs' array and extract 'picture' binaries.
    Optimized to scan byte arrays directly.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        # Check if field name is "imgs"
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
                        # Check if field name is "picture"
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


def process_image(img_bytes, img_size=config.IMG_SIZE):
    """
    Decodes, resizes, and normalizes an image for model input.

    Args:
        img_bytes (bytes): Raw binary image data (JPEG).
        img_size (int): Target height/width.

    Returns:
        torch.Tensor: Normalized image tensor (3, H, W).
    """
    if not img_bytes:
        return torch.zeros((3, img_size, img_size), dtype=torch.float32)

    # Decode using OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        return torch.zeros((3, img_size, img_size), dtype=torch.float32)

    # Handle grayscale or BGR -> RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    # Using INTER_LINEAR for efficiency as we are generally upsampling (180->224)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize
    # Convert [0, 255] -> [0.0, 1.0] and HWC -> CHW
    tensor = transforms.functional.to_tensor(img)

    # Standard ImageNet normalization
    tensor = transforms.functional.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return tensor


class BSONIterator:
    """
    Iterates over a BSON file using a metadata DataFrame to seek directly to records.
    Yields processed image tensors and metadata.
    """

    def __init__(self, bson_path, metadata_df, img_size=config.IMG_SIZE):
        """
        Args:
            bson_path (str): Path to the .bson file.
            metadata_df (pd.DataFrame): DataFrame containing '_id', 'bson_offset', 'bson_length'.
            img_size (int): Target image size.
        """
        self.bson_path = bson_path
        self.metadata = metadata_df
        self.img_size = img_size

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        # Rename _id to product_id because namedtuple fields cannot start with underscore.
        iter_df = self.metadata.rename(columns={"_id": "product_id"})

        with open(self.bson_path, "rb") as f:
            # Use itertuples for faster iteration than iterrows
            for row in iter_df.itertuples(index=False):
                offset = row.bson_offset
                length = row.bson_length

                # Seek and read specific record
                f.seek(offset)
                doc_data = f.read(length)

                # Extract raw image bytes
                img_binaries = extract_images_from_bson(doc_data)

                # Process all images for this product
                # Result is a list of tensors [Tensor(3, H, W), ...]
                processed_imgs = [process_image(b, self.img_size) for b in img_binaries]

                # Get ID
                _id = row.product_id

                # Handle category_id if it exists (train/val), else None (test)
                category_id = getattr(row, "category_id", None)

                yield _id, processed_imgs, category_id
