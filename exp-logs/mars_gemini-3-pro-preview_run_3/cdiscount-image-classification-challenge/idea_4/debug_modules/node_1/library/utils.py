import os
import struct
import cv2
import numpy as np
import torch
import random
from torchvision import transforms
from library.config import Config


# ==========================================
# REPRODUCIBILITY
# ==========================================
def seed_everything(seed=42):
    """
    Sets the random seed for all relevant libraries to ensure determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# BSON PARSING HELPERS
# ==========================================
def _get_val_size(type_byte, data, ptr):
    """
    Helper function to determine the size of a BSON value based on its type.
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
    Parses a raw BSON document byte string to find the 'imgs' array
    and extract 'picture' binary data.

    Args:
        data (bytes): The raw BSON document.

    Returns:
        list[bytes]: A list of binary image strings found in the record.
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
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            # Skip other fields in image doc
                            v_len = _get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = _get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = _get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


# ==========================================
# DATA LOADING & PROCESSING
# ==========================================
def read_bson_record(source, offset, length):
    """
    Reads a specific BSON record from the source file.

    Args:
        source (str or file-like object): Path to the .bson file or an open file object.
        offset (int): Byte offset where the record starts.
        length (int): Length of the record in bytes.

    Returns:
        bytes: The raw BSON document.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            f.seek(offset)
            return f.read(length)
    else:
        # Assume source is an open file object
        source.seek(offset)
        return source.read(length)


# Standard ResNet-50 Preprocessing
# Resize to 224x224 (from ~180x180) and normalize with ImageNet stats
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(img_bytes, transform=None):
    """
    Decodes a raw image byte string and applies preprocessing transformations.

    Args:
        img_bytes (bytes): Binary image data (JPEG).
        transform (callable, optional): PyTorch transform pipeline.
                                        Defaults to standard ResNet-50 validation transform.

    Returns:
        torch.Tensor: The processed image tensor (C, H, W).
    """
    if transform is None:
        transform = DEFAULT_TRANSFORM

    # Decode image from bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodes to BGR

    if img is None:
        # Handle potentially corrupt images by returning a black image
        # ResNet expects 3 channels
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        # Convert BGR (OpenCV default) to RGB (PIL/PyTorch default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transforms
    # ToPILImage handles the conversion from numpy array to PIL Image
    return transform(img)
