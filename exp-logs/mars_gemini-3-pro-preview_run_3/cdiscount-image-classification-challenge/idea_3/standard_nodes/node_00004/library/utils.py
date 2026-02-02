import struct
import os
import cv2
import numpy as np
import pandas as pd
from library.config import SUBMISSION_FILE_PATH


def get_val_size(type_byte, data, ptr):
    """
    Helper function to determine the size of a BSON value based on its type byte.
    Used to skip fields efficiently during parsing.
    """
    if type_byte == 0x01:  # double
        return 8
    elif type_byte == 0x02:  # string
        if ptr + 4 > len(data):
            return 0
        s_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + s_len
    elif type_byte == 0x03:  # document
        if ptr + 4 > len(data):
            return 0
        d_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return d_len
    elif type_byte == 0x04:  # array
        if ptr + 4 > len(data):
            return 0
        a_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return a_len
    elif type_byte == 0x05:  # binary
        if ptr + 4 > len(data):
            return 0
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


def read_bson_images_at_offset(bson_source, offset, length):
    """
    Reads a specific BSON record from a file (or path) at a given offset and
    extracts all images associated with the product.

    Args:
        bson_source (str or file-like): Path to the .bson file or an open file handle (rb).
        offset (int): Byte offset where the record starts.
        length (int): Total length of the record in bytes.

    Returns:
        list[np.ndarray]: A list of decoded images (BGR format).
    """
    close_file = False
    f = None

    try:
        # Handle both file path and open file object
        if isinstance(bson_source, (str, os.PathLike)):
            f = open(bson_source, "rb")
            close_file = True
        else:
            f = bson_source

        f.seek(offset)
        data = f.read(length)

        if len(data) < length:
            # Short read, return empty
            return []

    finally:
        if close_file and f is not None:
            f.close()

    images = []
    ptr = 4  # Skip total size (int32) at the beginning of BSON document
    data_len = len(data)

    while ptr < data_len - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name (cstring)
        name_end = data.find(b"\x00", ptr)
        if name_end == -1:
            break

        name = data[ptr:name_end].decode("utf-8", errors="ignore")
        ptr = name_end + 1

        if name == "imgs" and type_byte == 0x04:
            # Found 'imgs' array
            if ptr + 4 > data_len:
                break
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1 and ap < data_len:
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                if ename_end == -1:
                    break
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    if ap + 4 > data_len:
                        break
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1 and dp < data_len:
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        if dname_end == -1:
                            break
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            if dp + 4 > data_len:
                                break
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_start = dp + 5
                            img_end = img_start + bin_len

                            if img_end <= data_len:
                                img_bytes = data[img_start:img_end]
                                # Decode image
                                nparr = np.frombuffer(img_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    images.append(img)

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


def calculate_accuracy(predictions, labels):
    """
    Calculates categorization accuracy.

    Args:
        predictions (np.ndarray or list): Predicted category IDs.
        labels (np.ndarray or list): Ground truth category IDs.

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)

    if len(preds_arr) != len(labels_arr):
        raise ValueError(
            f"Shape mismatch: predictions {preds_arr.shape} vs labels {labels_arr.shape}"
        )

    correct = (preds_arr == labels_arr).sum()
    total = len(labels_arr)

    if total == 0:
        return 0.0

    return float(correct) / float(total)


def save_submission(ids, predictions, filename=SUBMISSION_FILE_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.ndarray): Product IDs.
        predictions (list or np.ndarray): Predicted category IDs.
        filename (str): Output file path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame({"_id": ids, "category_id": predictions})

    # Ensure integer types
    df["_id"] = df["_id"].astype(int)
    df["category_id"] = df["category_id"].astype(int)

    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
