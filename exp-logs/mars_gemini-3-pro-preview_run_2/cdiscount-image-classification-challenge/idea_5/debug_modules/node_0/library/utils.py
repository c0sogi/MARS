import struct
import io
import cv2
import numpy as np
import torch
import os

# ==== BSON Type Constants ====
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _read_cstring(buffer, offset):
    """Reads a null-terminated string from the buffer."""
    end = offset
    while end < len(buffer) and buffer[end] != 0:
        end += 1
    # Decode and return string + new offset (skipping null byte)
    return buffer[offset:end].decode("utf-8", errors="ignore"), end + 1


def _skip_value(buffer, offset, dtype):
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
        return offset + 4 + 1 + l  # length + subtype + data
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
        # Fallback: if we encounter an unknown type, we can't reliably skip.
        # In a strict parser we'd raise error, but here we just return offset
        # which will likely cause downstream failure, but avoids immediate crash.
        return offset


def read_bson_images(source, offset, length):
    """
    Reads a BSON record from a file (or path) and extracts images.

    Args:
        source: File path (str) or open file object (binary mode).
        offset: Byte offset where the record starts.
        length: Length of the record in bytes.

    Returns:
        List of numpy arrays (RGB images).
    """
    data = None

    # Handle both file path and file object
    if isinstance(source, str):
        with open(source, "rb") as f:
            f.seek(offset)
            data = f.read(length)
    else:
        # Assume source is a file-like object
        source.seek(offset)
        data = source.read(length)

    if not data or len(data) != length:
        # Return empty list if read failed or incomplete
        return []

    images = []

    # BSON parsing logic tailored to extract 'imgs' -> 'picture'
    # Skip total size header (4 bytes)
    curr_off = 4
    data_len = len(data)

    while curr_off < data_len - 1:  # Last byte is null terminator
        dtype = data[curr_off]
        curr_off += 1
        key, curr_off = _read_cstring(data, curr_off)

        if key == "imgs" and dtype == TYPE_ARRAY:
            # Found the images array
            arr_size = struct.unpack_from("<i", data, curr_off)[0]
            arr_end = curr_off + arr_size
            curr_off += 4  # Skip size

            # Iterate array elements
            while curr_off < arr_end - 1:
                e_type = data[curr_off]
                curr_off += 1
                e_key, curr_off = _read_cstring(data, curr_off)

                if e_type == TYPE_DOC:
                    doc_size = struct.unpack_from("<i", data, curr_off)[0]
                    doc_end = curr_off + doc_size

                    # Search inside the document for 'picture'
                    sub_off = curr_off + 4
                    while sub_off < doc_end - 1:
                        s_type = data[sub_off]
                        sub_off += 1
                        s_key, sub_off = _read_cstring(data, sub_off)

                        if s_key == "picture" and s_type == TYPE_BINARY:
                            b_len = struct.unpack_from("<i", data, sub_off)[0]
                            sub_off += 4
                            subtype = data[sub_off]
                            sub_off += 1

                            # Extract binary data
                            img_bytes = data[sub_off : sub_off + b_len]

                            # Decode image
                            # Use numpy frombuffer -> cv2.imdecode
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

                            if img is not None:
                                # Convert BGR to RGB
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                images.append(img)

                            sub_off += b_len
                        else:
                            sub_off = _skip_value(data, sub_off, s_type)

                    curr_off = doc_end
                else:
                    curr_off = _skip_value(data, curr_off, e_type)
        else:
            curr_off = _skip_value(data, curr_off, dtype)

    return images
