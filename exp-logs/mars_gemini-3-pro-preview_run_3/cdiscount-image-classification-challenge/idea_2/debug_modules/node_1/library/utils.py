import struct
import cv2
import numpy as np
import torchvision.transforms as T
from library.config import IMG_SIZE


def get_val_size(type_byte, data, ptr):
    """
    Helper function to determine the size of a BSON value based on its type byte.
    Used to skip over fields that are not relevant to image extraction.
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


def extract_images_from_bson(data):
    """
    Parses a raw BSON document byte string to find the 'imgs' array,
    extracts the 'picture' binary data, and decodes them into RGB numpy arrays.

    Args:
        data (bytes): The raw bytes of a single BSON document.

    Returns:
        list[np.ndarray]: A list of decoded images in RGB format.
    """
    images = []
    ptr = 4  # Skip total size header (int32)
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
            if ptr + 4 > length:
                break
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                if ename_end == -1:
                    break
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    if ap + 4 > length:
                        break
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        if dname_end == -1:
                            break
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            if dp + 4 > length:
                                break
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]

                            # Decode Image
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            # IMREAD_COLOR ensures 3 channels (BGR) even if grayscale source
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if img is not None:
                                # Convert BGR to RGB
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def get_transforms(img_size=IMG_SIZE):
    """
    Returns the torchvision transformations pipeline.
    Since we are using a frozen backbone, we use deterministic resizing and normalization
    consistent with ImageNet pre-training.

    Args:
        img_size (int): The target height and width for resizing.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
