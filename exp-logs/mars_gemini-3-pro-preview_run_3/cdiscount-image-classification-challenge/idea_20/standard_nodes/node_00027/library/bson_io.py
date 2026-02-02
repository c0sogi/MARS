import os
import struct
import numpy as np
import cv2
from library.config import Config


class BSONImageReader:
    """
    Handles low-level I/O operations for reading images from BSON files.
    Uses random access via byte offsets to efficiently retrieve product records.
    """

    def __init__(self, bson_file_path):
        """
        Initialize the reader with the path to the BSON file.

        Args:
            bson_file_path (str): Path to the .bson file (train or test).
        """
        self.bson_file_path = bson_file_path
        self.file_handle = None

    def _open_file(self):
        """Opens the file handle if it is not already open."""
        if self.file_handle is None:
            self.file_handle = open(self.bson_file_path, "rb")

    def close(self):
        """Closes the file handle if it is open."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        """Destructor to ensure file handle is closed."""
        self.close()

    def __getstate__(self):
        """
        Custom pickling behavior for multiprocessing safety.
        File handles cannot be pickled, so we set it to None in the state.
        The handle will be re-opened lazily in the worker process.
        """
        state = self.__dict__.copy()
        state["file_handle"] = None
        return state

    def read_product(self, offset, length):
        """
        Reads a product record from the BSON file at the specified offset and extracts images.

        Args:
            offset (int): The byte offset where the record starts.
            length (int): The total length of the BSON record in bytes.

        Returns:
            list[np.ndarray]: A list of decoded images in RGB format (H, W, 3).
                              Returns an empty list if no valid images are found.
        """
        self._open_file()

        try:
            self.file_handle.seek(offset)
            data = self.file_handle.read(length)
        except (OSError, ValueError):
            # Handle potential file handle issues (e.g., stale handle in forked process)
            self.close()
            self._open_file()
            self.file_handle.seek(offset)
            data = self.file_handle.read(length)

        # Parse BSON to get raw image bytes
        img_binaries = self._extract_images_from_bson(data)

        images = []
        for img_bytes in img_binaries:
            # Decode image from memory buffer
            nparr = np.frombuffer(img_bytes, np.uint8)
            # Load as color (BGR)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Convert BGR (OpenCV default) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

        return images

    def _get_val_size(self, type_byte, data, ptr):
        """
        Helper to determine the size in bytes of a BSON value based on its type.
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

    def _extract_images_from_bson(self, data):
        """
        Parses the raw BSON bytes of a product record to find the 'imgs' array
        and extract the binary data for each 'picture'.
        """
        images = []
        ptr = 4  # Skip total size header (int32)
        length = len(data)

        while ptr < length - 1:
            type_byte = data[ptr]
            ptr += 1

            # Read Field Name (null-terminated string)
            name_end = data.find(b"\x00", ptr)
            if name_end == -1:
                break

            # Decode name (needed to identify 'imgs')
            name = data[ptr:name_end].decode("utf-8", errors="ignore")
            ptr = name_end + 1

            # Check if this is the 'imgs' array
            if name == "imgs" and type_byte == 0x04:
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
                                images.append(img_bytes)
                                dp += 4 + 1 + bin_len
                            else:
                                # Skip other fields in image doc
                                v_len = self._get_val_size(dtype, data, dp)
                                dp += v_len

                        ap += doc_len
                    else:
                        v_len = self._get_val_size(etype, data, ap)
                        ap += v_len

                ptr += arr_len
            else:
                # Skip this field
                v_len = self._get_val_size(type_byte, data, ptr)
                ptr += v_len

        return images
