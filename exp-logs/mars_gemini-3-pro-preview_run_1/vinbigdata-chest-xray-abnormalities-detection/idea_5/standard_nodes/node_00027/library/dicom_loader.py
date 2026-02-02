import os
import struct
import numpy as np
import cv2
import logging
import rasterio
from library.config import Config

# Setup logger
logger = logging.getLogger("DicomLoader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DicomParser:
    """
    A robust, dependency-free DICOM parser designed to extract pixel data
    and critical metadata (Rows, Columns, Photometric Interpretation)
    by reading binary streams directly.
    """

    def __init__(self, path):
        self.path = path
        self.tags = {}
        self.pixel_data = None
        self.is_explicit_vr = True
        self.is_little_endian = True
        self.meta = {
            "Rows": None,
            "Columns": None,
            "PhotometricInterpretation": "MONOCHROME2",
            "BitsStored": 8,
            "SamplesPerPixel": 1,
        }

        try:
            with open(path, "rb") as f:
                self.data = f.read()
            self._parse()
        except Exception as e:
            logger.error(f"Failed to parse DICOM {path}: {e}")
            self.pixel_data = None

    def _parse(self):
        # Check DICM prefix
        if self.data[128:132] != b"DICM":
            # Not a standard DICOM file, might be raw or no header
            # We will try to rely on CV2 fallback in the loader if this happens
            return

        # Start parsing after header
        offset = 132

        # Heuristic to detect Transfer Syntax (Explicit vs Implicit)
        # We look at the first tag after the File Meta Information (Group 0002)
        # Usually, we just parse sequentially. Group 0002 is always Explicit LE.
        # After Group 0002, the transfer syntax applies.

        # Simple parser loop
        while offset < len(self.data):
            # Read Tag (Group, Element)
            if offset + 4 > len(self.data):
                break
            group = struct.unpack("<H", self.data[offset : offset + 2])[0]
            element = struct.unpack("<H", self.data[offset + 2 : offset + 4])[0]
            offset += 4

            # Handle Pixel Data (7FE0, 0010)
            if group == 0x7FE0 and element == 0x0010:
                self._handle_pixel_data(offset)
                break  # Stop parsing after finding pixel data

            # Determine VR and Length
            vr = ""
            length = 0

            # Logic for Explicit vs Implicit is complex to implement fully robustly
            # without a library. We use a simplified heuristic:
            # If the next 2 bytes are uppercase ASCII chars, it's likely Explicit VR.
            is_explicit = False
            if offset + 2 <= len(self.data):
                potential_vr = self.data[offset : offset + 2]
                if potential_vr.isalpha() and potential_vr.isupper():
                    is_explicit = True

            if is_explicit:
                vr = self.data[offset : offset + 2].decode("ascii", errors="ignore")
                offset += 2
                if vr in ["OB", "OW", "OF", "SQ", "UT", "UN"]:
                    offset += 2  # Reserved
                    length = struct.unpack("<I", self.data[offset : offset + 4])[0]
                    offset += 4
                else:
                    length = struct.unpack("<H", self.data[offset : offset + 2])[0]
                    offset += 2
            else:
                # Implicit VR
                length = struct.unpack("<I", self.data[offset : offset + 4])[0]
                offset += 4

            # Extract Value for critical tags
            if length > 0 and offset + length <= len(self.data):
                value_bytes = self.data[offset : offset + length]

                # Rows (0028, 0010)
                if group == 0x0028 and element == 0x0010:
                    self.meta["Rows"] = self._bytes_to_int(value_bytes)

                # Columns (0028, 0011)
                if group == 0x0028 and element == 0x0011:
                    self.meta["Columns"] = self._bytes_to_int(value_bytes)

                # Photometric Interpretation (0028, 0004)
                if group == 0x0028 and element == 0x0004:
                    self.meta["PhotometricInterpretation"] = (
                        value_bytes.decode("ascii").strip().strip("\x00")
                    )

                # Bits Stored (0028, 0101)
                if group == 0x0028 and element == 0x0101:
                    self.meta["BitsStored"] = self._bytes_to_int(value_bytes)

                # Samples Per Pixel (0028, 0002)
                if group == 0x0028 and element == 0x0002:
                    self.meta["SamplesPerPixel"] = self._bytes_to_int(value_bytes)

            offset += length

    def _handle_pixel_data(self, offset):
        # Parse PixelData length
        # In Explicit VR, PixelData (OB/OW) has 2 reserved bytes + 4 bytes length
        # In Implicit VR, it has 4 bytes length.
        # We check the bytes preceding the tag call in the loop, but here we are AT the value start position?
        # No, the loop logic above calls this function passing 'offset' which is AFTER the tag.

        # We need to re-evaluate the length header for PixelData specifically because the loop structure
        # above might have consumed VR bytes if it thought it was explicit.
        # Let's look at the bytes at 'offset' relative to the tag.
        # Actually, simpler approach: The Pixel Data element is usually the last one.
        # We can just take the rest of the file if we can't parse length perfectly.

        # However, let's try to find the length.
        # If Explicit: VR(2) + Reserved(2) + Length(4)
        # If Implicit: Length(4)

        # Check VR
        is_explicit = False
        if self.data[offset : offset + 2].isalpha():
            is_explicit = True

        data_start = offset
        length = 0

        if is_explicit:
            # VR (e.g. OW or OB)
            data_start += 2
            # Reserved
            data_start += 2
            # Length
            length = struct.unpack("<I", self.data[data_start : data_start + 4])[0]
            data_start += 4
        else:
            # Length
            length = struct.unpack("<I", self.data[data_start : data_start + 4])[0]
            data_start += 4

        # If length is undefined (0xFFFFFFFF), it is encapsulated data (compressed)
        if length == 0xFFFFFFFF:
            # Encapsulated data. We will try to use CV2 on the raw bytes of the whole file
            # or extract the fragment. For simplicity/robustness, we'll flag for CV2 fallback.
            self.pixel_data = None
            return

        raw_pixels = self.data[data_start : data_start + length]

        # Convert to numpy
        dtype = np.uint8
        if self.meta["BitsStored"] > 8:
            dtype = np.uint16

        try:
            arr = np.frombuffer(raw_pixels, dtype=dtype)

            # Reshape
            if self.meta["Rows"] and self.meta["Columns"]:
                # Check if size matches
                expected_size = (
                    self.meta["Rows"]
                    * self.meta["Columns"]
                    * self.meta["SamplesPerPixel"]
                )
                if arr.size == expected_size:
                    if self.meta["SamplesPerPixel"] > 1:
                        arr = arr.reshape(
                            (
                                self.meta["Rows"],
                                self.meta["Columns"],
                                self.meta["SamplesPerPixel"],
                            )
                        )
                    else:
                        arr = arr.reshape((self.meta["Rows"], self.meta["Columns"]))
                    self.pixel_data = arr
                else:
                    # Size mismatch, maybe compressed stream that frombuffer read as garbage
                    self.pixel_data = None
            else:
                self.pixel_data = None
        except Exception:
            self.pixel_data = None

    def _bytes_to_int(self, b):
        if len(b) == 2:
            return struct.unpack("<H", b)[0]
        elif len(b) == 4:
            return struct.unpack("<I", b)[0]
        elif len(b) == 1:
            return struct.unpack("<B", b)[0]
        return 0


def read_dicom_image(path: str, fix_monochrome: bool = True):
    """
    Reads a DICOM image from the specified path.

    Args:
        path (str): Path to the .dicom file.
        fix_monochrome (bool): If True, inverts MONOCHROME1 images to be MONOCHROME2.

    Returns:
        tuple: (image_array, original_height, original_width)
            image_array is a numpy array (H, W, 3) normalized to 0-255 uint8.
    """
    img = None

    # Strategy 1: Rasterio (GDAL) - Robust for compressed DICOM (Cite debug_lesson_4)
    try:
        with rasterio.open(path) as src:
            img = src.read()  # Returns (C, H, W)
            if img.shape[0] == 1:
                img = img[0]  # (H, W)
            else:
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    except Exception:
        pass

    # Strategy 2: Manual Parser (Existing)
    if img is None:
        parser = DicomParser(path)
        img = parser.pixel_data

    # Strategy 3: OpenCV Fallback
    if img is None:
        # Try reading the file directly with cv2 (sometimes works if plugins are present)
        # or try decoding the bytes if we found the data but couldn't reshape it
        try:
            # Attempt to read as a standard image (handling encapsulated JPEG inside DICOM)
            # We pass the whole file buffer to imdecode
            with open(path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None

    if img is None:
        # Last resort: Create a black image to prevent pipeline crash, log error
        logger.error(f"Could not decode image: {path}. Returning blank placeholder.")
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

    # Handle Dimensions
    if len(img.shape) == 2:
        h, w = img.shape
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        h, w = img.shape[:2]
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Assume BGR from CV2

    # Photometric Correction (MONOCHROME1 -> MONOCHROME2)
    # Only apply if we parsed metadata and it says MONOCHROME1
    if fix_monochrome and parser.meta["PhotometricInterpretation"] == "MONOCHROME1":
        # Invert
        # Check max value based on dtype
        if img.dtype == np.uint8:
            img = 255 - img
        elif img.dtype == np.uint16:
            # We don't know the exact max (could be 12-bit 4095 or 16-bit 65535)
            # Safe bet for inversion is usually max of type or max of data
            img = 65535 - img

    # Normalize to 8-bit for consistency
    if img.dtype != np.uint8:
        # Min-Max normalize to 0-255
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
        img = img.astype(np.uint8)

    return img, h, w


def get_image_tensor(
    path: str, cache_dir: str = Config.CACHE_DIR, load_cached_data: bool = True
):
    """
    Retrieves the image tensor, using a caching mechanism to speed up access.

    Args:
        path (str): Full path to the DICOM file.
        cache_dir (str): Directory to store/load cached .npy files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (image_array, original_height, original_width)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename based on image ID (filename without extension)
    filename = os.path.basename(path)
    image_id = os.path.splitext(filename)[0]
    cache_path = os.path.join(cache_dir, f"{image_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load the array
            img = np.load(cache_path)
            # Extract dimensions directly from the loaded array
            if len(img.shape) == 2:
                h, w = img.shape
            else:
                h, w = img.shape[:2]
            return img, h, w
        except Exception as e:
            logger.warning(f"Failed to load cache for {image_id}: {e}. Re-processing.")

    # 2. Process from scratch
    img, h, w = read_dicom_image(path)

    # 3. Save to cache
    try:
        np.save(cache_path, img)
    except Exception as e:
        logger.warning(f"Failed to save cache for {image_id}: {e}")

    return img, h, w
