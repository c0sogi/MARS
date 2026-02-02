import os
import struct
import numpy as np
import cv2
import logging
import rasterio
import io
from PIL import Image
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

    def __init__(self, path=None, data=None):
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
            if data is not None:
                self.data = data
            elif path is not None:
                with open(path, "rb") as f:
                    self.data = f.read()
            else:
                return

            self._parse()
        except Exception as e:
            # logger.error(f"Failed to parse DICOM {path}: {e}")
            self.pixel_data = None

    def _parse(self):
        # Check DICM prefix
        if len(self.data) < 132 or self.data[128:132] != b"DICM":
            return

        # Start parsing after header
        offset = 132

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

                if group == 0x0028 and element == 0x0010:  # Rows
                    self.meta["Rows"] = self._bytes_to_int(value_bytes)
                if group == 0x0028 and element == 0x0011:  # Columns
                    self.meta["Columns"] = self._bytes_to_int(value_bytes)
                if group == 0x0028 and element == 0x0004:  # Photometric
                    self.meta["PhotometricInterpretation"] = (
                        value_bytes.decode("ascii").strip().strip("\x00")
                    )
                if group == 0x0028 and element == 0x0101:  # BitsStored
                    self.meta["BitsStored"] = self._bytes_to_int(value_bytes)
                if group == 0x0028 and element == 0x0002:  # SamplesPerPixel
                    self.meta["SamplesPerPixel"] = self._bytes_to_int(value_bytes)

            offset += length

    def _handle_pixel_data(self, offset):
        # Check VR
        is_explicit = False
        if offset + 2 <= len(self.data) and self.data[offset : offset + 2].isalpha():
            is_explicit = True

        data_start = offset
        length = 0

        if is_explicit:
            data_start += 2  # VR
            data_start += 2  # Reserved
            length = struct.unpack("<I", self.data[data_start : data_start + 4])[0]
            data_start += 4
        else:
            length = struct.unpack("<I", self.data[data_start : data_start + 4])[0]
            data_start += 4

        # If length is undefined (0xFFFFFFFF), it is encapsulated data (compressed)
        if length == 0xFFFFFFFF:
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


def extract_embedded_image(data):
    """
    Scans binary data for JPEG/JP2 headers and attempts decoding using OpenCV and PIL.
    This is a robust fallback for encapsulated DICOM data when pydicom is missing.
    """
    candidates = []

    # 1. Search for JPEG Header (FF D8)
    # Limit search to avoid scanning huge arrays if not found early
    start = 0
    for _ in range(5):
        idx = data.find(b"\xff\xd8", start)
        if idx == -1:
            break
        candidates.append(idx)
        start = idx + 2

    # 2. Search for JPEG 2000 Codestream (FF 4F FF 51)
    start = 0
    for _ in range(5):
        idx = data.find(b"\xff\x4f\xff\x51", start)
        if idx == -1:
            break
        candidates.append(idx)
        start = idx + 4

    # 3. Search for JPEG 2000 Signature (00 00 00 0C 6A 50)
    start = 0
    for _ in range(5):
        idx = data.find(b"\x00\x00\x00\x0c\x6a\x50", start)
        if idx == -1:
            break
        candidates.append(idx)
        start = idx + 12

    for offset in candidates:
        stream = data[offset:]

        # Method A: OpenCV (Fast, supports many formats if compiled)
        try:
            # np.frombuffer is efficient
            arr = np.frombuffer(stream, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                # Sanity check dimensions
                if img.shape[0] > 16 and img.shape[1] > 16:
                    return img
        except Exception:
            pass

        # Method B: PIL (Robust to trailing garbage)
        try:
            pil_img = Image.open(io.BytesIO(stream))
            img = np.array(pil_img)

            if img is not None and img.size > 0:
                if img.shape[0] > 16 and img.shape[1] > 16:
                    # PIL is RGB, OpenCV expects BGR usually.
                    # Our pipeline handles BGR->RGB conversion in read_dicom_image.
                    # If we return RGB here, the loader will swap it to BGR (Red->Blue).
                    # So we must convert PIL's RGB to BGR before returning.
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    return img
        except Exception:
            pass

    return None


def read_dicom_image(path: str, fix_monochrome: bool = True):
    """
    Reads a DICOM image from the specified path.
    """
    img = None
    file_bytes = None

    # Strategy 1: Rasterio (GDAL)
    # Sometimes works for compressed DICOM if drivers are present
    try:
        with rasterio.open(path) as src:
            img = src.read()
            if img.shape[0] == 1:
                img = img[0]
            else:
                img = np.transpose(img, (1, 2, 0))
    except Exception:
        pass

    # Load bytes for subsequent strategies if needed
    if img is None:
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return (
                np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8),
                Config.IMG_SIZE,
                Config.IMG_SIZE,
            )

    # Strategy 2: Manual Parser (Uncompressed)
    if img is None:
        parser = DicomParser(data=file_bytes)
        img = parser.pixel_data

    # Strategy 3: Robust Embedded Image Search (Compressed/Encapsulated)
    # Cite debug_lesson_1 (Low-Level Binary Parsing)
    if img is None:
        img = extract_embedded_image(file_bytes)

    # Strategy 4: OpenCV Fallback (Whole file)
    # Cite debug_lesson_4 (Layer Multiple Decoding Backends)
    if img is None:
        try:
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Failure: Return Blank
    if img is None:
        logger.error(f"Could not decode image: {path}. Returning blank placeholder.")
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

    # Handle Dimensions & Channels
    if len(img.shape) == 2:
        h, w = img.shape
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
    # Only if we have metadata from the parser
    if (
        fix_monochrome
        and "parser" in locals()
        and parser.meta["PhotometricInterpretation"] == "MONOCHROME1"
    ):
        if img.dtype == np.uint8:
            img = 255 - img
        elif img.dtype == np.uint16:
            img = 65535 - img

    # Normalize to 8-bit
    if img.dtype != np.uint8:
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
