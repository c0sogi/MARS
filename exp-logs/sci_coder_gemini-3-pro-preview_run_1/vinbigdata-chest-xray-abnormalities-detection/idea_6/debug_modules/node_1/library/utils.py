import os
import struct
import numpy as np
from library.config import Config


class SimpleDicomParser:
    """
    A minimal, heuristic-based DICOM parser to handle binary reading
    without external dependencies. Assumes Little Endian byte order.
    """

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.content = f.read()

        # Skip 128-byte preamble + 4-byte 'DICM' prefix if present
        if len(self.content) > 132 and self.content[128:132] == b"DICM":
            self.start_idx = 132
        else:
            self.start_idx = 0

    def find_tag(self, group, element):
        """Finds the offset of a specific tag (Group, Element)."""
        tag_bytes = struct.pack("<HH", group, element)
        return self.content.find(tag_bytes, self.start_idx)

    def get_value(self, group, element, vr_type="US"):
        """Extracts value for a given tag based on expected Value Representation (VR)."""
        idx = self.find_tag(group, element)
        if idx == -1:
            return None

        # Move past tag (4 bytes)
        current = idx + 4

        # Check for Explicit VR (2 uppercase chars)
        if current + 2 > len(self.content):
            return None
        possible_vr = self.content[current : current + 2]
        is_explicit = possible_vr.isalpha() and possible_vr.isupper()

        length = 0

        if is_explicit:
            vr = possible_vr.decode("ascii", errors="ignore")
            current += 2  # Skip VR

            if vr in ["OB", "OW", "OF", "SQ", "UT", "UN"]:
                # Reserved 2 bytes, then 4 byte length
                current += 2
                length = struct.unpack("<I", self.content[current : current + 4])[0]
                current += 4
            else:
                # 2 byte length
                length = struct.unpack("<H", self.content[current : current + 2])[0]
                current += 2
        else:
            # Implicit VR: 4 byte length
            length = struct.unpack("<I", self.content[current : current + 4])[0]
            current += 4

        value_bytes = self.content[current : current + length]

        if len(value_bytes) < length:
            return None

        if vr_type == "US":  # Unsigned Short
            if len(value_bytes) >= 2:
                return struct.unpack("<H", value_bytes[:2])[0]
            return 0
        elif vr_type == "CS":  # Code String
            return value_bytes.decode("ascii", errors="ignore").strip().strip("\x00")

        return value_bytes

    def get_pixel_data(self, bits_allocated=16):
        """Extracts the raw pixel data array."""
        # Tag (7FE0, 0010) - Pixel Data
        idx = self.find_tag(0x7FE0, 0x0010)
        if idx == -1:
            # Fallback: sometimes tags are different in very old files, but standard is 7FE0,0010
            return None

        current = idx + 4
        possible_vr = self.content[current : current + 2]
        is_explicit = possible_vr.isalpha() and possible_vr.isupper()

        length = 0

        if is_explicit:
            vr = possible_vr.decode("ascii", errors="ignore")
            current += 2
            if vr in ["OB", "OW", "OF", "SQ", "UT", "UN"]:
                current += 2
                length = struct.unpack("<I", self.content[current : current + 4])[0]
                current += 4
            else:
                length = struct.unpack("<H", self.content[current : current + 2])[0]
                current += 2
        else:
            length = struct.unpack("<I", self.content[current : current + 4])[0]
            current += 4

        # Handle undefined length (Encapsulated/Compressed)
        if length == 0xFFFFFFFF:
            # For this task, we assume data is not encapsulated (JPEG) as we lack libraries.
            # We take the rest of the file as data.
            data = self.content[current:]
        else:
            data = self.content[current : current + length]

        dtype = np.uint16 if bits_allocated > 8 else np.uint8

        # Ensure data length is a multiple of dtype size
        bytes_per_pixel = 2 if dtype == np.uint16 else 1
        num_pixels = len(data) // bytes_per_pixel
        valid_len = num_pixels * bytes_per_pixel

        return np.frombuffer(data[:valid_len], dtype=dtype)


def read_dicom_semantically_normalized(path):
    """
    Parses a DICOM file, extracting the image and normalizing it semantically.
    Inverts MONOCHROME1 images so that 0 is consistently 'Black'.

    Returns:
        img (np.ndarray): Float32 normalized image (0-1).
        rows (int): Original height.
        cols (int): Original width.
    """
    parser = SimpleDicomParser(path)

    # Extract Metadata
    rows = parser.get_value(0x0028, 0x0010, "US")  # Rows
    cols = parser.get_value(0x0028, 0x0011, "US")  # Columns
    bits_allocated = parser.get_value(0x0028, 0x0100, "US") or 16
    photometric = parser.get_value(0x0028, 0x0004, "CS")  # Photometric Interpretation

    # Extract Pixel Data
    pixel_data = parser.get_pixel_data(bits_allocated=bits_allocated)

    if pixel_data is None:
        # Critical failure fallback: return empty square
        return np.zeros((1024, 1024), dtype=np.float32), 1024, 1024

    # Reshape
    if rows and cols and len(pixel_data) >= rows * cols:
        img = pixel_data[: rows * cols].reshape((rows, cols))
    else:
        # Fallback: Infer square dimensions
        side = int(np.sqrt(len(pixel_data)))
        img = pixel_data[: side * side].reshape((side, side))
        rows, cols = side, side

    # Semantic Normalization
    # If Photometric Interpretation is MONOCHROME1, 0 is White.
    # We want 0 to be Black (Air) to match MONOCHROME2.
    if photometric == "MONOCHROME1":
        img = np.max(img) - img

    # Min-Max Normalization to 0-1 range
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img, rows, cols


def get_image_and_dimensions(image_id, path, load_cached_data=True):
    """
    Coupled loader that returns the image and its original dimensions.
    Implements caching to speed up training.

    Args:
        image_id (str): Unique image identifier.
        path (str): Relative path to the DICOM file (e.g., 'train/xxx.dicom').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        img (np.ndarray): Image array.
        h (int): Original height.
        w (int): Original width.
    """
    # Determine cache location
    # path is like "train/xxxx.dicom". We map this to "working/idea_6/cache/train/xxxx.npy"
    subfolder = os.path.dirname(path)
    filename = f"{image_id}.npy"
    cache_dir = os.path.join(Config.CACHE_DIR, subfolder)
    cache_path = os.path.join(cache_dir, filename)

    # Ensure subdirectory exists (redundant if Config.setup called, but safe)
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            img = np.load(cache_path)
            h, w = img.shape[:2]
            return img, h, w
        except Exception:
            # If load fails (corrupt), proceed to process
            pass

    # 2. Process from scratch
    # Construct full input path
    full_input_path = os.path.join(Config.INPUT_DIR, path)

    # Use custom parser
    try:
        img, h, w = read_dicom_semantically_normalized(full_input_path)
    except Exception as e:
        # Fallback for completely broken files
        print(f"Error reading {full_input_path}: {e}")
        img = np.zeros((1024, 1024), dtype=np.float32)
        h, w = 1024, 1024

    # 3. Save to cache
    try:
        np.save(cache_path, img)
    except Exception:
        pass  # Ignore save errors (disk full, etc.)

    return img, h, w
