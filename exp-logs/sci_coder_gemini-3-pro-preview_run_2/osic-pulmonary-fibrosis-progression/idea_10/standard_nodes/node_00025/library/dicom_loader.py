import os
import struct
import numpy as np
from library.config import Config


def read_dicom_slice(path):
    """
    Reads a single DICOM file and returns the image converted to Hounsfield Units.
    Implements a fallback binary parser since pydicom is not available.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except IOError:
        return np.zeros((512, 512), dtype=np.float32)

    # Helper to parse specific tags
    def parse_tag(tag_bytes, vr_type="US"):
        # Find the tag in the binary data
        offset = data.find(tag_bytes)
        if offset == -1:
            return None

        # Move past the 4-byte tag
        current = offset + 4

        # Check for Explicit VR (Value Representation)
        # Explicit VR has a 2-byte VR code (e.g., 'US', 'DS', 'OW')
        vr = data[current : current + 2]
        is_explicit = False

        # Common VR codes
        if vr in [
            b"US",
            b"DS",
            b"OW",
            b"OB",
            b"UI",
            b"CS",
            b"LO",
            b"SH",
            b"IS",
            b"FL",
            b"FD",
            b"SL",
            b"SS",
            b"UL",
        ]:
            is_explicit = True
            current += 2

        length = 0
        if is_explicit:
            # Long VRs have 2 reserved bytes then 4 bytes length
            if vr in [b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"]:
                current += 2
                if current + 4 > len(data):
                    return None
                length = struct.unpack("<I", data[current : current + 4])[0]
                current += 4
            else:
                # Short VRs have 2 bytes length
                if current + 2 > len(data):
                    return None
                length = struct.unpack("<H", data[current : current + 2])[0]
                current += 2
        else:
            # Implicit VR always has 4 bytes length
            if current + 4 > len(data):
                return None
            length = struct.unpack("<I", data[current : current + 4])[0]
            current += 4

        if current + length > len(data):
            return None

        value_bytes = data[current : current + length]

        if vr_type == "US":  # Unsigned Short
            if len(value_bytes) >= 2:
                return struct.unpack("<H", value_bytes[:2])[0]
        elif vr_type == "DS":  # Decimal String
            try:
                val_str = (
                    value_bytes.decode("ascii", errors="ignore").strip().strip("\x00")
                )
                return float(val_str)
            except ValueError:
                return None

        return None

    # DICOM Tags (Group, Element) in Little Endian
    # Rows: (0028, 0010) -> b'\x28\x00\x10\x00'
    # Columns: (0028, 0011) -> b'\x28\x00\x11\x00'
    # Rescale Intercept: (0028, 1052) -> b'\x28\x00\x52\x10'
    # Rescale Slope: (0028, 1053) -> b'\x28\x00\x53\x10'

    rows = parse_tag(b"\x28\x00\x10\x00", "US")
    cols = parse_tag(b"\x28\x00\x11\x00", "US")
    intercept = parse_tag(b"\x28\x00\x52\x10", "DS")
    slope = parse_tag(b"\x28\x00\x53\x10", "DS")

    # Defaults if metadata missing
    if rows is None:
        rows = 512
    if cols is None:
        cols = 512
    if intercept is None:
        intercept = -1024.0
    if slope is None:
        slope = 1.0

    # Extract Pixel Data: Tag (7FE0, 0010) -> b'\xe0\x7f\x10\x00'
    pixel_tag = b"\xe0\x7f\x10\x00"
    offset = data.find(pixel_tag)

    pixel_bytes = None

    if offset != -1:
        current = offset + 4
        # Check VR for Pixel Data
        vr = data[current : current + 2]
        is_explicit_pixel = vr in [b"OW", b"OB"]

        length = 0
        if is_explicit_pixel:
            current += 2  # VR
            current += 2  # Reserved
            if current + 4 <= len(data):
                length = struct.unpack("<I", data[current : current + 4])[0]
                current += 4
        else:
            if current + 4 <= len(data):
                length = struct.unpack("<I", data[current : current + 4])[0]
                current += 4

        # If length is valid and contained in file
        if length > 0 and current + length <= len(data):
            pixel_bytes = data[current : current + length]
        else:
            # Fallback: take the end of the file based on expected size
            expected_size = rows * cols * 2  # Assuming 16-bit
            if len(data) >= expected_size:
                pixel_bytes = data[-expected_size:]
    else:
        # Tag not found, fallback to end of file
        expected_size = rows * cols * 2
        if len(data) >= expected_size:
            pixel_bytes = data[-expected_size:]

    if pixel_bytes is None:
        # Return empty image if parsing fails completely
        return np.zeros((rows, cols), dtype=np.float32)

    # Convert bytes to numpy array
    # CT data is usually stored as int16 or uint16
    try:
        img = np.frombuffer(pixel_bytes, dtype=np.int16).astype(np.float32)
    except Exception:
        return np.zeros((rows, cols), dtype=np.float32)

    # Handle size mismatches
    if img.size != rows * cols:
        # Try uint16
        try:
            img = np.frombuffer(pixel_bytes, dtype=np.uint16).astype(np.float32)
        except Exception:
            pass

        if img.size != rows * cols:
            if img.size > rows * cols:
                img = img[: rows * cols]
            else:
                # Pad with minimum value (air)
                img = np.pad(
                    img, (0, rows * cols - img.size), "constant", constant_values=-2000
                )

    img = img.reshape((rows, cols))

    # Apply Rescale Slope and Intercept to get HU
    img = img * slope + intercept

    return img


def load_scan(scan_dir, load_cached_data=True):
    """
    Loads all DICOM slices from a directory, sorts them by instance number,
    and returns a 3D numpy volume (Depth, Height, Width).

    Implements caching to ./working/idea_10/{patient_id}.npy
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Derive patient ID from directory path
    # scan_dir usually ends with the patient ID
    patient_id = os.path.basename(scan_dir.rstrip(os.sep))
    if not patient_id:
        # Fallback if path is weird
        patient_id = "unknown_patient"

    cache_path = os.path.join(Config.WORKING_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            return volume
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Process from scratch
    if not os.path.exists(scan_dir):
        return np.zeros((0, 512, 512), dtype=np.float32)

    # List .dcm files
    files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".dcm")]

    # Sort files by the integer value of the filename (e.g., '10.dcm' -> 10)
    # This assumes filenames correspond to Z-position/Instance Number
    def try_int(f):
        try:
            return int(os.path.splitext(f)[0])
        except ValueError:
            return 0

    files.sort(key=try_int)

    images = []
    for f in files:
        path = os.path.join(scan_dir, f)
        img = read_dicom_slice(path)
        images.append(img)

    if not images:
        volume = np.zeros((0, 512, 512), dtype=np.float32)
    else:
        # Ensure all images have the same shape before stacking
        # Use the shape of the middle slice as reference
        ref_shape = images[len(images) // 2].shape

        processed_images = []
        for img in images:
            if img.shape != ref_shape:
                # Simple crop or pad could be done here, but for now we skip mismatched
                # or resize? Given constraints, we skip to avoid crashing stack
                continue
            processed_images.append(img)

        if processed_images:
            volume = np.stack(processed_images)
        else:
            volume = np.zeros((0, ref_shape[0], ref_shape[1]), dtype=np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception:
        pass  # Non-critical failure

    return volume
