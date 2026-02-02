import os
import re
import numpy as np
import cv2


def read_dicom_robust(file_path):
    """
    Reads a DICOM file with a robust fallback mechanism.

    1. Attempts to read using OpenCV.
    2. If OpenCV fails, attempts to read raw binary data based on file size
       (supporting 512x512 and 256x256 resolutions).

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image array (uint16), or a zero array if loading fails.
    """
    if not os.path.exists(file_path):
        return np.zeros((512, 512), dtype=np.uint16)

    # Attempt 1: OpenCV
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # Attempt 2: Raw Binary Fallback
    # DICOM files usually have a header followed by pixel data.
    # We infer resolution from file size.
    try:
        file_size = os.path.getsize(file_path)

        # 512x512 pixels * 2 bytes/pixel = 524,288 bytes
        # Typical file size ~525kB
        if file_size >= 524288:
            shape = (512, 512)
            expected_bytes = 512 * 512 * 2
        # 256x256 pixels * 2 bytes/pixel = 131,072 bytes
        # Typical file size ~132kB
        elif file_size >= 131072:
            shape = (256, 256)
            expected_bytes = 256 * 256 * 2
        else:
            # Unknown dimension, return safe default
            return np.zeros((512, 512), dtype=np.uint16)

        with open(file_path, "rb") as f:
            data = f.read()
            # Pixel data is typically at the end of the file
            pixel_data = data[-expected_bytes:]
            img_array = np.frombuffer(pixel_data, dtype=np.uint16)
            img_array = img_array.reshape(shape)
            return img_array

    except Exception:
        pass

    # Final Fallback
    return np.zeros((512, 512), dtype=np.uint16)


def get_flair_anchor(flair_path):
    """
    Scans the FLAIR directory to identify the slice index with the maximum signal intensity.
    This slice serves as the 'Anchor' for volumetric slab selection.

    Args:
        flair_path (str): Path to the FLAIR modality directory.

    Returns:
        int: The image number (index) of the slice with the highest mean intensity.
    """
    if not os.path.exists(flair_path):
        return 0

    files = os.listdir(flair_path)
    # Filter for .dcm files
    files = [f for f in files if f.endswith(".dcm")]

    if not files:
        return 0

    # Helper to extract image number: Image-123.dcm -> 123
    def extract_number(filename):
        match = re.search(r"Image-(\d+)", filename)
        return int(match.group(1)) if match else 0

    # Create a list of (image_number, filename), sorted by image number
    sorted_files = sorted([(extract_number(f), f) for f in files], key=lambda x: x[0])

    max_intensity = -1.0
    anchor_index = 0

    # If no files could be parsed correctly, we might default to the middle
    if not sorted_files:
        return 0

    # Cite solution_lesson_node_00018: Exclude top/bottom % to avoid artifacts
    num_files = len(sorted_files)
    start_idx = int(num_files * Config.ROI_EXCLUDE_BUFFER)
    end_idx = int(num_files * (1 - Config.ROI_EXCLUDE_BUFFER))

    # Ensure we have a valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_files

    search_files = sorted_files[start_idx:end_idx]

    # Iterate through slices to calculate intensity profile
    for img_num, filename in search_files:
        full_path = os.path.join(flair_path, filename)
        img = read_dicom_robust(full_path)

        # We use mean intensity as a robust metric for "amount of brain/tumor signal"
        # This avoids single-pixel noise spikes that might affect np.max()
        current_intensity = np.mean(img)

        if current_intensity > max_intensity:
            max_intensity = current_intensity
            anchor_index = img_num

    # Fallback: if all images were black or empty, pick the middle slice of the full volume
    if max_intensity <= 0 and sorted_files:
        mid_idx = len(sorted_files) // 2
        anchor_index = sorted_files[mid_idx][0]

    return anchor_index


def compute_mip(slices):
    """
    Computes the Maximum Intensity Projection (MIP) for a stack of slices.

    Args:
        slices (list of np.ndarray): A list of 2D image arrays (slabs).

    Returns:
        np.ndarray: A single 2D image array representing the MIP.
    """
    if not slices:
        return np.zeros((512, 512), dtype=np.uint16)

    # Stack slices into a 3D volume (Depth, Height, Width)
    try:
        stack = np.stack(slices, axis=0)
        # Compute maximum intensity along the depth axis
        mip = np.max(stack, axis=0)
        return mip
    except ValueError:
        # Handle case where slices might have different shapes (unlikely but possible)
        # Return the middle slice as fallback
        mid = len(slices) // 2
        return slices[mid]
