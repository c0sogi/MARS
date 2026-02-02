import os
import sys
import random
import ast
import numpy as np
import torch
import cv2
import pydicom
from concurrent.futures import ThreadPoolExecutor
from library import config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file and converts it to a standard uint8 numpy array.

    Args:
        path (str): Path to the .dcm file.
        fix_monochrome (bool): If True, inverts MONOCHROME1 images so air is black.

    Returns:
        np.array: The image data normalized to [0, 255] as uint8.
    """
    try:
        dcm = pydicom.dcmread(path)
        data = dcm.pixel_array.astype(np.float32)

        # Photometric Interpretation handling
        if fix_monochrome and hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data

        # Min-Max Normalization to [0, 255]
        data = data - np.min(data)
        max_val = np.max(data)
        if max_val > 0:
            data = data / max_val

        data = (data * 255).astype(np.uint8)
        return data
    except Exception as e:
        # Return a blank image in case of corruption, ensuring pipeline doesn't crash
        # Ideally this should be logged
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)


def resize_image(image, size=None):
    """
    Resizes an image to the target square size.
    """
    if size is None:
        size = config.IMG_SIZE

    if image.shape[0] == size and image.shape[1] == size:
        return image

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def box_to_mask(box_str, height, width):
    """
    Converts a bounding box string (from CSV) into a binary segmentation mask.

    Args:
        box_str (str): A string representation of a list of dictionaries,
                       e.g., "[{'x': 10, 'y': 10, 'width': 50, 'height': 50}]".
        height (int): Original height of the image.
        width (int): Original width of the image.

    Returns:
        np.array: A binary mask of shape (height, width) where 1 indicates opacity.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Handle NaN or empty strings
    if box_str != box_str or box_str == "" or box_str == "nan":
        return mask

    try:
        boxes = ast.literal_eval(box_str)
        for box in boxes:
            x = int(float(box["x"]))
            y = int(float(box["y"]))
            w = int(float(box["width"]))
            h = int(float(box["height"]))

            # Clip coordinates to image boundaries
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(width, x + w)
            y_max = min(height, y + h)

            # Draw rectangle
            mask[y_min:y_max, x_min:x_max] = 1
    except Exception:
        # If parsing fails, return empty mask
        pass

    return mask


def mask_to_boxes(mask, threshold=0.5):
    """
    Converts a prediction mask into bounding boxes using contour detection.

    Args:
        mask (np.array): Probability mask or binary mask.
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: A list of boxes in format [xmin, ymin, xmax, ymax, score].
    """
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < config.MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate confidence score as the mean probability within the bounding box
        # We use the original mask for this
        roi = mask[y : y + h, x : x + w]
        if roi.size > 0:
            score = roi.mean()
        else:
            score = 0.0

        boxes.append([x, y, x + w, y + h, score])

    return boxes


def process_and_cache_images(df, cache_key, load_cached_data=True):
    """
    Loads images from DICOMs, resizes them, and caches them to disk as .npy files.
    This speeds up training by avoiding repeated DICOM I/O and resizing.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path' and 'image_id'.
        cache_key (str): Unique identifier for this split (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping image_id to numpy array image.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    images_path = os.path.join(config.WORKING_DIR, f"{cache_key}_images.npy")
    ids_path = os.path.join(config.WORKING_DIR, f"{cache_key}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        try:
            print(f"Loading cached images for {cache_key}...")
            images_arr = np.load(images_path)
            ids_arr = np.load(ids_path)

            # Reconstruct dictionary
            return {img_id: img for img_id, img in zip(ids_arr, images_arr)}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing images for {cache_key}...")

    # Helper for parallel processing
    def _process_row(row):
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])
        img = read_dicom(full_path)
        img = resize_image(img, config.IMG_SIZE)
        return row["image_id"], img

    # Use ThreadPoolExecutor for I/O bound task
    ids_list = []
    images_list = []

    # Convert df to list of dicts for iteration
    rows = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        results = list(executor.map(_process_row, rows))

    for img_id, img in results:
        ids_list.append(img_id)
        images_list.append(img)

    # Convert to numpy arrays
    images_arr = np.array(images_list, dtype=np.uint8)
    ids_arr = np.array(ids_list)

    # 3. Save to cache
    np.save(images_path, images_arr)
    np.save(ids_path, ids_arr)

    print(f"Cached {len(images_arr)} images for {cache_key}.")

    return {img_id: img for img_id, img in zip(ids_arr, images_arr)}


def format_prediction_string(boxes):
    """
    Formats a list of boxes into the submission string format.

    Args:
        boxes (list): List of [xmin, ymin, xmax, ymax, score].

    Returns:
        str: Prediction string, e.g., "opacity 0.5 100 100 200 200 ..." or "none 1 0 0 1 1".
    """
    if not boxes:
        return "none 1 0 0 1 1"

    pred_strings = []
    for box in boxes:
        # box format: [xmin, ymin, xmax, ymax, score]
        xmin, ymin, xmax, ymax, score = box
        pred_strings.append(f"opacity {score:.4f} {xmin} {ymin} {xmax} {ymax}")

    return " ".join(pred_strings)
