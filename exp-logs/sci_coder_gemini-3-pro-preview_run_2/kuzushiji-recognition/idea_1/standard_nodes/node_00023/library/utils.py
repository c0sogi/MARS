import os
import cv2
import numpy as np
import torch
from library.config import Config, seed_everything


def load_image(image_path):
    """
    Loads an image from the given path using OpenCV and converts it to RGB.

    Args:
        image_path (str): The path to the image. Can be relative to Config.INPUT_DIR
                          or an absolute path.

    Returns:
        np.ndarray: The loaded image in RGB format.
    """
    # Resolve the full path
    if os.path.isabs(image_path):
        full_path = image_path
    else:
        full_path = os.path.join(Config.INPUT_DIR, image_path)

    # Check existence
    if not os.path.exists(full_path):
        # Attempt to find it in train/test subdirectories if direct path fails
        # This adds robustness if metadata paths are just filenames
        train_path = os.path.join(
            Config.INPUT_DIR, "train_images", os.path.basename(image_path)
        )
        test_path = os.path.join(
            Config.INPUT_DIR, "test_images", os.path.basename(image_path)
        )

        if os.path.exists(train_path):
            full_path = train_path
        elif os.path.exists(test_path):
            full_path = test_path
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = cv2.imread(full_path)
    if img is None:
        raise ValueError(f"Failed to load image at {full_path}. File may be corrupt.")

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def parse_labels(label_str):
    """
    Parses the raw label string from the dataset into a list of bounding boxes.

    Args:
        label_str (str): A space-separated string in the format "Unicode X Y W H ...".

    Returns:
        list: A list of dictionaries, where each dict contains:
              {'char': str, 'x': int, 'y': int, 'w': int, 'h': int}
    """
    if not isinstance(label_str, str) or not label_str:
        return []

    parts = label_str.split()

    # Each character label consists of 5 parts: Unicode, X, Y, W, H
    if len(parts) % 5 != 0:
        # Return empty list or handle gracefully if format is unexpected
        return []

    labels = []
    for i in range(0, len(parts), 5):
        try:
            char = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])
            labels.append({"char": char, "x": x, "y": y, "w": w, "h": h})
        except ValueError:
            # Skip malformed entries
            continue

    return labels


def generate_heatmap_target(target_shape, boxes, original_shape=None, sigma=3.0):
    """
    Generates a Gaussian heatmap from a list of bounding boxes (using centers).
    Cite solution_lesson_node_00001: Using center points instead of binary masks.

    Args:
        target_shape (tuple): The desired output shape (Height, Width).
        boxes (list): List of bounding boxes.
        original_shape (tuple, optional): Original image shape for scaling.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: A heatmap of shape target_shape (float32).
    """
    h, w = target_shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Determine scaling factors
    scale_x = 1.0
    scale_y = 1.0

    if original_shape is not None:
        orig_h, orig_w = original_shape
        scale_x = w / orig_w
        scale_y = h / orig_h

    # Pre-compute Gaussian kernel
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    center = size // 2
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    kernel = np.exp(-((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma**2))

    for box in boxes:
        if isinstance(box, dict):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        else:
            bx, by, bw, bh = box[0], box[1], box[2], box[3]

        # Calculate center in target coordinates
        cx = (bx + bw / 2) * scale_x
        cy = (by + bh / 2) * scale_y

        cx_int, cy_int = int(cx), int(cy)

        # Paste kernel
        x1 = max(0, cx_int - center)
        y1 = max(0, cy_int - center)
        x2 = min(w, cx_int + center + 1)
        y2 = min(h, cy_int + center + 1)

        kx1 = center - (cx_int - x1)
        ky1 = center - (cy_int - y1)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            # Use max to handle overlapping characters
            current_val = heatmap[y1:y2, x1:x2]
            kernel_val = kernel[ky1:ky2, kx1:kx2]
            heatmap[y1:y2, x1:x2] = np.maximum(current_val, kernel_val)

    return heatmap
