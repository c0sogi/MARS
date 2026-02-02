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


def generate_target_mask(target_shape, boxes, original_shape=None):
    """
    Generates a binary segmentation mask from a list of bounding boxes.

    Args:
        target_shape (tuple): The desired output shape (Height, Width).
        boxes (list): A list of bounding boxes. Each item can be a dict
                      (from parse_labels) or a tuple/list [x, y, w, h].
        original_shape (tuple, optional): The (Height, Width) of the original image.
                                          If provided, coordinates in 'boxes' will be scaled
                                          to match 'target_shape'.

    Returns:
        np.ndarray: A binary mask of shape target_shape (uint8), where 1 indicates
                    the presence of a character.
    """
    h, w = target_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Determine scaling factors
    scale_x = 1.0
    scale_y = 1.0

    if original_shape is not None:
        orig_h, orig_w = original_shape
        scale_x = w / orig_w
        scale_y = h / orig_h

    for box in boxes:
        # Extract box coordinates
        if isinstance(box, dict):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        else:
            bx, by, bw, bh = box[0], box[1], box[2], box[3]

        # Apply scaling
        x = bx * scale_x
        y = by * scale_y
        width = bw * scale_x
        height = bh * scale_y

        # Convert to integer coordinates for the mask
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))

        # Clip coordinates to the image boundaries
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        # Draw the box on the mask
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask
