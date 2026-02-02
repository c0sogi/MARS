import os
import cv2
import numpy as np
from library.config import Config


def load_image(image_path):
    """
    Loads an image from the specified path, converts it to RGB, and resizes it
    to the dimensions specified in Config.IMAGE_SIZE.

    Args:
        image_path (str): The file path to the image.

    Returns:
        numpy.ndarray: The processed image array (H, W, C) in RGB format,
                       or None if the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        return None

    # Load image using OpenCV (loads in BGR format by default)
    img = cv2.imread(image_path)

    if img is None:
        return None

    # Convert from BGR to RGB as expected by most deep learning models (e.g., timm)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the target size defined in Config (e.g., 224x224)
    # INTER_CUBIC is generally preferred for high-quality resizing
    target_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    return img


def generate_rotated_views(image, angles=None):
    """
    Generates rotated versions of the input image based on the specified angles.
    Fills the background with white (255, 255, 255) to match the leaf dataset
    style (black leaves on white background).

    Args:
        image (numpy.ndarray): The input image array (H, W, C).
        angles (list, optional): A list of angles in degrees.
                                 Defaults to Config.ROTATION_ANGLES if None.

    Returns:
        list: A list of numpy.ndarray images corresponding to the rotations.
    """
    if image is None:
        return []

    if angles is None:
        angles = Config.ROTATION_ANGLES

    rotated_images = []
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    for angle in angles:
        # If angle is 0, simply use the original image
        if angle == 0:
            rotated_images.append(image.copy())
            continue

        # Calculate the rotation matrix
        # scale=1.0 ensures the size of the leaf is preserved
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the affine transformation (rotation)
        # borderValue=(255, 255, 255) ensures that the empty space created by
        # rotation is filled with white, matching the dataset background.
        rotated_img = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        rotated_images.append(rotated_img)

    return rotated_images
