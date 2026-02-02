import os
import cv2
import numpy as np
import pandas as pd
import logging
from library.config import Config
from library.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Implements Moment-Aligned Canonical Anchoring logic.
    """

    def __init__(self, target_size=Config.IMAGE_SIZE):
        self.target_size = target_size
        self.anchor_angles = Config.ANCHOR_ANGLES

    def calculate_principal_axis(self, binary_img):
        """
        Calculates the orientation of the principal axis of the binary object.
        Returns the angle in degrees required to align the axis vertically.

        Args:
            binary_img (np.ndarray): Binary image where object is non-zero.

        Returns:
            float: Angle in degrees.
        """
        moments = cv2.moments(binary_img)
        if moments["m00"] == 0:
            return 0.0

        # Central moments
        mu11 = moments["mu11"]
        mu20 = moments["mu20"]
        mu02 = moments["mu02"]

        # Calculate orientation
        # theta is the angle of the principal axis relative to the x-axis
        theta_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        theta_deg = np.degrees(theta_rad)

        return theta_deg

    def align_image(self, img):
        """
        Aligns the binary leaf image such that its principal axis is vertical.
        Also centers the leaf.

        Args:
            img (np.ndarray): Input image (BGR or Gray).

        Returns:
            np.ndarray: Rotated and centered image.
        """
        # Ensure image is binary (Leaf=White, BG=Black) for moments
        # The dataset description says "black leaves against white backgrounds".
        # We invert for processing.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Threshold to get binary mask (Leaf=255, BG=0)
        # Assuming white background (high values) and black leaf (low values)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Calculate orientation
        angle_deg = self.calculate_principal_axis(thresh)

        # We want to rotate the image so the principal axis is vertical (90 degrees).
        # If the object is at `angle_deg`, we rotate by `90 - angle_deg`.
        rotation_angle = 90 - angle_deg

        # Find centroid for rotation center
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2

        # Rotate
        h, w = img.shape[:2]
        # Calculate new bounding box to avoid cropping
        M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust translation
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        # Warp
        # Fill border with White (255) since original BG is white
        rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))

        return rotated

    def crop_and_resize(self, img):
        """
        Crops the leaf tightly and resizes to target_size with padding (preserving aspect ratio).

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Resized image of shape (target_size, target_size, 3).
        """
        # Invert to find bounding box of the leaf
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback if empty, just resize
            resized = cv2.resize(img, (self.target_size, self.target_size))
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            return resized

        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop
        crop = img[y : y + h, x : x + w]

        # Resize with padding (Letterbox)
        # Create square canvas of target_size (White background)
        canvas = np.full((self.target_size, self.target_size, 3), 255, dtype=np.uint8)

        # Scale factor
        scale = min(self.target_size / w, self.target_size / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center position
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2

        # Paste
        if len(resized_crop.shape) == 2:
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2BGR)

        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_crop

        return canvas

    def get_orthogonal_views(self, aligned_img):
        """
        Generates 4 orthogonal views (0, 90, 180, 270) of the aligned image.

        Args:
            aligned_img (np.ndarray): Image already aligned to vertical axis.

        Returns:
            np.ndarray: Array of shape (4, target_size, target_size, 3).
        """
        views = []

        # Base image (0 degrees)
        # Ensure it's resized/cropped to target size first
        base = self.crop_and_resize(aligned_img)

        h, w = base.shape[:2]
        center = (w // 2, h // 2)

        for angle in self.anchor_angles:
            if angle == 0:
                views.append(base)
            else:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Border value white
                rotated = cv2.warpAffine(base, M, (w, h), borderValue=(255, 255, 255))
                views.append(rotated)

        return np.array(views)

    def process_single_file(self, file_path):
        """
        Full pipeline for a single file.

        Args:
            file_path (str): Relative path to image.

        Returns:
            np.ndarray: Array of shape (4, target_size, target_size, 3).
        """
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return np.zeros((4, self.target_size, self.target_size, 3), dtype=np.uint8)

        img = cv2.imread(full_path)
        if img is None:
            logger.warning(f"Failed to read image: {full_path}")
            return np.zeros((4, self.target_size, self.target_size, 3), dtype=np.uint8)

        aligned = self.align_image(img)
        views = self.get_orthogonal_views(aligned)
        return views


def process_dataset_images(metadata_path, cache_name, load_cached_data=True):
    """
    Processes all images in the metadata file and caches the result.

    Args:
        metadata_path (str): Path to metadata CSV.
        cache_name (str): Identifier for the cache file (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, ids)
            images (np.ndarray): Array of shape (N, 4, H, W, 3) containing processed images.
            ids (np.ndarray): Array of IDs corresponding to the images.
    """
    cache_dir = Config.WORKING_DIR
    img_cache_path = os.path.join(cache_dir, f"{cache_name}_images.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_name}_ids.npy")

    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(id_cache_path)
    ):
        logger.info(f"Loading cached images from {img_cache_path}")
        try:
            images = np.load(img_cache_path)
            ids = np.load(id_cache_path)
            return images, ids
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing...")

    logger.info(f"Processing images for {cache_name} from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    processor = ImageProcessor()

    all_views = []
    all_ids = []

    # Iterate through dataset
    for idx, row in df.iterrows():
        file_path = row["file_path"]
        image_id = row["id"]

        views = processor.process_single_file(file_path)

        all_views.append(views)
        all_ids.append(image_id)

    all_views = np.array(all_views, dtype=np.uint8)
    all_ids = np.array(all_ids)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(img_cache_path, all_views)
    np.save(id_cache_path, all_ids)
    logger.info(f"Saved processed images to {img_cache_path}")

    return all_views, all_ids
