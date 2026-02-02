import os
import random
import numpy as np
import torch
import cv2


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The pixels are numbered from top to bottom, then left to right (Fortran-style).

    Args:
        img (np.ndarray): Binary mask (0 for background, 1 for object).

    Returns:
        str: Space-separated string of start positions and run lengths.
    """
    # Flatten in column-major order (Fortran-style)
    pixels = img.flatten(order="F")

    # Pad with 0 at start and end to detect changes at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): Target shape of the mask (height, width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if mask_rle is None or str(mask_rle) == "nan" or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-based indexing to 0-based
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (Fortran-style)
    return img.reshape(shape, order="F")


def polygons_to_mask(polygons, shape):
    """
    Rasterizes a list of polygons into a binary mask.

    Args:
        polygons (list): List of polygon coordinates. Each entry is a list of rings,
                         where the first ring is the exterior and subsequent rings are holes.
                         Format matches the 'coordinates' field in the dataset JSONs.
        shape (tuple): Target shape of the mask (height, width).

    Returns:
        np.ndarray: Binary mask where the polygon areas are 1 and background is 0.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if not polygons:
        return mask

    for poly_coords in polygons:
        # poly_coords is a list of rings: [exterior, hole1, hole2, ...]
        if not poly_coords:
            continue

        # Draw exterior ring with 1
        exterior = np.array(poly_coords[0], dtype=np.int32)
        cv2.fillPoly(mask, [exterior], 1)

        # Draw holes with 0 (if any)
        if len(poly_coords) > 1:
            holes = [np.array(ring, dtype=np.int32) for ring in poly_coords[1:]]
            cv2.fillPoly(mask, holes, 0)

    return mask


def compute_intersection_union(pred_mask, true_mask):
    """
    Computes the intersection and union sums for a pair of masks.
    Useful for accumulating stats for global Dice calculation.

    Args:
        pred_mask (np.ndarray or torch.Tensor): Predicted binary mask.
        true_mask (np.ndarray or torch.Tensor): Ground truth binary mask.

    Returns:
        tuple: (intersection_sum, union_sum)
    """
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.detach().cpu().numpy()

    # Ensure binary
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)

    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)

    return intersection, union


def calculate_global_dice(intersection_sum, union_sum):
    """
    Calculates the Dice coefficient from accumulated intersection and union values.

    Args:
        intersection_sum (float): Total intersection pixels across the dataset.
        union_sum (float): Total sum of pixels (pred + true) across the dataset.

    Returns:
        float: Dice coefficient. Returns 1.0 if union is 0 (both empty).
    """
    if union_sum == 0:
        return 1.0

    return 2.0 * intersection_sum / union_sum
